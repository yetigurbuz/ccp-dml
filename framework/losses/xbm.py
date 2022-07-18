import tensorflow as tf

from .. import losses
from ..utilities.loss_and_metric_utils import labels2Pairs


from ..configs import default
from ..configs.config import CfgNode as CN

# xbm loss
xbm = CN()
xbm.function = 'original_contrastive'
xbm.batches_in_mem = 128
xbm.start_at = 1000
xbm.xbm_weight = 1.0
xbm.pair_loss_weight = 1.0

default.cfg.loss.xbm = xbm

class XBMLoss(tf.keras.losses.Loss):

    def __init__(self, model, cfg, **kwargs):
        if 'name' in kwargs.keys():
            name = kwargs['name']
        else:
            name = 'asap'

        super(XBMLoss, self).__init__(reduction=tf.keras.losses.Reduction.NONE, name=name)

        class LossWithXBM(getattr(losses, cfg.loss.xbm.function)):
            def __init__(self, model, cfg, **kwargs):

                self.embedding_size = cfg.model.embedding_head.embedding_size
                self.batch_size = cfg.training.classes_per_batch * cfg.training.sample_per_class
                self.batches_in_mem = cfg.loss.xbm.batches_in_mem

                # after start_at iterations, fill xbm-memory for batches_in_mem iterations
                self.start_at = cfg.loss.xbm.start_at + self.batches_in_mem

                self.xbm_proxy = model.arch.add_weight(
                    name='xbm/proxy_embedding',
                    shape=(self.batches_in_mem, self.batch_size, self.embedding_size),
                    dtype=tf.float32,
                    initializer=tf.keras.initializers.zeros,
                    trainable=False)

                self.xbm_label = model.arch.add_weight(
                    name='xbm/proxy_label',
                    shape=(self.batches_in_mem, self.batch_size, 1),
                    dtype=tf.int32,
                    initializer=tf.keras.initializers.zeros,
                    trainable=False)

                self.xbm_ptr = model.arch.add_weight(
                    name='xbm/pointer',
                    shape=(1, 1),
                    dtype=tf.int64,
                    initializer=tf.keras.initializers.zeros,
                    trainable=False)

                # The variable is used to check whether queue is initialized
                self.ready = tf.Variable(
                    initial_value=tf.constant(False, tf.bool),
                    trainable=False, shape=(), dtype=tf.bool)

                self.xbm_weight = cfg.loss.xbm.xbm_weight
                self.pair_loss_weight = cfg.loss.xbm.pair_loss_weight

                super(LossWithXBM, self).__init__(model=model, cfg=cfg, **kwargs)

            def enqueueDequeue(self, embeddings, labels):

                indx = tf.math.mod(self.xbm_ptr, self.batches_in_mem)

                self.xbm_proxy.scatter_nd_update(
                    indx,
                    tf.expand_dims(embeddings, axis=0)
                )

                self.xbm_label.scatter_nd_update(
                    indx,
                    tf.expand_dims(labels, axis=0)
                )

                self.xbm_ptr.assign_add(tf.ones(shape=(1, 1), dtype=tf.int64))

                xbm_embs = tf.reshape(
                    self.xbm_proxy,
                    shape=(self.batch_size * self.batches_in_mem, self.embedding_size))

                xbm_lbls = tf.reshape(
                    self.xbm_label,
                    shape=(self.batch_size * self.batches_in_mem, 1))

                self.ready.assign(tf.squeeze(tf.greater_equal(self.xbm_ptr, self.start_at)))

                return xbm_embs, xbm_lbls


            def call(self, y_true, y_pred):

                labels, embeddings = y_true, y_pred

                xbm_embs, xbm_lbls = self.enqueueDequeue(embeddings, labels)

                if self.normalize_embeddings:
                    embeddings = self.l2normalization(embeddings)

                pdists = self.pdist_fn(embeddings=embeddings,
                                       ref_embeddings=None,
                                       squared=self.squared_distance)

                pos_pair_mask, neg_pair_mask = labels2Pairs(labels=labels,
                                                            ref_labels=None,
                                                            structured=self.structured_batch,
                                                            num_classes=self.classes_per_batch)

                loss0 = self._computeLoss(pdists, pos_pair_mask, neg_pair_mask, labels)

                xbm_loss = tf.cond(
                    pred=tf.equal(self.ready, False),
                    true_fn=lambda: 0.0,
                    false_fn=lambda: super(LossWithXBM, self).call(y_true, y_pred))

                '''tf.print('\n-xbm-')
                tf.print(tf.squeeze(y_true), summarize=-1)
                tf.print(tf.squeeze(xbm_lbls), summarize=-1)
                tf.print('ready')
                tf.print(self.ready)
                tf.print('loss')
                tf.print(xbm_loss)'''

                return self.pair_loss_weight * loss0 + self.xbm_weight * xbm_loss

        self.loss_fn = LossWithXBM(model=model,
                                   cfg=cfg,
                                   **kwargs)


    def call(self, y_true, y_pred):

        return self.loss_fn.call(y_true, y_pred)
