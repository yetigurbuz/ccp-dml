import tensorflow as tf

from .. import losses
from ..utilities.loss_and_metric_utils import labels2Pairs


from ..configs import default
from ..configs.config import CfgNode as CN

# virtual_softmax configs
# ========================
virtual_softmax = CN()
virtual_softmax.function = 'proxy_nca'

default.cfg.loss.virtual_softmax = virtual_softmax

# !TODO: must be converted to class-based implementation such as xbm
def VirtualSoftmaxLoss(model,
                       cfg,
                       **kwargs):

    class LossWithVS(getattr(losses, cfg.loss.ps.loss_function)):
        def __init__(self, model, cfg, **kwargs):
            super(LossWithVS, self).__init__(model=model, cfg=cfg, **kwargs)


        def call(self, y_true, y_pred):

            labels, embeddings = y_true, y_pred

            proxies = tf.concat(self.class_representatives, axis=0)
            proxy_labels = tf.concat(self.proxy_labels, axis=0)

            # normalize
            if self.normalize_embeddings:
                embeddings = self.l2normalization(embeddings)
                proxies = self.l2normalization(proxies)

            pdists = self.pdist_fn(embeddings=embeddings,
                                   ref_embeddings=proxies,
                                   squared=self.squared_distance)



            self_dot = - tf.reduce_sum(tf.square(embeddings), axis=-1, keepdims=True)

            pos_pair_mask, neg_pair_mask = labels2Pairs(labels=labels,
                                                        ref_labels=proxy_labels,
                                                        structured=False,
                                                        num_classes=None)

            if not self.proxy_anchor:  # then samples are anchors rather than proxies
                pdists = tf.transpose(pdists)
                pos_pair_mask = tf.transpose(pos_pair_mask)
                neg_pair_mask = tf.transpose(neg_pair_mask)
                ref_labels_to_pass = labels

            pdists = tf.concat([pdists, self_dot], axis=-1)
            pos_pair_mask = tf.concat([pos_pair_mask, tf.zeros_like(labels, dtype=tf.float32)], axis=-1)
            neg_pair_mask = tf.concat([neg_pair_mask, tf.ones_like(labels, dtype=tf.float32)], axis=-1)

            loss = self._computeLoss(pdists, pos_pair_mask, neg_pair_mask, ref_labels_to_pass)

            '''tf.print('\n-PS-')
            tf.print(tf.squeeze(y_true), summarize=-1)
            tf.print(tf.squeeze(augmented_labels), summarize=-1)'''

            return loss

    return LossWithVS(
        model=model,
        cfg=cfg,
        **kwargs)