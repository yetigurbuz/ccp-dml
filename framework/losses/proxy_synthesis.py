import tensorflow as tf

from .. import losses
from ..utilities.loss_and_metric_utils import labels2Pairs

from ..configs import default
from ..configs.config import CfgNode as CN

# proxy synthesis
ps_cfg = CN()
ps_cfg.function = 'proxy_nca'
ps_cfg.alpha = 0.4
ps_cfg.mu = 1.0


default.cfg.loss.ps = ps_cfg


def PSLoss(model,
           cfg,
           **kwargs):

    class LossWithPS(getattr(losses, cfg.loss.ps.loss_function)):
        def __init__(self, model, cfg, **kwargs):

            self.ps_alpha = cfg.loss.ps.alpha
            self.ps_mu = cfg.loss.ps.mu

            num_classes = model.num_classes

            classes_per_batch = cfg.training.classes_per_batch
            sample_per_class = cfg.training.sample_per_class
            batch_size = classes_per_batch * sample_per_class

            self.num_lambda_samples = 1  # or batchsize

            self.num_synthetic_samples = round(batch_size * self.ps_mu)

            self.pseudo_labels = tf.expand_dims(
                tf.add(
                    tf.constant(num_classes, dtype=tf.int32),
                    tf.tile(tf.range(classes_per_batch, dtype=tf.int32), multiples=(sample_per_class,))
                )[:self.num_synthetic_samples],
                axis=1
            )

            self.lambda_sampler = tf.compat.v1.distributions.Beta(self.ps_alpha, self.ps_alpha)

            super(LossWithPS, self).__init__(model=model, cfg=cfg, **kwargs)


        def call(self, y_true, y_pred):

            labels, embeddings = y_true, y_pred

            labels = tf.cast(tf.squeeze(labels), tf.int32)
            proxies = tf.concat(self.class_representatives, axis=0)
            proxy_labels = tf.concat(self.proxy_labels, axis=0)

            # normalize
            if self.normalize_embeddings:
                embeddings = self.l2normalization(embeddings)
                proxies = self.l2normalization(proxies)

            # get the proxies with pos sample in the batch
            pos_proxies = tf.gather(proxies, labels)

            # sample lambda rate
            ps_rate = self.lambda_sampler.sample(sample_shape=(self.num_lambda_samples, 1))

            # generate synthetic proxies and corresponding samples
            synthetic_proxies = ps_rate * pos_proxies + (1.0 - ps_rate) * tf.roll(pos_proxies, shift=1, axis=0)
            synthetic_emb = ps_rate * embeddings + (1.0 - ps_rate) * tf.roll(embeddings, shift=1, axis=0)

            # take the ps_mu ratio amount of synthetic examples
            synthetic_proxies = synthetic_proxies[:self.num_synthetic_samples]
            synthetic_emb = synthetic_emb[:self.num_synthetic_samples]

            # normalize again
            if self.normalize_embeddings:
                synthetic_emb = self.l2normalization(synthetic_emb)
                synthetic_proxies = self.l2normalization(synthetic_proxies)

            augmented_embeddings = tf.concat([embeddings, synthetic_emb], axis=0)
            augmented_labels = tf.concat([y_true, self.pseudo_labels], axis=0)
            augmented_proxies = tf.concat([proxies, synthetic_proxies], axis=0)
            augmented_proxy_labels = tf.concat([proxy_labels, self.pseudo_labels], axis=0)


            pdists = self.pdist_fn(embeddings=augmented_embeddings,
                                   ref_embeddings=augmented_proxies,
                                   squared=self.squared_distance)

            pos_pair_mask, neg_pair_mask = labels2Pairs(labels=augmented_labels,
                                                        ref_labels=augmented_proxy_labels,
                                                        structured=False,
                                                        num_classes=None)

            if not self.proxy_anchor:  # then samples are anchors rather than proxies
                pdists = tf.transpose(pdists)
                pos_pair_mask = tf.transpose(pos_pair_mask)
                neg_pair_mask = tf.transpose(neg_pair_mask)
                ref_labels_to_pass = labels

            loss = self._computeLoss(pdists, pos_pair_mask, neg_pair_mask, ref_labels_to_pass)

            '''tf.print('\n-PS-')
            tf.print(tf.squeeze(y_true), summarize=-1)
            tf.print(tf.squeeze(augmented_labels), summarize=-1)'''

            return loss

    return LossWithPS(
        model=model,
        cfg=cfg,
        **kwargs)