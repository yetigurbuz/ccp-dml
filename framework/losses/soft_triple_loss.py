import tensorflow as tf
from .base_losses import GenericPairBasedLoss


from ..configs import default
from ..configs.config import CfgNode as CN

# soft triple
soft_triple_cfg = CN()
soft_triple_cfg.proxy_per_class = 10
soft_triple_cfg.lambda_softmax = 78.02
soft_triple_cfg.gamma_entropy = 58.95
soft_triple_cfg.tau_reg = 0.3754
soft_triple_cfg.delta_margin = 0.4307
soft_triple_cfg.proxy_lrm = 100.

default.cfg.loss.soft_triple = soft_triple_cfg

# !TODO: update according to new generic class
class SoftTripleLoss(GenericPairBasedLoss):
    def __init__(self,
                 model,
                 num_classes,
                 proxy_per_class=10,
                 lambda_softmax=78.02,
                 gamma_entropy=58.95,
                 tau_reg=0.3754,
                 delta_margin=0.4307,
                 proxy_lr=5.37e-04,
                 **kwargs):

        self.lambda_softmax = lambda_softmax
        self.gamma = 1. / gamma_entropy
        self.tau_reg = tau_reg
        self.delta_margin = delta_margin

        self.intra_proxy_dist_mask = tf.expand_dims(
            1. - tf.linalg.band_part(
                tf.ones(shape=(proxy_per_class, proxy_per_class)),
                num_lower=-1, num_upper=0),
            axis=0)

        if 'use_proxy' in kwargs.keys():
            kwargs['use_proxy'] = True
        else:
            kwargs.update({'use_proxy': True})

        super(SoftTripleLoss, self).__init__(model=model,
                                             num_classes=num_classes,
                                             proxy_per_class=proxy_per_class,
                                             proxy_name='softtriple_loss',
                                             **kwargs)

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        pos_pair_mask = pos_pair_mask[:self.num_classes]
        neg_pair_mask = neg_pair_mask[:self.num_classes]

        csims = tf.stack(
            tf.split(-pdists,
                     num_or_size_splits=self.proxy_per_class,
                     axis=0),
            axis=1)

        cprobs = tf.math.softmax(csims * self.gamma, axis=1)

        '''# csims = tf.reduce_sum(cprobs * csims, axis=1) * self.lambda_softmax

        # max_val = tf.stop_gradient(tf.reduce_max(csims, axis=0, keepdims=True))

        # logits = tf.math.exp(csims - max_val)

        pos_terms = tf.reduce_sum(
            logits * tf.math.exp(- self.lambda_softmax * self.delta_margin) * pos_pair_mask,
            axis=0) + 1e-16

        neg_terms = tf.reduce_sum(logits * neg_pair_mask, axis=0)
        soft_triple_loss = tf.reduce_mean(tf.math.log1p(neg_terms / pos_terms))'''

        csims = tf.reduce_sum(cprobs * csims, axis=1)

        logits = self.lambda_softmax * (csims - self.delta_margin * pos_pair_mask)

        soft_triple_loss = tf.reduce_mean(
            - tf.math.log(
                tf.reduce_sum(tf.nn.softmax(logits, axis=0) * pos_pair_mask, axis=0)
            )
        )

        # center loss
        center_loss = 0.0
        if self.proxy_per_class > 1:
            centers = tf.stack(
                tf.split(self.class_representatives[0],
                         num_or_size_splits=self.proxy_per_class,
                         axis=0),
                axis=1)
            intra_proxy_dists = tf.sqrt(
                tf.nn.relu(
                    2. - 2. * tf.reduce_sum(
                        tf.expand_dims(centers, axis=1) * tf.expand_dims(centers, axis=2),
                        axis=-1)
                )
            )
            denom = self.num_classes * self.proxy_per_class * (self.proxy_per_class - 1)
            center_loss = tf.reduce_sum(self.intra_proxy_dist_mask * intra_proxy_dists) / denom

        loss = soft_triple_loss + self.tau_reg * center_loss

        return loss

class soft_triple(SoftTripleLoss):
    def __init__(self, model, cfg, **kwargs):

        super(soft_triple, self).__init__(
            model=model,
            classes_per_batch=cfg.training.classes_per_batch,
            num_classes=model.num_classes,
            **cfg.loss.computation_head,
            **cfg.loss.soft_triple,
            **kwargs
        )