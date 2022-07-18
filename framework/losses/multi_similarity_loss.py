import tensorflow as tf
from .base_losses import GenericPairBasedLoss

from ..configs import default
from ..configs.config import CfgNode as CN

# ms
multi_similarity_cfg = CN()
multi_similarity_cfg.alpha_margin = 2.0
multi_similarity_cfg.beta_margin = 50.
multi_similarity_cfg.lambda_margin = 0.5

default.cfg.loss.multi_similarity = multi_similarity_cfg

class MultiSimilarityLoss(GenericPairBasedLoss):
    def __init__(self, alpha_margin=2., beta_margin=50., lambda_margin=0.5,
                 **kwargs):

        self.alpha = alpha_margin
        self.beta = beta_margin
        self.lambd = lambda_margin

        if 'distance_function' in kwargs.keys():
            kwargs['distance_function'] = 'cos'
        else:
            kwargs.update({'distance_function': 'cos'})

        super(MultiSimilarityLoss, self).__init__(**kwargs)

        self.proxy_anchor = False

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        psims = - pdists

        #pos_exponent = self.alpha * (pdists - self.lambd) * pos_pair_mask
        pos_exponent = self.alpha * (self.lambd - psims) * pos_pair_mask
        #pos_maxs = tf.stop_gradient(tf.reduce_max(pos_exponent, axis=-1, keepdims=True))

        #neg_exponent = self.beta * (self.lambd - pdists) * neg_pair_mask
        neg_exponent = self.beta * (psims - self.lambd) * neg_pair_mask
        #neg_maxs = tf.stop_gradient(tf.reduce_max(neg_exponent, axis=-1, keepdims=True))

        # OLD: Compute the loss:
        ''' loss_p = 1.0 / self.alpha * tf.math.log1p(
            tf.reduce_sum(
                tf.exp(self.alpha * (pdists - self.lambd)) * pos_pair_mask, axis=-1))

        loss_n = 1.0 / self.beta * tf.math.log1p(
            tf.reduce_sum(
                tf.exp(self.beta * (self.lambd - pdists)) * neg_pair_mask, axis=-1))'''

        # Compute the loss:
        loss_p = 1.0 / self.alpha * tf.math.log1p(
            tf.reduce_sum(
                tf.exp(pos_exponent) * pos_pair_mask, axis=-1))

        loss_n = 1.0 / self.beta * tf.math.log1p(
            tf.reduce_sum(
                tf.exp(neg_exponent) * neg_pair_mask, axis=-1))
        '''loss_p = tf.math.log(
            tf.add(
                tf.math.exp(-pos_maxs),
                tf.reduce_sum(
                    tf.math.exp(pos_exponent - pos_maxs) * pos_pair_mask,
                    axis=-1)
            )
        ) / self.alpha

        loss_n = tf.math.log(
            tf.add(
                tf.math.exp(-neg_maxs),
                tf.reduce_sum(
                    tf.exp(neg_exponent - neg_maxs) * neg_pair_mask,
                    axis=-1)
            )
        ) / self.beta'''

        multi_similarity_loss = tf.reduce_mean(loss_p + loss_n)

        '''num_pos = tf.reduce_sum(
            tf.cast(
                tf.cast(
                    tf.reduce_sum(pos_pair_mask, axis=-1),
                    tf.bool),
                tf.float32))
        num_neg = tf.reduce_sum(
            tf.cast(
                tf.cast(
                    tf.reduce_sum(neg_pair_mask, axis=-1),
                    tf.bool),
                tf.float32))

        multi_similarity_loss = tf.reduce_sum(loss_p) / num_neg + tf.reduce_sum(loss_n) / num_neg'''

        return multi_similarity_loss

class multi_similarity(MultiSimilarityLoss):
    def __init__(self, model, cfg, **kwargs):

        super(multi_similarity, self).__init__(
            model=model,
            classes_per_batch=cfg.training.classes_per_batch,
            num_classes=model.num_classes,
            **cfg.loss.computation_head,
            **cfg.loss.multi_similarity,
            **kwargs
        )