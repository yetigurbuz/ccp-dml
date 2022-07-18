import tensorflow as tf

from .base_losses import GenericPairBasedLoss

from ..configs import default
from ..configs.config import CfgNode as CN

# ms
# proxy anchor
proxy_anchor_cfg = CN()
proxy_anchor_cfg.delta_margin = 0.1
proxy_anchor_cfg.alpha_scale = 32.0
proxy_anchor_cfg.proxy_lrm = 100.

default.cfg.loss.proxy_anchor = proxy_anchor_cfg

class ProxyAnchorLoss(GenericPairBasedLoss):
    def __init__(self, model, num_classes, delta_margin=0.1, alpha_scale=32., **kwargs):

        self.delta = delta_margin
        self.alpha = alpha_scale

        if 'use_proxy' in kwargs.keys():
            kwargs['use_proxy'] = True
        else:
            kwargs.update({'use_proxy': True})

        if 'distance_function' in kwargs.keys():
            kwargs['distance_function'] = 'cos'
        else:
            kwargs.update({'distance_function': 'cos'})

        super(ProxyAnchorLoss, self).__init__(model=model,
                                              num_classes=num_classes,
                                              proxy_name='proxyanchor_loss',
                                              **kwargs)

        self.proxy_anchor = True

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        psims = - pdists

        pos_exponent = self.alpha * (self.delta - psims)
        neg_exponent = self.alpha * (self.delta + psims)

        '''pos_term = tf.math.log1p(
            tf.reduce_sum(
                tf.exp(pos_exponent) * pos_pair_mask, axis=-1))

        neg_term = tf.math.log1p(
            tf.reduce_sum(
                tf.exp(neg_exponent) * neg_pair_mask, axis=-1))'''

        p_sim_sum = tf.reduce_sum(
            tf.where(pos_pair_mask > 0, tf.exp(pos_exponent), tf.zeros_like(pos_exponent)),
            axis=-1)
        n_sim_sum = tf.reduce_sum(
            tf.where(neg_pair_mask > 0, tf.exp(neg_exponent), tf.zeros_like(neg_exponent)),
            axis=-1)

        pos_term = tf.math.log1p(p_sim_sum)
        neg_term = tf.math.log1p(n_sim_sum)

        num_pos = tf.reduce_sum(
            tf.cast(
                tf.cast(
                    tf.reduce_sum(pos_pair_mask, axis=-1),
                    tf.bool),
                tf.float32))
        num_pos = tf.stop_gradient(num_pos)

        num_neg = tf.reduce_sum(
            tf.cast(
                tf.cast(
                    tf.reduce_sum(neg_pair_mask, axis=-1),
                    tf.bool),
                tf.float32))#self.num_classes
        num_neg = tf.stop_gradient(num_neg)

        proxy_anchor_loss = tf.reduce_sum(pos_term) / num_pos + tf.reduce_sum(neg_term) / num_neg

        return proxy_anchor_loss

class proxy_anchor(ProxyAnchorLoss):
    def __init__(self, model, cfg, **kwargs):

        super(proxy_anchor, self).__init__(
            model=model,
            classes_per_batch=cfg.training.classes_per_batch,
            num_classes=model.num_classes,
            **cfg.loss.computation_head,
            **cfg.loss.proxy_anchor,
            **kwargs
        )