import tensorflow as tf
from .base_losses import GenericPairBasedLoss

from ..configs import default
from ..configs.config import CfgNode as CN

# proxy nca
proxy_nca_cfg = CN()
proxy_nca_cfg.lambda_softmax = 0.11
proxy_nca_cfg.delta_margin = 0.0
proxy_nca_cfg.proxy_lrm = 100.

default.cfg.loss.proxy_nca = proxy_nca_cfg

# !TODO: update according to new generic class
class ProxyNCALoss(GenericPairBasedLoss):
    def __init__(self, model, num_classes, lambda_softmax=0.11, delta_margin=0.0, **kwargs):

        self.lambda_softmax = 1.0 / lambda_softmax
        self.delta_margin = delta_margin

        if 'use_proxy' in kwargs.keys():
            kwargs['use_proxy'] = True
        else:
            kwargs.update({'use_proxy': True})

        if 'distance_function' in kwargs.keys():
            kwargs['distance_function'] = 'cos'
        else:
            kwargs.update({'distance_function': 'cos'})


        super(ProxyNCALoss, self).__init__(model=model,
                                           num_classes=num_classes,
                                           proxy_name='proxy_nca_loss',
                                           **kwargs)
        self.proxy_anchor = False

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        psims = - pdists

        logits = self.lambda_softmax * (psims - self.delta_margin * pos_pair_mask)

        prob_of_nhood = tf.nn.softmax(logits, axis=-1)

        expected_num_pos = tf.reduce_sum(prob_of_nhood * pos_pair_mask, axis=-1) + 1.e-16

        xent = - tf.math.log(expected_num_pos)

        loss = tf.reduce_mean(xent)


        '''axis_maxs = tf.stop_gradient(tf.reduce_max(psims, axis=0, keepdims=True))

        logits = tf.math.exp(psims - axis_maxs)

        pos_terms = tf.reduce_sum(logits * pos_pair_mask, axis=0) + 1e-16

        neg_terms = tf.reduce_sum(logits * neg_pair_mask, axis=0)

        loss = tf.reduce_mean(tf.math.log1p(neg_terms / pos_terms))'''

        return loss

class proxy_nca(ProxyNCALoss):
    def __init__(self, model, cfg, **kwargs):

        super(proxy_nca, self).__init__(
            model=model,
            classes_per_batch=cfg.training.classes_per_batch,
            num_classes=model.num_classes,
            **cfg.loss.computation_head,
            **cfg.loss.proxy_nca,
            **kwargs
        )