import tensorflow as tf
from . import miners
from .base_losses import GenericPairBasedLoss

from ..configs import default
from ..configs.config import CfgNode as CN

# contrastive
contrastive_cfg = CN()
contrastive_cfg.pos_margin = 0.0
contrastive_cfg.neg_margin = 0.5

default.cfg.loss.contrastive = contrastive_cfg

# original single margin contrastive
original_contrastive_cfg = CN()
original_contrastive_cfg.margin = 0.5
default.cfg.loss.original_contrastive = original_contrastive_cfg

class ContrastiveLoss(GenericPairBasedLoss):
    def __init__(self, pos_margin=0., neg_margin=1., **kwargs):

        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

        super(ContrastiveLoss, self).__init__(**kwargs)

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        pos_term = (pdists - self.pos_margin) * pos_pair_mask
        neg_term = (self.neg_margin - pdists) * neg_pair_mask

        pos_term_mask = tf.stop_gradient(miners.nonZeroTerms(pos_term))
        neg_term_mask = tf.stop_gradient(miners.nonZeroTerms(neg_term))

        if self.avg_nonzero_only:
            num_pos_term = tf.reduce_sum(pos_term_mask) + 1e-16
            num_neg_term = tf.reduce_sum(neg_term_mask) + 1e-16
            num_terms = num_pos_term + num_neg_term + 1e-16

            contrastive_loss = tf.add(tf.reduce_sum(pos_term * pos_term_mask) / num_pos_term,
                                      tf.reduce_sum(neg_term * neg_term_mask) / num_neg_term)
        else:
            contrastive_loss = tf.add(tf.reduce_mean(pos_term * pos_term_mask),
                                      tf.reduce_mean(neg_term * neg_term_mask))

        return contrastive_loss


class contrastive(ContrastiveLoss):
    def __init__(self, model, cfg, **kwargs):

        super(contrastive, self).__init__(
            model=model,
            num_classes=model.num_classes,
            classes_per_batch=cfg.training.classes_per_batch,
            **cfg.loss.computation_head,
            **cfg.loss.contrastive,
            **kwargs
        )


class OriginalContrastiveLoss(GenericPairBasedLoss):
    def __init__(self, margin=0.5, **kwargs):


        self.margin = margin
        self.eps = 1e-5

        if 'distance_function' in kwargs.keys():
            kwargs['distance_function'] = 'cos'
        else:
            kwargs.update({'distance_function': 'cos'})

        super(OriginalContrastiveLoss, self).__init__(**kwargs)


    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        n = tf.reduce_sum(tf.ones_like(pdists), axis=0)[0]

        psims = - pdists

        pos_term = (1.0 + pdists - self.eps) * pos_pair_mask
        neg_term = (psims - self.margin) * neg_pair_mask

        pos_term_mask = tf.stop_gradient(miners.nonZeroTerms(pos_term))
        neg_term_mask = tf.stop_gradient(miners.nonZeroTerms(neg_term))

        contrastive_loss = tf.add(tf.reduce_sum(pos_term * pos_term_mask),
                                  tf.reduce_sum(neg_term * neg_term_mask)) / n


        return contrastive_loss

class original_contrastive(OriginalContrastiveLoss):
    def __init__(self, model, cfg, **kwargs):

        super(original_contrastive, self).__init__(
            model=model,
            classes_per_batch=cfg.training.classes_per_batch,
            num_classes=model.num_classes,
            **cfg.loss.computation_head,
            **cfg.loss.original_contrastive,
            **kwargs
        )

def fromTuples(anc, pos, neg, pos_margin=0., neg_margin=.5):

    d_ap = tf.math.reduce_euclidean_norm(anc - pos, axis=-1)
    d_an = tf.math.reduce_euclidean_norm(anc - neg, axis=-1)

    pos_term = d_ap - pos_margin
    neg_term = neg_margin - d_an

    pos_term_mask = miners.nonZeroTerms(pos_term)
    neg_term_mask = miners.nonZeroTerms(neg_term)

    num_pos_term = tf.reduce_sum(pos_term_mask)
    num_neg_term = tf.reduce_sum(neg_term_mask)
    num_terms = num_pos_term + num_neg_term + 1e-16

    contrastive_loss = (tf.reduce_sum(pos_term * pos_term_mask) +
                        tf.reduce_sum(neg_term * neg_term_mask) / num_terms)

    return contrastive_loss
