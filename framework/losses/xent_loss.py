import tensorflow as tf
from .base_losses import GenericPairBasedLoss
from ..utilities.loss_and_metric_utils import neg_cosine_similarity

# !TODO: update according to new generic class
class XEntLoss(GenericPairBasedLoss):
    def __init__(self, classes_per_batch, lambda_softmax=0.11, **kwargs):

        self.lambda_softmax = 1.0 / lambda_softmax

        if 'use_ref_samples' in kwargs.keys():
            kwargs['use_ref_samples'] = True
        else:
            kwargs.update({'use_ref_samples': True})

        super(XEntLoss, self).__init__(classes_per_batch=classes_per_batch,
                                       **kwargs)

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        psims = - pdists * self.lambda_softmax

        axis_maxs = tf.stop_gradient(tf.reduce_max(psims, axis=0, keepdims=True))

        logits = tf.math.exp(psims - axis_maxs)

        pos_terms = tf.reduce_sum(logits * pos_pair_mask, axis=0) + 1e-16

        neg_terms = tf.reduce_sum(logits * neg_pair_mask, axis=0)

        '''tf.print('\n')
        print_tensor = neg_terms / pos_terms
        tf.print(print_tensor, summarize=-1)'''

        loss = tf.reduce_mean(tf.math.log1p(neg_terms / pos_terms))

        return loss
