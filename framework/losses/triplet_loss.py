import tensorflow as tf
from . import miners
from .base_losses import GenericPairBasedLoss
from ..utilities import loss_and_metric_utils

from ..configs import default
from ..configs.config import CfgNode as CN

# triplet loss
triplet_cfg = CN()
triplet_cfg.margin = 0.05
triplet_cfg.triplets_from_mat = True

default.cfg.loss.triplet = triplet_cfg


class TripletLoss(GenericPairBasedLoss):
    def __init__(self, margin=.05, triplets_from_mat=True, **kwargs):
        self.margin = margin
        #self.proxy_anchor = proxy_anchor
        self.triplets_from_mat = triplets_from_mat

        super(TripletLoss, self).__init__(**kwargs)

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        '''if not self.proxy_anchor:
            # then samples are anchors: ||sample-proxy(+)|| - ||sample-proxy(-)|| + margin
            pdists = tf.transpose(pdists)
            pos_pair_mask = tf.transpose(pos_pair_mask)
            neg_pair_mask = tf.transpose(neg_pair_mask)'''

        # else: ||proxy-sample(+)|| - ||proxy-sample(-)|| + margin

        if self.triplets_from_mat: # memory demanding

            # get triplet distances and triplet mask
            tdist, triplet_mask = loss_and_metric_utils.getTripletDistances(
                pairwise_distances=pdists,
                pos_pair_mask=pos_pair_mask,
                neg_pair_mask=neg_pair_mask,
                pos_expand_axis=1,
                neg_expand_axis=2,
                from_mat=True,
                pos_per_anchor=None,
                neg_per_anchor=None)

            '''# get triplets
            triplet_mask = loss_and_metric_utils.pairs2Triplets(pos_pair_mask, neg_pair_mask)

            # triplet distances
            tdist = loss_and_metric_utils.getTripletDistances(pdists)'''

            # triplet terms
            triplet_terms = (tdist + self.margin) * triplet_mask

        else:
            if self.proxy_anchor or (self.miner is not None):
                raise NotImplementedError(
                    '''Forming triplets from pairs is not implemented yet for the most general case. 
                    Only ||sample - proxy(+)|| - ||sample - proxy(-)|| triplets can be formed 
                    for UNMINED pairs (no mining, we know exactly #proxy(-) and #proxy(+) for each sample.''')

            # else: sample is anchor and proxies are positive and negatives examplars

            # get triplet distances and triplet mask
            tdist, triplet_mask = loss_and_metric_utils.getTripletDistances(
                pairwise_distances=pdists,
                pos_pair_mask=pos_pair_mask,
                neg_pair_mask=neg_pair_mask,
                pos_expand_axis=1,
                neg_expand_axis=2,
                from_mat=False)#,
                #pos_per_anchor=self.proxy_per_class,
                #neg_per_anchor=self.proxy_per_class*(self.num_classes-1))

            # triplet terms
            triplet_terms = (tdist + self.margin) * triplet_mask

        triplet_terms_mask = tf.stop_gradient(miners.nonZeroTerms(triplet_terms))

        if self.avg_nonzero_only:
            num_triplet_terms = tf.reduce_sum(triplet_terms_mask) + 1e-16
            triplet_loss = tf.reduce_sum(triplet_terms * triplet_terms_mask) / num_triplet_terms
        else:
            triplet_loss = tf.reduce_mean(triplet_terms * triplet_terms_mask)

        return triplet_loss

class triplet(TripletLoss):
    def __init__(self, model, cfg, **kwargs):

        super(triplet, self).__init__(
            model=model,
            classes_per_batch=cfg.training.classes_per_batch,
            num_classes=model.num_classes,
            **cfg.loss.computation_head,
            **cfg.loss.triplet,
            **kwargs
        )