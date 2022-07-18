import tensorflow as tf

def nonZeroTerms(loss_terms):
    loss_mask = tf.cast(tf.greater(loss_terms, 0.0), dtype=tf.float32)

    return loss_mask

def hardNegatives(pdists, pos_pair_mask, neg_pair_mask, wrt_representatives=False):
    # assumes structured batch: m-per-class

    neg_dists = pdists * neg_pair_mask

    axis_maximums = tf.math.reduce_max(neg_dists, axis=1, keepdims=True)

    neg_sims = (axis_maximums - neg_dists) * neg_pair_mask

    wrt_representatives = False
    if wrt_representatives:
        dims = tf.shape(pdists)
        num_pos = dims[1] // dims[0]

        # get the indices of the most difficult <num_pos> number of negatives
        smallest = tf.math.top_k(neg_sims, k=num_pos, sorted=True)[0][:, -1]


        neg_pair_mask = tf.cast(tf.greater_equal(neg_sims, tf.expand_dims(smallest, axis=1)), tf.float32)

    else:
        '''tf.print('inputs')
        tf.print(pdists, summarize=-1)
        tf.print('=====')
        tf.print(pos_pair_mask, summarize=-1)
        tf.print('=====')
        tf.print(neg_pair_mask, summarize=-1)'''

        num_pos = tf.maximum(1,
                             tf.cast(tf.reduce_max(tf.reduce_sum(pos_pair_mask, axis=-1)), tf.int32))
        # get the indices of the most difficult <num_pos> number of negatives
        smallest = tf.math.top_k(neg_sims, k=num_pos, sorted=True)[0][:, -1]
        neg_pair_mask = tf.cast(tf.greater_equal(neg_sims, tf.expand_dims(smallest, axis=1)), tf.float32)

        '''sample_per_class = tf.cast(tf.reduce_sum(pos_pair_mask[0]) + 1, tf.int32)
        num_classes = tf.shape(pos_pair_mask)[0] // sample_per_class

        pos_pair_mask = tf.linalg.band_part(pos_pair_mask, 0, -1)

        num_pos = sample_per_class * (sample_per_class - 1) // 2

        merged_shape = tf.concat([tf.expand_dims(num_classes, axis=0),
                                  tf.expand_dims(-1, axis=0)],
                                 axis=0)

        neg_sims = tf.reshape(neg_sims, shape=merged_shape)

        neg_sims = tf.reshape(tf.transpose(neg_sims), shape=merged_shape)

        # get the indices of the most difficult <num_pos> number of negatives
        smallest = tf.math.top_k(neg_sims, k=num_pos, sorted=True)[0][:, -1]

        neg_pair_mask = tf.cast(tf.greater_equal(neg_sims, tf.expand_dims(smallest, axis=1)), tf.float32)

        neg_pair_mask = tf.transpose(neg_pair_mask)

        neg_pair_mask = tf.transpose(tf.reshape(neg_pair_mask, shape=tf.shape(pos_pair_mask)))'''

        '''tf.print('outputs')
        tf.print(pos_pair_mask, summarize=-1)
        tf.print('=====')
        tf.print(neg_pair_mask, summarize=-1)'''

    return pos_pair_mask, neg_pair_mask

def distanceWeighted(pdists, pos_pair_mask, neg_pair_mask, wrt_representatives=False):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4

    ap = tf.where(tf.cast(pos_pair_mask, tf.bool))
    #an = tf.where(tf.cast(neg_pair_mask, tf.bool))

    anchor_idx = tf.cast(ap[:, 0], tf.int64)
    #pos_idx = ap[:, 1]

    d = 128
    dist = tf.maximum(pdists, cut_off)


    log_weight = (2.0 - d) * tf.math.log(dist) - (
            (d - 3.0) / 2.0) * tf.math.log( 1.0 - 0.25 * tf.square(dist))

    weight = tf.exp(log_weight - tf.reduce_max(log_weight))

    weight = weight * neg_pair_mask * tf.cast(dist < nonzero_loss_cutoff, dtype=tf.float32)

    weight_sum = tf.reduce_sum(weight, axis=1, keepdims=True)
    weight = (weight +
              tf.cast(weight_sum == 0, dtype=tf.float32) * neg_pair_mask)

    weight = weight / weight_sum

    weight = tf.gather(weight, anchor_idx)

    neg_idx = tf.squeeze(tf.random.categorical(weight, num_samples=1, dtype=tf.int64))

    vals = tf.ones_like(neg_idx, dtype=tf.float32)
    indcs = tf.concat([tf.expand_dims(anchor_idx, axis=1), tf.expand_dims(neg_idx, axis=1)], axis=1)

    neg_pair_mask = tf.scatter_nd(indcs, vals, (128, 128))
        #tf.sparse.to_dense(tf.SparseTensor(indcs, vals, dense_shape=(128, 128)))

    return pos_pair_mask, neg_pair_mask


