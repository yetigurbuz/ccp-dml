import tensorflow as tf
from ..utilities.loss_and_metric_utils import getPairwiseDistances

@tf.function
def mAP(embeddings, labels, atK, ref_embeddings=None, ref_labels=None, has_self_match=True):

    if ref_embeddings is None:
        ref_embeddings = embeddings
        ref_labels = labels
        has_self_match = True

    max_k = atK[-1]
    total_queries = tf.cast(ref_labels.get_shape().as_list()[0], tf.float32)

    pdists = getPairwiseDistances(embeddings, ref_embeddings)
    pmatches = tf.cast(tf.equal(ref_labels, tf.transpose(labels)), tf.float32)
    total_positives = tf.reduce_sum(pmatches, axis=1, keepdims=True)

    if has_self_match:
        total_positives -= 1 # -1 for self match
        max_k += 1 # +1 for self retrieval

    total_positives = tf.minimum(total_positives, tf.expand_dims(tf.cast(atK, tf.float32), axis=0))

    _, indices = tf.math.top_k(-pdists, k=max_k, sorted=True)

    #sorted_labels = tf.gather_nd(tf.squeeze(labels), tf.expand_dims(indices, axis=-1))
    #pmatches = tf.cast(tf.equal(ref_labels, sorted_labels)[:, 1:], tf.float32)

    pmatches = tf.gather_nd(pmatches, tf.expand_dims(indices, axis=-1), batch_dims=1)

    if has_self_match:
        pmatches = pmatches[:, 1:] # ignore self retrieval

    hits = tf.cumsum(pmatches, axis=1)
    prec = hits / tf.cast(tf.expand_dims(tf.range(1, max_k), axis=0), dtype=tf.float32)

    avg_prec = tf.cumsum(prec * pmatches, axis=1)

    #if has_self_match:
    atK -= 1

    avg_prec_at_k = tf.gather(avg_prec, atK, axis=1) / total_positives

    mean_avg_prec = tf.reduce_sum(avg_prec_at_k, axis=0) / total_queries

    return mean_avg_prec

@tf.function
def atKMetrics(embeddings, labels, atK, ref_embeddings=None, ref_labels=None, has_self_match=True):


    if ref_embeddings is None:
        ref_embeddings = embeddings
        ref_labels = labels
        has_self_match = True

    max_k = atK[-1]
    total_queries = tf.cast(ref_labels.get_shape().as_list()[0], tf.float32)

    pdists = getPairwiseDistances(embeddings, ref_embeddings)
    pmatches = tf.cast(tf.equal(ref_labels, tf.transpose(labels)), tf.float32)
    total_positives = tf.reduce_sum(pmatches, axis=1, keepdims=True)
    if has_self_match:
        total_positives -= 1 # -1 for self match
        max_k += 1 # +1 for self retrieval

    # check whether queries with zero positives exist
    invalid_mask = tf.squeeze(tf.equal(total_positives, tf.constant(0., tf.float32)))

    # remove queries with zero positives
    pdists = tf.boolean_mask(pdists, tf.math.logical_not(invalid_mask))
    pmatches = tf.boolean_mask(pmatches, tf.math.logical_not(invalid_mask))
    total_positives = tf.boolean_mask(total_positives, tf.math.logical_not(invalid_mask))

    # update total queries
    total_queries -= tf.reduce_sum(tf.cast(invalid_mask, tf.float32))


    max_total = tf.cast(tf.reduce_max(total_positives), tf.int32)

    max_k = tf.cond(tf.greater(max_total, max_k),
                    true_fn=lambda: max_total + 1 * tf.cast(has_self_match, tf.int32),
                    false_fn=lambda: tf.cast(max_k, tf.int32))


    total_positives_at_k = tf.minimum(total_positives, tf.expand_dims(tf.cast(atK, tf.float32), axis=0))

    _, indices = tf.math.top_k(-pdists, k=max_k, sorted=True)

    #sorted_labels = tf.gather_nd(tf.squeeze(labels), tf.expand_dims(indices, axis=-1))
    #pmatches = tf.cast(tf.equal(ref_labels, sorted_labels)[:, 1:], tf.float32)

    pmatches = tf.gather_nd(pmatches, tf.expand_dims(indices, axis=-1), batch_dims=1)

    if has_self_match:
        pmatches = pmatches[:, 1:]  # ignore self retrieval
        max_k -= 1

    hits = tf.cumsum(pmatches, axis=1)
    prec = hits / tf.cast(tf.expand_dims(tf.range(1, max_k + 1), axis=0), dtype=tf.float32)
    rec = tf.cast(tf.greater(hits, 0), dtype=tf.float32)

    avg_prec = tf.cumsum(prec * pmatches, axis=1)

    #if has_self_match:
    atK -= 1

    # MAP
    avg_prec_at_k = tf.gather(avg_prec, atK, axis=1) / total_positives_at_k
    avg_prec_at_pN = tf.gather_nd(avg_prec,
                                  tf.expand_dims(tf.cast(total_positives, tf.int32) - 1, axis=-1),
                                  batch_dims=1) / total_positives
    mean_avg_prec_at_k = tf.reduce_sum(tf.concat([avg_prec_at_k, avg_prec_at_pN], axis=1), axis=0) / total_queries

    # precision
    prec_at_k = tf.gather(prec, atK, axis=1)
    prec_at_pN = tf.gather_nd(prec,
                              tf.expand_dims(tf.cast(total_positives, tf.int32) - 1, axis=-1),
                              batch_dims=1)
    mean_prec_at_k = tf.reduce_sum(tf.concat([prec_at_k, prec_at_pN], axis=1), axis=0) / total_queries


    # recall
    rec_at_k = tf.gather(rec, atK, axis=1)
    rec_at_pN = tf.gather_nd(rec,
                             tf.expand_dims(tf.cast(total_positives, tf.int32) - 1, axis=-1),
                             batch_dims=1)
    mean_rec_at_k = tf.reduce_sum(tf.concat([rec_at_k, rec_at_pN], axis=1), axis=0) / total_queries


    return tf.concat([mean_avg_prec_at_k, mean_prec_at_k, mean_rec_at_k], axis=0)