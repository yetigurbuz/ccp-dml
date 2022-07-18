import tensorflow as tf

@tf.function
def pairwiseDistance(ref_embeddings, embeddings, squared=False, zero_diagonal=False):

    pairwise_distances_squared = tf.add(
        tf.reduce_sum(tf.square(ref_embeddings), axis=1, keepdims=True),
        tf.expand_dims(tf.reduce_sum(tf.square(embeddings), axis=1), axis=0), # i.e. transpose
    ) - 2.0 * tf.matmul(ref_embeddings, embeddings, transpose_b=True)

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.nn.relu(pairwise_distances_squared)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(pairwise_distances_squared)


    if zero_diagonal:
        num_data = tf.shape(ref_embeddings)[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(tf.ones([num_data]))
        pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)

    return pairwise_distances

@tf.function
def gradPairwiseDistance(d_pdists, x, y, pdists, squared: bool = False):

    if not squared:
        # Get the mask where the zero distances are at. (right derivative of sqrt)
        zero_mask = tf.cast(tf.math.less_equal(pdists, 0.0), dtype=tf.float32)

        d_pdists = tf.divide(
            d_pdists * (1.0 - zero_mask),
            pdists + tf.cast(zero_mask, dtype=tf.float32) * 1e-16) / 2.0


    dx = 2.0 * tf.add(
        tf.multiply(x, tf.reduce_sum(d_pdists, axis=1, keepdims=True)),
        - tf.matmul(d_pdists, y))

    dy = 2.0 * tf.add(
        tf.multiply(y, tf.expand_dims(tf.reduce_sum(d_pdists, axis=0), axis=1)), # inherently transpose
        - tf.matmul(d_pdists, x, transpose_a=True))

    return dx, dy

@tf.function
def getPairwiseDistances(embeddings, ref_embeddings=None, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability.
            output[i, j] = || embeddings[i, :] - ref_embdedings[j, :] ||_2
            Args:
              embeddings: 2-D Tensor of size [number of data, feature dimension].
              ref_embdedings: 2-D Tensor of size [number of data, feature dimension].

              squared: Boolean, whether or not to square the pairwise distances.
            Returns:
              pairwise_distances: 2-D Tensor of size [number of data, number of data].
            """
    zero_diagonal = False
    if ref_embeddings is None:
        ref_embeddings = embeddings
        zero_diagonal = True

    def pairwise_distance(x_1, x_2):
        pdists = pairwiseDistance(x_1, x_2, squared=squared, zero_diagonal=zero_diagonal)
        grad_fn = lambda dy: gradPairwiseDistance(dy, x_1, x_2, pdists, squared=squared)
        return pdists, grad_fn

    pairwise_distance_matrix = tf.custom_gradient(pairwise_distance)(ref_embeddings, embeddings)

    return pairwise_distance_matrix

@tf.function
def cosine_distance(embeddings, ref_embeddings=None, squared=False, **kwargs):
    """Computes the cosine distance matrix.
    output[i, j] = 1 - cosine_similarity(ref_embeddings[i, :], embeddings[j, :])
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      angular_distances: 2-D Tensor of size [number of data, number of data].
    """
    # normalize input

    if ref_embeddings is None:
        ref_embeddings = embeddings

    # create adjaceny matrix of cosine similarity
    cosine_distances = 2. - 2. * tf.matmul(ref_embeddings, embeddings, transpose_b=True)

    # ensure all distances > 1e-16
    cosine_distances = tf.nn.relu(cosine_distances)

    # Optionally take the sqrt.
    if squared:
        cosine_distances = cosine_distances
    else:
        cosine_distances = tf.math.sqrt(cosine_distances)

    return cosine_distances

@tf.function
def neg_cosine_similarity(embeddings, ref_embeddings=None, **kwargs):
    """Computes the negative cosine similarity matrix.
    output[i, j] = - cosine_similarity(ref_embeddings[i, :], embeddings[j, :])
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      angular_distances: 2-D Tensor of size [number of data, number of data].
    """
    # normalize input

    if ref_embeddings is None:
        ref_embeddings = embeddings

    # create adjaceny matrix of cosine similarity
    cosine_similarity = tf.matmul(ref_embeddings, embeddings, transpose_b=True)

    return - cosine_similarity

@tf.function
def getTripletDistances(pairwise_distances,
                        pos_pair_mask,
                        neg_pair_mask,
                        pos_expand_axis=1,
                        neg_expand_axis=2,
                        from_mat=True,
                        pos_per_anchor=None,
                        neg_per_anchor=None):

    if from_mat:
        tdists = (tf.expand_dims(pairwise_distances, axis=pos_expand_axis) -
                  tf.expand_dims(pairwise_distances, axis=neg_expand_axis))

        triplet_mask = pairs2Triplets(pos_pair_mask=pos_pair_mask,
                                      neg_pair_mask=neg_pair_mask,
                                      pos_expand_axis=pos_expand_axis,
                                      neg_expand_axis=neg_expand_axis)
    else:
        """if (pos_per_anchor is None) or (neg_per_anchor is None):
            pos_per_anchor = tf.cast(tf.reduce_sum(pos_pair_mask, axis=1)[0], dtype=tf.int32)
            neg_per_anchor = tf.cast(tf.reduce_sum(neg_pair_mask, axis=1)[0], dtype=tf.int32)
            #raise NotImplementedError('''Forming triplets from pairs is not implemented yet for the most general case.
                    #Only ||sample - proxy(+)|| - ||sample - proxy(-)|| triplets can be formed
                    #for UNMINED pairs (no mining, we know exactly #proxy(-) and #proxy(+) for each sample.''')"""

        # negs
        neg_dists = pairwise_distances * neg_pair_mask

        axis_maximums = tf.math.reduce_max(neg_dists, axis=1, keepdims=True)

        neg_sims = (axis_maximums - neg_dists + .1) * neg_pair_mask

        # poss
        pos_dists = (pairwise_distances + 0.1) * pos_pair_mask

        #axis_maximums = tf.math.reduce_max(pos_dists, axis=1, keepdims=True)

        #pos_sims = (axis_maximums - pos_dists) * pos_pair_mask



        # get the indices of the most difficult <num_neg> number of negatives
        num_neg = tf.maximum(1,
                             tf.cast(tf.reduce_min(tf.reduce_sum(neg_pair_mask, axis=-1)), tf.int32))
        smallest = tf.math.top_k(neg_sims, k=num_neg, sorted=True)[0][:, -1]

        neg_pair_mask = tf.cast(tf.greater_equal(neg_sims, tf.expand_dims(smallest, axis=1)), tf.float32)

        # get the indices of the most difficult <num_pos> number of positives
        num_pos = tf.maximum(1,
                             tf.cast(tf.reduce_min(tf.reduce_sum(pos_pair_mask, axis=-1)), tf.int32))
        smallest = tf.math.top_k(pos_dists, k=num_pos, sorted=True)[0][:, -1]

        pos_pair_mask = tf.cast(tf.greater_equal(pos_dists, tf.expand_dims(smallest, axis=1)), tf.float32)

        #num_anchors = pairwise_distances.shape[0]
        pos_per_anchor = tf.cast(tf.reduce_max(tf.reduce_sum(pos_pair_mask, axis=1)), dtype=tf.int32)
        neg_per_anchor = tf.cast(tf.reduce_max(tf.reduce_sum(neg_pair_mask, axis=1)), dtype=tf.int32)

        '''tf.print('\nnum pos considered:')
        tf.print(num_pos)
        tf.print('\nnum neg considered:')
        tf.print(num_neg)
        tf.print('\nnum pos from:')
        tf.print(pos_per_anchor)
        tf.print('\nnum neg from:')
        tf.print(neg_per_anchor)'''


        ap = tf.where(tf.cast(pos_pair_mask, tf.bool))
        an = tf.where(tf.cast(neg_pair_mask, tf.bool))

        ap_dists = tf.gather_nd(pairwise_distances, ap)
        an_dists = tf.gather_nd(pairwise_distances, an)

        an_dists = tf.repeat(
            tf.reshape(an_dists, shape=(-1, neg_per_anchor)),
                       repeats=[pos_per_anchor], axis=0)

        tdists = tf.expand_dims(ap_dists, axis=1) - an_dists

        triplet_mask = tf.ones_like(tdists)

    return tdists, triplet_mask


@tf.function
def labels2Pairs(labels, ref_labels=None, structured=False, num_classes=None):

    has_self_match = False
    if ref_labels is None:
        ref_labels = labels
        has_self_match = True

    if structured:
        # then batch has elements of c_i1, c_j1, c_k1, ..., c_i2, c_j2, c_k2, ..., ..., c_iN, c_jN, c_kN, ...
        row_repeats = tf.cast(tf.shape(ref_labels)[0] / num_classes, tf.int32)
        col_repeats = tf.cast(tf.shape(labels)[0] / num_classes, tf.int32)
        matched = tf.tile(tf.eye(num_classes), (row_repeats, col_repeats))
    else:
        matched = tf.cast(tf.equal(ref_labels, tf.transpose(labels)), tf.float32)

    neg_pair_mask = 1 - matched
    pos_pair_mask = matched
    if has_self_match:
        pos_pair_mask -= tf.eye(tf.shape(ref_labels)[0])

    return pos_pair_mask, neg_pair_mask

@tf.function
def pairs2Triplets(pos_pair_mask, neg_pair_mask, pos_expand_axis=1, neg_expand_axis=2):
    triplet_mask = (tf.expand_dims(pos_pair_mask, axis=pos_expand_axis) *
                    tf.expand_dims(neg_pair_mask, axis=neg_expand_axis))
    # note that expand axes must be consistent with triplet loss computation
    return triplet_mask

@tf.function
def getPairwiseDistancesAutoDiff(embeddings, ref_embeddings=None, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
        output[i, j] = || embeddings[i, :] - ref_embdedings[j, :] ||_2
        Args:
          embeddings: 2-D Tensor of size [number of data, feature dimension].
          ref_embdedings: 2-D Tensor of size [number of data, feature dimension].

          squared: Boolean, whether or not to square the pairwise distances.
        Returns:
          pairwise_distances: 2-D Tensor of size [number of data, number of data].
        """

    if ref_embeddings is None:
        ref_embeddings = embeddings

    diff = tf.expand_dims(ref_embeddings, axis=1) - tf.expand_dims(embeddings, axis=0)
    pairwise_distances = tf.reduce_sum(tf.square(diff), axis=-1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    pairwise_distances = tf.maximum(pairwise_distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        zero_mask = tf.cast(tf.equal(pairwise_distances, 0.0), tf.float32)
        pairwise_distances = pairwise_distances + zero_mask * 1e-16

        pairwise_distances = tf.sqrt(pairwise_distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        pairwise_distances = pairwise_distances * (1.0 - zero_mask)

    return pairwise_distances


