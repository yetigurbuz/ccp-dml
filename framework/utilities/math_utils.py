import tensorflow as tf

def broadcastMatMul(a, b, transpose_a=False, transpose_b=False):
    '''feat_axis is after transpose
    for example if shape(a) = (batch, widht, height, feat_size) then its feat_axis_a is -1'''


    a_shape = a.get_shape().as_list()
    b_shape = b.get_shape().as_list()

    rank_a = len(a_shape)
    rank_b = len(b_shape)

    if rank_a - rank_b == 1:

        if rank_b < 2:
            raise ValueError('Min rank must be 2.')

        if rank_a > 5:
            raise ValueError('Max supported feature dims is 3 but got {}. Suggestion: Reduce feature dims by flattening.'.format(rank_a - 2))

        if transpose_a:
            a = tf.transpose(a, perm=([0] + [rank_a - 1] + list(range(1, rank_a - 1))))

        if transpose_b:
            b = tf.transpose(b, perm=(list(range(1, rank_b)) + [0]))

        a_shape = a.get_shape().as_list()
        b_shape = b.get_shape().as_list()

        if not (a_shape[2:] == b_shape[:-1]):
            raise ValueError('Feature shapes do not match: a: {} vs b: {}'.format(a_shape[2:], b_shape[:-1]))

        prod = tf.nn.convolution(a, tf.expand_dims(b, axis=0))

        if rank_b > 2:
            prod = tf.squeeze(prod, list(range(2, rank_b)))

        return prod

    elif rank_b - rank_a == 1:

        if rank_a < 2:
            raise ValueError('Min rank must be 2.')

        if rank_b > 5:
            raise ValueError('Max supported feature dims is 3 but got {}. Suggestion: Reduce feature dims by flattening.'.format(rank_b - 2))

        if not transpose_a:
            a = tf.transpose(a, perm=(list(range(1, rank_a)) + [0]))

        if not transpose_b:
            b = tf.transpose(b, perm=([0] + [rank_b - 1] + list(range(1, rank_b - 1))))

        a_shape = a.get_shape().as_list()
        b_shape = b.get_shape().as_list()

        if not (b_shape[2:] == a_shape[:-1]):
            raise ValueError('feature shapes do not match: b: {} vs a: {}'.format(b_shape[2:], a_shape[:-1]))

        prod = tf.nn.convolution(b, tf.expand_dims(a, axis=0))
        if rank_a > 2:
            prod = tf.squeeze(prod, list(range(2, rank_a)))

        return tf.transpose(prod, perm=(0, 2, 1))

    elif rank_a == rank_b:

        if rank_a == 2:
            prod = tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
            return prod

        else:
            if rank_a > 4:
                raise ValueError('Max supported feature dims is 3 but got {}. Suggestion: Reduce feature dims by flattening.'.format(rank_a - 1))

            if transpose_a:
                a = tf.transpose(a, perm=([rank_a - 1] + list(range(0, rank_a - 1))))
            if transpose_b:
                b = tf.transpose(b, perm=(list(range(1, rank_b)) + [0]))

            a_shape = a.get_shape().as_list()
            b_shape = b.get_shape().as_list()

            if not (a_shape[1:] == b_shape[:-1]):
                raise ValueError('feature shapes do not match: a: {} vs b: {}'.format(a_shape[1:], b_shape[:-1]))

            prod = tf.nn.convolution(tf.expand_dims(a, axis=0), tf.expand_dims(b, axis=0))
            if rank_b > 2:
                prod = tf.squeeze(prod, [0] + list(range(2, rank_b)))

            return prod
    else:
        raise ValueError('Either rank difference must be 1 or both must be of the same rank.')

    '''if not (len(b_shape) == 2):
        raise ValueError('b must be of rank-2 when a is of rank-3.')
    if not (len(a_shape) == 2):
        raise ValueError('a must be of rank-2 when b is of rank-3.')'''


@tf.function
def broadcastPairwiseDistance(x, y, squared=False):
    ''' assuming x has batch of M vectors of D-dim and y has N vectors of the same dimension
     returns pdist_kij = ||x_ki - y_j||_2 or similarly if y has batch of N vectors
     returns  pdist_kij = ||x_i - y_kj||_2'''

    rank_x = len(x.get_shape().as_list())
    rank_y = len(y.get_shape().as_list())


    # self terms
    x_T_x = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    y_T_y = tf.reduce_sum(tf.square(y), axis=-1)

    # cross terms
    if rank_x > rank_y:
        y_T_y = tf.reshape(y_T_y, shape=(1, 1, -1))
    elif rank_y > rank_x:
        y_T_y = tf.expand_dims(y_T_y, axis=1)
        x_T_x = tf.expand_dims(x_T_x, axis=0)
    else:
        y_T_y = tf.expand_dims(y_T_y, axis=0)

    x_T_y = broadcastMatMul(x, y, transpose_b=True)

    pairwise_distances_squared = x_T_x - 2.0 * x_T_y + y_T_y

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.nn.relu(pairwise_distances_squared)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(pairwise_distances_squared)

    return pairwise_distances

@tf.function
def gradPairwiseDistance(d_pdists, x, y, pdists, squared: bool = False):

    rank_x = len(x.get_shape().as_list())
    rank_y = len(y.get_shape().as_list())
    rank_pdists = len(pdists.get_shape().as_list())


    # dsqrt compute path
    if not squared:
        # Get the mask where the zero distances are at. (right derivative of sqrt)
        zero_mask = tf.cast(tf.math.less_equal(pdists, 0.0), dtype=tf.float32)

        d_pdists = tf.divide(
            d_pdists * (1.0 - zero_mask),
            pdists + tf.cast(zero_mask, dtype=tf.float32) * 1e-16) / 2.0


    col_accumulated = tf.reduce_sum(d_pdists, axis=-1, keepdims=True)
    row_accumulated = tf.expand_dims(tf.reduce_sum(d_pdists, axis=-2), axis=-1)  # inherently transpose
    if rank_x < rank_y:
        col_accumulated = tf.reduce_sum(col_accumulated, axis=0)
        grad_by_y = tf.reduce_sum(tf.matmul(d_pdists, y), axis=0)
        grad_by_x = broadcastMatMul(d_pdists, x, transpose_a=True)
    elif rank_y < rank_x:
        row_accumulated = tf.reduce_sum(d_pdists, axis=0)
        grad_by_y = broadcastMatMul(d_pdists, y)
        grad_by_x = tf.reduce_sum(tf.matmul(d_pdists, x, transpose_a=True), axis=0)
    else:
        grad_by_y = tf.matmul(d_pdists, y)
        grad_by_x = tf.matmul(d_pdists, x, transpose_a=True)

    dx = 2.0 * tf.add(
        tf.multiply(x, col_accumulated),
        - grad_by_y)

    dy = 2.0 * tf.add(
        tf.multiply(y, row_accumulated),
        - grad_by_x)

    return dx, dy

@tf.function
def pairwiseL2Distance(x, y, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability.
    assuming x has batch of M vectors of D-dim and y has N vectors of the same dimension
     returns pdist_kij = ||x_ki - y_j||_2
     or similarly if y has batch of N vectors
     returns  pdist_kij = ||x_i - y_kj||_2
            Args:
              x: 2(3)-D Tensor of size [(batch size), number of data, feature dimension].
              y: 2(3)-D Tensor of size [(batch size), number of data, feature dimension].

              squared: Boolean, whether or not to square the pairwise distances.
            Returns:
              pairwise_distances: 3-D Tensor of size [batch size, number of data, number of data].

    This function also supports when x is of shape=(batch, height, width, feature_size)
     and y is of shape=(num_features, feature_size) or vice-versa
     NOTE: we always assume both x and y has the last axis as feature axis i.e., -1
            """
    rank_x = len(x.get_shape().as_list())
    rank_y = len(y.get_shape().as_list())

    shape_x = tf.shape(x)
    shape_y = tf.shape(y)

    out_shape = None

    if rank_x > rank_y:
        if rank_x > 3:
            x = tf.reshape(x, shape=(-1, shape_x[-1]))
            if rank_y > 2:
                y = tf.reshape(y, shape=(-1, shape_y[-1]))
                out_shape = tf.concat([shape_x[:-1], shape_y[:-1]], axis=0)
            else:
                out_shape = tf.concat([shape_x[:-1], [shape_y[0]]], axis=0)

    elif rank_y > rank_x:
        # we assume first dimension of the greater rank tensor is the batch dimension
        if rank_y > 3:
            y = tf.reshape(y, shape=(shape_y[0], -1, shape_y[-1]))
            if rank_x > 2:
                x = tf.reshape(x, shape=(-1, shape_x[-1]))
                out_shape = tf.concat([[shape_y[0]], shape_x[:-1], shape_y[1:-1]], axis=0)
            else:
                out_shape = tf.concat([[shape_y[0], shape_x[0]], shape_y[1:-1]], axis=0)
    else:
        if rank_x > 2:
            x = tf.reshape(x, shape=(-1, shape_x[-1]))
            y = tf.reshape(y, shape=(-1, shape_y[-1]))
            out_shape = tf.concat([shape_x[:-1], shape_y[:-1]], axis=0)

    def pairwise_distance(x_1, x_2):
        pdists = broadcastPairwiseDistance(x_1, x_2, squared=squared)
        grad_fn = lambda dy: gradPairwiseDistance(dy, x_1, x_2, pdists, squared=squared)
        return pdists, grad_fn

    pairwise_distance_matrix = tf.custom_gradient(pairwise_distance)(x, y)

    if out_shape is not None:
        pairwise_distance_matrix = tf.reshape(pairwise_distance_matrix, shape=out_shape)

    return pairwise_distance_matrix

"""    if rank_x > 4:
        raise ValueError(
            'broadcastPairwiseDistance function got rank {} input as x. Batch of 1D or 2D vectors are supported for broadcastPairwiseDistance.'.format(
                rank_x))
    if rank_y > 4:
        raise ValueError(
            'broadcastPairwiseDistance function got rank {} input as y. Batch of 1D or 2D vectors are supported for broadcastPairwiseDistance.'.format(
                rank_y))

    if (rank_x > 2) and (rank_y > 2):
        raise ValueError(
            '''broadcastPairwiseDistance function got rank {} and {} inputs as x and y.
            Batch of 1D or 2D vectors vs N-many vectors are supported for broadcastPairwiseDistance.
            Namely, (batch, width, height, feat_dim) vs (num_feat, feat_dim) or 
            (batch, num_feat_1, feat_dim) vs (num_feat_2, feat_dim) or
            (num_feat_1, feat_dim) vs (num_feat_2, feat_dim) cases are supported for now.'''.format(rank_x, rank_y))"""






'''x = 0.00001*tf.random.uniform(shape=(3, 5, 8))
w_b = 0.00001*tf.random.uniform(shape=(4, 8))


# x.shape should be [B, 1, N, D]
# self._background_support.shape should be [1, M, 1, D]
# !TODO: implement broadcast version for memory efficiency (now consume: BxMxNxD)
with tf.GradientTape() as t:
    t.watch((x, w_b))
    c_old = tf.math.reduce_euclidean_norm(tf.expand_dims(x, axis=1) - tf.expand_dims(tf.expand_dims(w_b, axis=1), axis=0), axis=-1)
    loss = tf.reduce_sum(c_old)

dx_tf, dw_tf = t.gradient(loss, (x, w_b))

with tf.GradientTape() as t:
    t.watch((x, w_b))
    c_bcast = pairwiseL2Distance(w_b, x)
    loss = tf.reduce_sum(c_bcast)

dx, dw = t.gradient(loss, (x, w_b))

#c_bcast = broadcastPairwiseDistances(w_b, x)

#dc = tf.ones_like(c_bcast)

#dw, dx = gradPairwiseDistances(tf.ones_like(c_bcast), w_b, x, c_bcast)

print('max error rate dists: {}'.format(tf.reduce_max(tf.abs(c_old - c_bcast) / c_old)))
print('max error rate dx: {}'.format(tf.reduce_max(tf.abs(dx - dx_tf) / dx)))
print('max error rate dw: {}'.format(tf.reduce_max(tf.abs(dw - dw_tf) / dw)))'''

'''import numpy as np
null_space_dim = 3
sol_space_dim = 4

B_ = tf.convert_to_tensor(np.arange(24).reshape(2,3,2,2) , dtype=tf.float32)
B = tf.convert_to_tensor(np.random.randint(1, 5, size=(1,3,2,2)), dtype=tf.float32)

H = broadcastMatMul(B_, tf.squeeze(B), transpose_b=True)

H_old = tf.reduce_sum(tf.expand_dims(B, axis=1) * tf.expand_dims(B_, axis=2), axis=[3, 4])

dy = tf.convert_to_tensor(np.arange(3 * 4).reshape(3,2,2), dtype=tf.float32)

dz = broadcastMatMul(a=tf.squeeze(B),
                     b=tf.expand_dims(dy, axis=-1))

dz_old = tf.expand_dims(tf.reduce_sum(B * tf.expand_dims(dy, axis=1), axis=(2, 3)), axis=2)'''
