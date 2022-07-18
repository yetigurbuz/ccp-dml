import tensorflow as tf

class FromTensorInitializer(tf.keras.initializers.Initializer):

  def __init__(self, tensor):
    self.tensor = tensor

  def __call__(self, shape, dtype=None, **kwargs):
    return tf.constant(value=self.tensor, shape=shape, dtype=dtype)

  def get_config(self):  # To support serialization
    return {"tensor": self.tensor}

class ProxyLabelInitializer(tf.keras.initializers.Initializer):

  def __init__(self, proxy_per_class):
    self.proxy_per_class = proxy_per_class

  def __call__(self, shape, dtype=None, **kwargs):
      num_classes = shape[0] // self.proxy_per_class
      return tf.expand_dims(
          tf.tile(tf.range(num_classes, dtype=dtype),
                  multiples=[self.proxy_per_class]), axis=1)

  def get_config(self):  # To support serialization
    return {"proxy_per_class": self.proxy_per_class}

def makeProxy(model, num_classes, proxy_per_class, num_sets=1, trainable=True, name='class'):

    embedding_size = model.get_layer('EmbeddingHead')._embedding_size
    num_proxy = num_classes * proxy_per_class
    class_representatives = [model.add_weight(name='{}/proxy_embedding_{}'.format(name, k),
                                              shape=(num_proxy, embedding_size),
                                              dtype=tf.float32,
                                              initializer=tf.keras.initializers.he_normal,
                                              trainable=trainable) for k in range(num_sets)]

    label_initializer = ProxyLabelInitializer(proxy_per_class)
    proxy_labels = [model.add_weight(name='{}/proxy_label_{}'.format(name, k),
                                     shape=(num_proxy, 1),
                                     dtype=tf.int32,
                                     initializer=label_initializer,
                                     trainable=False) for k in range(num_sets)]

    return class_representatives, proxy_labels

@tf.function
def greedyKCenter(point_set, initial_set, K, normalized_embeddings=False):

    # 1) get the points of maximum min-distance
    # 2) remove picked points from representative pool (make distance zero)
    # 3) compute distances between picked samples and the point set
    # 4) update min-distances, ie. min(mindist,newdists)
    # 5) return 1 until K many points are picked

    def updateSet(s: tf.TensorArray, x: tf.Tensor, x_: tf.Tensor, m: tf.Tensor, k: tf.Tensor):
        # 1)
        argmax = tf.expand_dims(tf.argmax(m, axis=-1, output_type=tf.int32), axis=1)
        p_ = tf.gather(x_, argmax, batch_dims=1)

        # 3)
        d = tf.add(
            tf.reduce_sum(tf.square(p_), axis=-1),
            tf.reduce_sum(tf.square(x_), axis=-1)
        ) - 2.0 * tf.reduce_sum(x_ * p_, axis=-1) #d = 2. - 2. * tf.reduce_sum(x_ * p_, axis=-1) for unit vectors

        # 4)
        m = tf.minimum(m, d)

        # note that 2 is implicitly performed during 3 and 4
        p = tf.gather(x, argmax, batch_dims=1)
        s = s.write(k, value=tf.squeeze(p, axis=1))

        # 5)
        k = tf.add(k, 1)

        return s, m, k

    point_set_ = point_set
    if normalized_embeddings:
        point_set_ = tf.nn.l2_normalize(point_set_, axis=-1)
        initial_set = tf.nn.l2_normalize(initial_set, axis=-1)

    dists = tf.add(
        tf.reduce_sum(tf.square(point_set_), axis=-1, keepdims=True), # ||x||^2 +
        tf.expand_dims(tf.reduce_sum(tf.square(initial_set), axis=-1), axis=-2),  # ||y||^2 +
    ) - 2.0 * tf.matmul(point_set_, initial_set, transpose_b=True) # -2xTy

    '''# distance of unit norm vectors: 2 - 2xTy
    dists = tf.subtract(
        2.,
        tf.multiply(
            2.,
            tf.reduce_sum(
                tf.multiply(
                    tf.expand_dims(point_set_unit, axis=2),
                    tf.expand_dims(initial_set, axis=1)
                ), axis=-1)
        )
    )'''


    k_0 = tf.constant(0, dtype=tf.int32)
    m_0 = tf.reduce_min(dists, axis=-1)
    s_0 = tf.TensorArray(dtype=tf.float32, size=K, dynamic_size=False, clear_after_read=False,
                         element_shape=tf.TensorShape([point_set.shape[0], point_set.shape[-1]]))

    s, m, k = tf.while_loop(cond=lambda s, m, k: tf.less(k, K),
                            body=lambda s, m, k: updateSet(s=s, x=point_set, x_=point_set_, m=m, k=k),
                            loop_vars=(s_0, m_0, k_0))

    return s.concat()

'''s = tf.random.uniform(shape=(3, 4, 8))
x = tf.random.uniform(shape=(3, 7, 8))
s = tf.nn.l2_normalize(s, axis=-1)
x = tf.nn.l2_normalize(x, axis=-1)
p = greedyKCenter(x, s, 4)'''