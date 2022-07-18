import tensorflow as tf

class EMARegularizer():
    def __init__(self, model, momentum=0.99, alpha=1e-7, beta=1e-5, gamma=1e-5):
        '''
        :param model: model to be optimized
        :param momentum: moving average momentum
        :param alpha: attractive regularizer weight 1e-7
        :param beta: repulsive regularizer weight 1e-5
        :param gamma: weight decay regularizer weight
        '''

        self.ema_weights = [tf.Variable(shape=w.get_shape(),
                                        initial_value=tf.zeros_like(w),
                                        trainable=False,
                                        dtype=tf.float32) for w in model.trainable_weights]

        self.num_weights = len(self.ema_weights)

        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta * tf.cast(tf.add_n([tf.keras.backend.count_params(p)
                                             for p in model.trainable_weights]), tf.float32)
        self.gamma = gamma

    def __call__(self, grads_and_vars):

        grads, vars = zip(*grads_and_vars)

        # update moving average
        ema = [self.ema_weights[k].assign(self.momentum * (self.ema_weights[k] - vars[k]) + vars[k])
               for k in range(self.num_weights)]

        # update gradient
        norm_2_square = tf.add_n([tf.reduce_sum(tf.square(vars[k] - ema[k]))
                                  for k in range(self.num_weights)])
        norm_2 = tf.sqrt(norm_2_square)

        coeff = (self.alpha * norm_2_square - self.beta / norm_2_square) / norm_2

        grads = [grads[k] + coeff * (vars[k] - ema[k]) + self.gamma * vars[k]
                 for k in range(self.num_weights)]


        return list(zip(grads, vars))

class WeightRegularizer():
    def __init__(self, model, penalty_weight=1e-5, excluded=None):
        '''
        :param model: model to be optimized
        :param penalty_weight: weight decay regularizer weight
        :param excluded: list of names of the variables to be excluded
        '''

        if excluded is None:
            excluded = []

        if not isinstance(excluded, (list, tuple)):
            excluded = [excluded]

        self._alpha = [0.0 if any([e in w.name for e in excluded]) else penalty_weight for w in model.trainable_weights]

        self.num_weights = len(self._alpha)

    def __call__(self, grads_and_vars):
        grads_and_vars = [(grads_and_vars[k][0] + self._alpha[k] * grads_and_vars[k][1], grads_and_vars[k][1])
                          for k in range(self.num_weights)]

        return grads_and_vars

class ClipGradients():
    def __init__(self, clip_norm=5.0, clip_value=5.0):
        '''
        :param clip_norm: global norm to clip
        :param clip_value: value to clip
        '''

        self._clip_norm = clip_norm
        self._clip_value = clip_value


    def __call__(self, grads_and_vars):

        grads, vars = zip(*grads_and_vars)

        if self._clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)

        if self._clip_value is not None:
            grads = [tf.clip_by_value(g, -self._clip_value, self._clip_value) for g in grads]

        return list(zip(grads, vars))

class LearningRateMultiplier():
    def __init__(self, model: tf.keras.Model, lrm: dict):
        '''
        :param lrm: name-value pair e.g. {'var_name_suffix': 100.0}
        '''

        var_names = [w.name for w in model.trainable_weights]

        self._lrm = []
        for name in var_names:
            for suffix in lrm.keys():
                if suffix in name:
                    self._lrm.append(lrm[suffix])
                else:
                    self._lrm.append(1.0)

        self.num_weights = len(self._lrm)

    def __call__(self, grads_and_vars):
        grads_and_vars = [(self._lrm[k] * grads_and_vars[k][0], grads_and_vars[k][1])
                          for k in range(self.num_weights)]

        return grads_and_vars

class DistanceRegularizer():
    def __init__(self, model_weights, penalty_weight=1e-5):
        '''
        :param model: model to be optimized
        :param penalty_weight: weight decay regularizer weight
        :param excluded: list of names of the variables to be excluded
        '''

        self._lambda = penalty_weight

        self._prev_weights = [tf.Variable(shape=w.get_shape(),
                                          initial_value=tf.zeros_like(w),
                                          trainable=False,
                                          dtype=tf.float32) for w in model_weights]

        self._lambda = tf.Variable(shape=(),
                                   initial_value=penalty_weight,
                                   trainable=False,
                                   dtype=tf.float32)

        self._num_weights = len(self._prev_weights)


    def __call__(self, grads_and_vars):

        non_proxy_g_and_v = [(g + self._lambda * (v - self._prev_weights[k]), v)
                             for k, (g, v) in enumerate(grads_and_vars[:self._num_weights])]

        grads_and_vars = non_proxy_g_and_v + grads_and_vars[self._num_weights:]

        return grads_and_vars

def updateGradientTransformers(model, weight_decay=None, excluded_vars=None, clipnorm=None, clipvalue=None, lrm=None,
                               mode='training'):
    if mode == 'training':
        gradient_transformers = model.gradient_transformers
    elif mode == 'warm_up':
        gradient_transformers = model.warm_up_gradient_transformers
    else:
        gradient_transformers = None

    if (weight_decay is not None) and weight_decay > 0.0:
        gradient_transformers.append(WeightRegularizer(model=model.arch,
                                                       penalty_weight=weight_decay,
                                                       excluded=excluded_vars))
    if (clipnorm is not None) or (clipvalue is not None):
        gradient_transformers.append(ClipGradients(clip_norm=clipnorm, clip_value=clipvalue))

    if lrm is not None:
        if len(lrm) > 0:
            gradient_transformers.append(LearningRateMultiplier(model=model.arch, lrm=lrm))


