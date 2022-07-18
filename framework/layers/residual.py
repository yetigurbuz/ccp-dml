import tensorflow as tf
#import tensorflow_addons as tfa


class CenterAround(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""

  def __init__(self, axis=-2, ref_value=0.0, **kwargs):
      super(CenterAround, self).__init__(**kwargs)
      self.axis = axis
      self.ref_value = ref_value

  def __call__(self, w):
    mean = tf.reduce_mean(w, axis=-self.axis, keepdims=True)
    return w - mean + self.ref_value

  def get_config(self):
      config = super(CenterAround, self).get_config()
      config.update({'axis': self.axis,
                     'ref_value': self.ref_value})
      return config


class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs

def activationBlock(use_activation, activation_block):

    arg_map = {'bn': 'BatchNormalization',
               'batchnormalization': 'BatchNormalization',
               'ln': 'LayerNormalization',
               'layernormalization': 'LayerNormalization',
               'sm': 'Softmax',
               'softmax': 'Softmax',
               'relu': 'ReLU',
               'softplus': 'softplus',
               'sp': 'softplus',
               'groupnormalization': 'GroupNormalization',
               'gn': 'GroupNormalization',
               'instancenormalization': 'InstanceNormalization',
               'in': 'InstanceNormalization'}

    activation_block = activation_block.lower()

    activation_layers = activation_block.split('_')

    if use_activation:
        layers = []
        for layer in activation_layers:
            layer_id = arg_map[layer]
            if hasattr(tf.keras.layers, layer_id):
                layers.append(getattr(tf.keras.layers, arg_map[layer])())
                '''
            elif hasattr(tfa.layers, layer_id):
                args = {'groups': 8} if layer_id == 'GroupNormalization' else {}
                layers.append(getattr(tfa.layers, arg_map[layer])(**args))'''
            else:
                layers.append(getattr(tf.keras.activations, layer_id))

        def activation_fn(x):
            for layer in layers:
                x = layer(x)
            return x
        return activation_fn
    else:
        def activation_fn(x):
            return x
        return activation_fn


def linearTransform(out_dim, spatial_decimation=2, spatial_aggregation='skip'):

        layers = []
        if spatial_aggregation == 'avg':
            # decimate
            layers.append(tf.keras.layers.AveragePooling2D(pool_size=spatial_decimation,
                                                           strides=spatial_decimation,
                                                           padding='SAME'))
            # transform
            layers.append(tf.keras.layers.Conv2D(out_dim,
                                                 kernel_size=1,
                                                 padding='SAME',
                                                 use_bias=False))
        elif spatial_aggregation == 'skip':
            # jointly decimate and transform
            layers.append(tf.keras.layers.Conv2D(out_dim,
                                                 kernel_size=1,
                                                 strides=spatial_decimation,
                                                 padding='SAME',
                                                 use_bias=False))
        elif spatial_aggregation == 'conv':
            # decimate by strided convolutions
            layers.append(tf.keras.layers.Conv2D(out_dim,
                                                 kernel_size=spatial_decimation,
                                                 strides=spatial_decimation,
                                                 padding='SAME',
                                                 use_bias=False))
        else:
            raise NotImplementedError(
                'for decimation only avg, skip and conv aggregations are available for now!')

        def forward_fn(x):
            for layer in layers:
                x = layer(x)
            return x

        return forward_fn

def resnetBlock(out_dim,
                out_dim_reduction_rate=2,
                spatial_decimation=1,
                spatial_aggregation='skip',
                zero_mean_embedding_kernel=False,
                input_injection=False,
                pre_activation=True,
                pre_activation_block='BN_ReLU',
                mid_activation=True,
                mid_activation_block='BN_ReLU',
                post_activation=True,
                post_activation_block='BN_ReLU'):

    reduced_dim = out_dim // out_dim_reduction_rate

    if input_injection:
        with tf.name_scope("input_injection"):
            injection_activation = activationBlock(use_activation=pre_activation,
                                                   activation_block=pre_activation_block)
            injection_transform = linearTransform(out_dim=reduced_dim,
                                                  spatial_decimation=spatial_decimation,
                                                  spatial_aggregation=spatial_aggregation)

    # pre activation
    pre_activation_fn = activationBlock(use_activation=pre_activation,
                                        activation_block=pre_activation_block)

    # dimension reduction
    linear_transform_fn = linearTransform(out_dim=reduced_dim,
                                          spatial_decimation=spatial_decimation,
                                          spatial_aggregation=spatial_aggregation)

    # mid activation
    mid_activation_fn = activationBlock(use_activation=mid_activation,
                                        activation_block=mid_activation_block)

    # 3x3 feature matching
    conv_fn = tf.keras.layers.Conv2D(filters=reduced_dim,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='SAME',
                                     use_bias=False)

    # post activation
    post_activation_fn = activationBlock(use_activation=post_activation,
                                         activation_block=post_activation_block)


    # semantic neighbourhood embedding
    if zero_mean_embedding_kernel:
        kernel_constraint = CenterAround(axis=-2, ref_value=0.0)
    else:
        kernel_constraint = None

    embedding_fn = tf.keras.layers.Conv2D(filters=out_dim,
                                          kernel_size=1,
                                          strides=1,
                                          padding='SAME',
                                          use_bias=False,
                                          kernel_constraint=kernel_constraint)
    add = tf.keras.layers.Add()

    def forward_fn(x, x0):

        x = pre_activation_fn(x)
        x = linear_transform_fn(x)

        if input_injection:
            x = add([x, injection_transform(injection_activation(x0))])

        x = mid_activation_fn(x)
        x = conv_fn(x)
        x = post_activation_fn(x)
        x = embedding_fn(x)

        return x

    return forward_fn



# higher level blocks that use above implementations mostly

# mapping to reduced spatial and increased feature dimension space with residual connection
def resMap(out_dim,
           out_dim_reduction_rate=2,
           spatial_decimation=1,
           spatial_aggregation='skip',
           zero_mean_embedding_kernel=False,
           pre_activation=True,
           pre_activation_block='BN_ReLU',
           mid_activation=True,
           mid_activation_block='BN_ReLU',
           post_activation=True,
           post_activation_block='BN_ReLU',
           #input_injection=False,
           **kwargs):

    res_block = resnetBlock(out_dim=out_dim,
                            out_dim_reduction_rate=out_dim_reduction_rate,
                            spatial_decimation=spatial_decimation,
                            spatial_aggregation=spatial_aggregation,
                            input_injection=False,
                            zero_mean_embedding_kernel=zero_mean_embedding_kernel,
                            pre_activation=pre_activation,
                            pre_activation_block=pre_activation_block,
                            mid_activation=mid_activation,
                            mid_activation_block=mid_activation_block,
                            post_activation=post_activation,
                            post_activation_block=post_activation_block)

    linear_map = linearTransform(out_dim=out_dim,
                                 spatial_decimation=spatial_decimation,
                                 spatial_aggregation=spatial_aggregation)
    add = tf.keras.layers.Add()

    def forward_fn(x):
        g = res_block(x, x)
        x = linear_map(x)
        x = add([x, g])
        return x

    return forward_fn

def resBlock(out_dim,
             input_injection=False,
             repeats=1,
             out_dim_reduction_rate=2,
             zero_mean_embedding_kernel=False,
             pre_activation=True,
             pre_activation_block='BN_ReLU',
             mid_activation=True,
             mid_activation_block='BN_ReLU',
             post_activation=True,
             post_activation_block='BN_ReLU',
             **kwargs):

    resnet_block = resnetBlock(out_dim=out_dim,
                               out_dim_reduction_rate=out_dim_reduction_rate,
                               spatial_decimation=1,
                               spatial_aggregation='skip',
                               input_injection=input_injection,
                               zero_mean_embedding_kernel=zero_mean_embedding_kernel,
                               pre_activation=pre_activation,
                               pre_activation_block=pre_activation_block,
                               mid_activation=mid_activation,
                               mid_activation_block=mid_activation_block,
                               post_activation=post_activation,
                               post_activation_block=post_activation_block)

    add = tf.keras.layers.Add()

    '''def block_fn(x, x0):
        g = resnet_block(x, x0=x0)
        x = add([x, g])
        return x'''

    def forward_fn(x):
        x0 = x
        for k in range(repeats):
            g = resnet_block(x, x0=x0)
            x = add([x, g])
        return x

    return forward_fn


class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 out_dim=None,
                 batch_size=None,
                 repeats=1,
                 out_dim_reduction_rate=2,
                 zero_mean_embedding_kernel=False,
                 input_injection=False,
                 is_implicit=False,
                 fn_regularizer=True,
                 reset_at=10,
                 max_it=1,
                 step_scaler=2e-3,
                 max_eigen=1.,
                 min_eigen=1.,
                 pre_activation=True,
                 pre_activation_block='BN_ReLU',
                 mid_activation=True,
                 mid_activation_block='BN_ReLU',
                 post_activation=True,
                 post_activation_block='BN_ReLU',
                 name=None,
                 **kwargs):
        super(ResBlock, self).__init__(name=name)

        self._repeats = repeats
        self._out_dim_reduction_rate = out_dim_reduction_rate
        self._zero_mean_embedding_kernel = zero_mean_embedding_kernel
        self._pre_activation = pre_activation
        self._pre_activation_block = pre_activation_block
        self._mid_activation = mid_activation
        self._mid_activation_block = mid_activation_block
        self._post_activation = post_activation
        self._post_activation_block = post_activation_block

        self._input_injection = input_injection
        self._is_implicit = is_implicit
        self._fn_regularizer = fn_regularizer
        self._step_scaler = step_scaler
        self._reset_at = reset_at
        self._max_it = max_it
        self._max_eigen = max_eigen
        self._min_eigen = min_eigen

        self._global_step = tf.Variable(name='global_step',
                                            shape=(), dtype=tf.float32,
                                            initial_value=0.,
                                            trainable=False)

        self._lambda_it = tf.Variable(name='lambda_iter',
                                          shape=(), dtype=tf.int32,
                                          initial_value=0,
                                          trainable=False)

        self._out_dim = out_dim
        self._batch_size = batch_size

    def build(self, input_shape):

        shape = [self._batch_size, input_shape[1], input_shape[2], input_shape[3]]
        self._lambda = tf.Variable(name='lambda',
                                   shape=shape,
                                   dtype=tf.float32,
                                   initial_value=tf.zeros(shape=shape),
                                   trainable=False)

        self._block_fn = resnetBlock(self._out_dim,
                                              out_dim_reduction_rate=2,
                                              spatial_decimation=1,
                                              spatial_aggregation='skip',
                                              input_injection=self._input_injection,
                                              zero_mean_embedding_kernel=self._zero_mean_embedding_kernel,
                                              pre_activation=self._pre_activation,
                                              pre_activation_block=self._pre_activation_block,
                                              mid_activation=self._mid_activation,
                                              mid_activation_block=self._mid_activation_block,
                                              post_activation=self._post_activation,
                                              post_activation_block=self._post_activation_block)

        self._add = tf.keras.layers.Add()

        inp1 = tf.keras.layers.Input(shape=input_shape[1:], batch_size=input_shape[0])
        inp2 = tf.keras.layers.Input(shape=input_shape[1:], batch_size=input_shape[0])

        self.block_as_layer = tf.keras.Model(inputs=[inp1, inp2], outputs=self._block_fn(inp1, inp2))
        self.block_weights = self.block_as_layer.trainable_weights

    def forward_block_while(self, x):

        x_in = x

        x_0 = x_in
        k_0 = tf.constant(0, dtype=tf.int32)
        k, x = (
            tf.while_loop(cond=lambda k, x: tf.less(k, self._repeats),
                          body=lambda k, x: (k + 1, self._add([x, self.block_as_layer([x, x_in])])),
                          loop_vars=(k_0, x_0),
                          shape_invariants=(k_0.get_shape(), x_0.get_shape())))

        return x

    def forward_block_for(self, x):

        x_in = x
        for k in range(self._repeats):
            g = self.block_as_layer([x, x_in])
            x = self._add([x, g])

        return x

    def backward_block(self, x_in, x, dx, params):

        # update adaptive parameters
        decayed_step = self._step_scaler * self._global_step.assign_add(1.)
        mu = (self._max_eigen + decayed_step) / (self._min_eigen + self._max_eigen + decayed_step)
        nu = 1. / (self._max_eigen + decayed_step)

        # compute g (to be used in gradient update)
        g = self.block_as_layer([x, x_in])

        # main body of iteration (inner loop function to update lambda)
        neumman_iteration = lambda lmbd: tf.add(
            mu * lmbd,
            tf.subtract(self.block_as_layer([x - mu * nu * lmbd, x_in]),
                        dx))

        # at reset: lambda_0 = g - dL_dx
        reset_lambda = lambda: self._lambda.assign(tf.subtract(g, dx))

        # if lambda_iterations = 0 then initialize new instance
        lmbd_0 = tf.cond(tf.equal(self._lambda_it, 0),
                         true_fn=reset_lambda,
                         false_fn=lambda: self._lambda)

        k_0 = tf.constant(0, dtype=tf.int32)

        k, lmbd = tf.nest.map_structure(tf.stop_gradient,
            tf.while_loop(cond=lambda k, lmbd: tf.less(k, self._max_it),
                          body=lambda k, lmbd: (k + 1, neumman_iteration(lmbd)),
                          loop_vars=(k_0, lmbd_0),
                          shape_invariants=(k_0.get_shape(), lmbd_0.get_shape())
                          )
                                         )

        lmbd = self._lambda.assign(lmbd)  # computed lambda

        # post iteration updates for iteration counters (reset etc.)
        it = self._lambda_it.assign_add(k)
        it = tf.cond(tf.greater_equal(it, self._reset_at),
                     true_fn=lambda: self._lambda_it.assign(0),
                     false_fn=lambda: self._lambda_it)

        # compute gradients using lambda
        grad_ys = lmbd
        if self._input_injection:
            dL = tf.gradients(ys=g, xs=[x_in] + params, grad_ys=grad_ys)
        else:
            # dg_dx_in is assumed to be Identitiy (projection problem)
            dL = [lmbd] + tf.gradients(ys=g, xs=params, grad_ys=grad_ys)

        return ([dL[0]], dL[1:])

    def backward_block_for(self, x_in, x, dx, params):

        # update adaptive parameters
        gs = self._global_step.assign_add(1.)
        decayed_step = self._step_scaler * gs
        mu = (self._max_eigen + decayed_step) / (self._min_eigen + self._max_eigen + decayed_step)
        nu = 1. / (self._max_eigen + decayed_step)
        eps = 1e-2 / (10*gs)

        # compute g (to be used in gradient update)
        g = self.block_as_layer([x, x_in])

        # if lambda_iterations = 0 then initialize new instance
        lmbd_0 = tf.cond(tf.equal(self._lambda_it, 0),
                         true_fn=lambda: self._lambda.assign(eps*g+dx),
                         false_fn=lambda: self._lambda)
        dg_dx_by_lmbd = tf.gradients(ys=g, xs=x, grad_ys=lmbd_0)[0]

        for k in range(self._max_it):
            lmbd_0 = self._lambda.assign(
                tf.add(eps*g+mu * lmbd_0,
                            tf.add(mu * nu * dg_dx_by_lmbd, dx)))
            dg_dx_by_lmbd = tf.gradients(ys=g, xs=x, grad_ys=lmbd_0)[0]

        # post iteration updates for iteration counters (reset etc.)
        it = self._lambda_it.assign_add(self._max_it)
        it = tf.cond(tf.greater_equal(it, self._reset_at),
                     true_fn=lambda: self._lambda_it.assign(0),
                     false_fn=lambda: self._lambda_it)

        # compute gradients using lambda
        grad_ys = lmbd_0
        if self._input_injection:
            dL = tf.gradients(ys=g, xs=[x_in] + params, grad_ys=grad_ys)
        else:
            # dg_dx_in is assumed to be Identitiy (projection problem)
            dL = [lmbd_0] + tf.gradients(ys=g, xs=params, grad_ys=grad_ys)

        tf.print('\ndx mag:', tf.reduce_mean(tf.abs(lmbd_0)))

        tf.print('gradient mag:', tf.reduce_mean(tf.abs(g)))
        tf.print('lambda mag:', tf.reduce_mean(tf.abs(self._lambda)))
        tf.print('lambda dg_dx_lmbd mag:', tf.reduce_mean(tf.abs(dg_dx_by_lmbd)))
        tf.print('weight:', tf.reduce_mean(tf.abs(params[-1])))

        return ([dL[0]], dL[1:])












    def backward_block_in_1(self, x_in, x, dx, params):

        # update adaptive parameters
        decayed_step = self._step_scaler * self._global_step.assign_add(1.)
        #mu = (self._max_eigen + decayed_step) / (self._min_eigen + self._max_eigen + decayed_step)
        #nu = 1. / (self._max_eigen + decayed_step)
        mu = 1 - 0.5
        nu = 1e-3
        eps = 1e-6


        if self._fn_regularizer:

            # compute g (to be used in gradient update)
            g = self.block_as_layer([x, x_in])


            tf.print('\ng(x) mag:',tf.reduce_mean(tf.abs(g)))
            tf.print('dL_dx mag:', tf.reduce_mean(tf.abs(dx)))
            tf.print('lambda mag:', tf.reduce_mean(tf.abs(self._lambda)))

            # if lambda_iterations = 0 then initialize new instance
            lmbd_0 = tf.cond(tf.equal(self._lambda_it, 0),
                             # at reset: lambda_0 = nu (g - dL_dx)
                             true_fn=lambda: self._lambda.assign(eps*g - dx),
                             false_fn=lambda: self._lambda)

            #dg_dx_by_lmbd = tf.gradients(ys=g, xs=x, grad_ys=lmbd_0)[0]
            #tf.print('dg_dx_lmbd mag:', tf.reduce_mean(tf.abs(dg_dx_by_lmbd)))

            # main body of iteration (inner loop function to update lambda)
            lmbd = self._lambda.assign( tf.add(mu * lmbd_0, tf.subtract(nu*g, dx)))
            '''lmbd = self._lambda.assign(
                tf.add(mu * lmbd_0,
                       tf.subtract(nu*self.block_as_layer([x - mu * nu * lmbd_0, x_in]),
                                   dx)))'''

            '''lmbd = self._lambda.assign(
                tf.add(eps*g,
                       tf.subtract(mu * lmbd_0,
                                   tf.add(mu * nu * dg_dx_by_lmbd, dx))))'''
            #tf.print('\nlambda:', lmbd_0[0][11:15, 17:21, 0])

        else:
            # compute g (to be used in gradient update)
            g = self.block_as_layer([x, x_in])

            # if lambda_iterations = 0 then initialize new instance
            lmbd_0 = tf.cond(tf.equal(self._lambda_it, 0),
                             true_fn=lambda: self._lambda.assign(-dx),
                             false_fn=lambda: self._lambda)
            tf.print('\ndx mag:', tf.reduce_sum(tf.square(lmbd_0)))
            # alternative block
            dg_dx_by_lmbd = tf.gradients(ys=g, xs=x, grad_ys=lmbd_0)[0]

            tf.print('gradient mag:', tf.reduce_sum(tf.square(g)))
            tf.print('lambda mag:', tf.reduce_sum(tf.square(self._lambda)))
            tf.print('lambda dg_dx_lmbd mag:', tf.reduce_sum(tf.square(dg_dx_by_lmbd)))

            lmbd = self._lambda.assign(
                tf.subtract(mu * lmbd_0,
                       tf.add(mu * nu * dg_dx_by_lmbd, -dx)))

            #tf.print('\nlambda:', lmbd_0[0][11:15, 17:21,0])





        # post iteration updates for iteration counters (reset etc.)
        it = self._lambda_it.assign_add(1)
        it = tf.cond(tf.greater_equal(it, self._reset_at),
                     true_fn=lambda: self._lambda_it.assign(0),
                     false_fn=lambda: self._lambda_it)

        # compute gradients using lambda
        grad_ys = lmbd
        if self._input_injection:
            dL = tf.gradients(ys=g, xs=[x_in]+params, grad_ys=grad_ys)
        else:
            # dg_dx_in is assumed to be Identitiy (projection problem)
            dL = [lmbd] + tf.gradients(ys=g, xs=params, grad_ys=grad_ys)

        #tf.print(dL[1:])
        tf.print('dL:', tf.reduce_mean(tf.abs(dL[5])))

        return ([dL[0]], dL[1:])

    def backward_depth_1(self, x_in, x, dx, params):

        # compute g (to be used in gradient update)
        g = self.block_as_layer([x_in, x_in])
        x = x_in + g

        dL = tf.gradients(ys=x, xs=[x_in] + params, grad_ys=dx)

        return ([dL[0]], dL[1:])

    def compute_block(self, x):

        if self._is_implicit:  # custom gradient
            def opt(x_in):
                x_0 = x_in
                x_opt = tf.stop_gradient(self.forward_block_while(x_0))
                grad_fn = lambda dy, variables=None: self.backward_block_for(x_0, x_opt, dy, variables)
                return x_opt, grad_fn

            x_star = tf.custom_gradient(opt)(x)

        else:  # auto gradient
            x_star = self.forward_block_while(x)

        return x_star

    def call(self, inputs, **kwargs):

        x = self.compute_block(inputs)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({'repeats': self._repeats,
                       'out_dim_reduction_rate': self._out_dim_reduction_rate,
                       'zero_mean_embedding_kernel': self._zero_mean_embedding_kernel,
                       'input_injection': self._input_injection,
                       'is_implicit': self._is_implicit,
                       'fn_regularizer': self._fn_regularizer,
                       'reset_at': self._reset_at,
                       'max_it': self._max_it,
                       'step_scaler': self._step_scaler,
                       'max_eigen': self._max_eigen,
                       'min_eigen': self._min_eigen,
                       'pre_activation': self._pre_activation,
                       'pre_activation_block': self._pre_activation_block,
                       'mid_activation': self._mid_activation,
                       'mid_activation_block': self._mid_activation_block,
                       'post_activation': self._post_activation,
                       'post_activation_block': self._post_activation_block,
                       'out_dim': self._out_dim
                       })

        return config












