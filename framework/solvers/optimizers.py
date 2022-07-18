class Optimizers:
    from .adamlrm import AdamLRM
    from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD


def buildOptimizer(optimizer_class, **kwargs):

    class Optimizer(optimizer_class):
        def __init__(self, **kwargs):

            #self._clipnorm = None
            #self._clipvalue = None
            self._gradient_transformers = None
            #print(kwargs)
            '''if 'clipnorm' in kwargs:
                self._clipnorm = kwargs.pop('clipnorm')

            if 'clipvalue' in kwargs:
                self._clipvalue = kwargs.pop('clipvalue')'''

            if 'gradient_transformers' in kwargs:
                self._gradient_transformers = kwargs.pop('gradient_transformers')

            if self._gradient_transformers is not None:
                if not isinstance(self._gradient_transformers, (tuple, list)):
                    self._gradient_transformers = [self._gradient_transformers]

            #print(kwargs)
            super(Optimizer, self).__init__(**kwargs)

        def _processGradients(self, grads_and_vars):

            if self._gradient_transformers is not None:
                for tfm_fn in self._gradient_transformers:
                    grads_and_vars = tfm_fn(grads_and_vars)

            '''grads, vars = zip(*grads_and_vars)

            if self._clipnorm is not None:
                grads, _ = tf.clip_by_global_norm(grads, self._clipnorm)

            if self._clipvalue is not None:
                grads = [tf.clip_by_value(g, -self._clipvalue, self._clipvalue) for g in grads]

            grads_and_vars = list(zip(grads, vars))'''

            return grads_and_vars

        # methods to be overridden
        # ============================================================

        def _compute_gradients(self, loss, var_list, grad_loss=None):
            """Compute gradients of `loss` for the variables in `var_list`.
                This is the first part of `minimize()`.  It returns a list
                of (gradient, variable) pairs where "gradient" is the gradient
                for "variable".  Note that "gradient" can be a `Tensor`, an
                `IndexedSlices`, or `None` if there is no gradient for the
                given variable.
                Args:
                  loss: A callable taking no arguments which returns the value to minimize.
                  var_list: list or tuple of `Variable` objects to update to minimize
                    `loss`, or a callable returning the list or tuple of `Variable` objects.
                    Use callable when the variable list would otherwise be incomplete before
                    `minimize` and the variables are created at the first time when `loss`
                    is called.
                  grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
                Returns:
                  A list of (gradient, variable) pairs. Variable is always present, but
                  gradient can be `None`.
                Raises:
                  TypeError: If `var_list` contains anything else than `Variable` objects.
                  ValueError: If some arguments are invalid, or var_list is None.
                """

            grads_and_vars = \
                super(Optimizer, self)._compute_gradients(loss, var_list, grad_loss)

            grads_and_vars = self._processGradients(grads_and_vars)


            return grads_and_vars

        def get_gradients(self, loss, params):
            """Returns gradients of `loss` with respect to `params`.
                Arguments:
                  loss: Loss tensor.
                  params: List of variables.
                Returns:
                  List of gradient tensors.
                Raises:
                  ValueError: In case any gradient cannot be computed (e.g. if gradient
                    function not implemented).
                """
            grads = super(Optimizer, self).get_gradients(loss, params)

            grads_and_vars = list(zip(grads, params))

            return self._processGradients(grads_and_vars)

        def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):

            #grads, vars = zip(*grads_and_vars)

            grads_and_vars = self._processGradients(list(grads_and_vars))

            #grads_and_vars = list(zip(grads, vars))

            return super(Optimizer, self).apply_gradients(grads_and_vars, name, experimental_aggregate_gradients)

        def get_config(self):

            config = super(Optimizer, self).get_config()
            '''if self._clipnorm is not None:
                config['clipnorm'] = self._clipnorm

            if self._clipvalue is not None:
                config['clipvalue'] = self._clipvalue'''

            if self._gradient_transformers is not None:
                config['gradient_transformers'] = self._gradient_transformers

            return config

    return Optimizer(**kwargs)
