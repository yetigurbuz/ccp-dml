import tensorflow as tf
from typing import List

from ...utilities import proxy_utils

class AlternatingProjectionCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 curr_weights: List[tf.Variable],
                 model_weigths: List[tf.Variable],
                 ccp_lambda: tf.Variable,
                 lambda_decay: float,
                 perturb_noise=0.0,
                 ccp_iterations=32,
                 step_at_batch=True,
                 early_stopping=False,
                 early_stopping_patience=8,
                 monitor_to_stop='loss',
                 stop_at_min=True,
                 class_representatives=None,
                 proxy_per_class=1,
                 representative_sampler=None,
                 representative_pool_size=1,
                 random_init=False,
                 normalized_embeddings=False):

        super(AlternatingProjectionCallback, self).__init__()

        self.curr_weights = curr_weights
        self.model_weigths = model_weigths
        self.ccp_lambda = ccp_lambda
        self.lambda_decay = lambda_decay
        self.step = 0
        self.ccp_iterations = ccp_iterations

        self.best_weights = [tf.Variable(shape=w.get_shape(),
                                         initial_value=tf.zeros_like(w),
                                         trainable=False,
                                         dtype=tf.float32) for w in model_weigths]

        self.perturb_noise = perturb_noise

        self.class_representatives = class_representatives
        self.best_class_representatives = None


        if self.class_representatives is not None:
            self.best_class_representatives = tf.Variable(shape=class_representatives.get_shape(),
                                                          initial_value=tf.zeros_like(class_representatives),
                                                          trainable=False,
                                                          dtype=tf.float32)
            if representative_sampler is None:
                raise ValueError('representative_sampler is None! Dataset to sample one sample from each class should be provided to use class proxies.')


        self.proxy_per_class = proxy_per_class

        self.get_embeddings = representative_sampler
        self.representative_pool_size = representative_pool_size
        self.normalized_embeddings = normalized_embeddings

        self.random_init = random_init

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.monitor_to_stop = monitor_to_stop

        self.stop_at_min = stop_at_min
        self.steps_wo_improvement = 0
        self.min_improvement_margin = 1e-5
        self.best_monitored_so_far = 1e16

        self.ccp_iteration_record = []
        self.dists = []

        if step_at_batch:
            self.on_batch_end = self.stepFn
        else:
            self.on_epoch_end = self.stepFn

        self.global_step = tf.Variable(initial_value=0.0,
                                       trainable=False, shape=(), dtype=tf.float32)

        self.ccp_lambda_0 = self.ccp_lambda.read_value()


    @tf.function
    def _updateRepresentatives(self):

        current_proxies = tf.stack(
            tf.split(self.best_class_representatives,
                     num_or_size_splits=self.proxy_per_class,
                     axis=0),
            axis=1)

        embedding_pool = None

        if self.representative_pool_size > 1: # perform greedy k-center to pick proxies
            pool_size = (self.representative_pool_size
                         if self.representative_pool_size >= self.proxy_per_class
                         else self.representative_pool_size * self.proxy_per_class)

            # sample multiple candidates

            k_0 = tf.constant(0, dtype=tf.int32)
            s_0 = tf.TensorArray(dtype=tf.float32, size=pool_size, dynamic_size=False, clear_after_read=False,
                                 element_shape=tf.TensorShape([current_proxies.shape[0], current_proxies.shape[-1]]))

            s, k = tf.while_loop(
                cond=lambda s_in, k_in: tf.less(k_in, pool_size),
                body=lambda s_in, k_in: (s_in.write(k_in, value=self.get_embeddings()),
                                         tf.add(k_in, 1)),
                loop_vars=(s_0, k_0))
            embedding_pool = tf.stack(
                tf.split(s.concat(),
                         num_or_size_splits=pool_size,
                         axis=0),
                axis=1)





            representative_embeddings = proxy_utils.greedyKCenter(embedding_pool,
                                                                  current_proxies,
                                                                  self.proxy_per_class,
                                                                  normalized_embeddings=self.normalized_embeddings)


        else:
            k_0 = tf.constant(0, dtype=tf.int32)
            s_0 = tf.TensorArray(dtype=tf.float32, size=self.proxy_per_class, dynamic_size=False, clear_after_read=False,
                                 element_shape=tf.TensorShape([self.class_representatives.shape[0] // self.proxy_per_class, self.class_representatives.shape[1]]))

            s, k = tf.while_loop(
                cond=lambda s_in, k_in: tf.less(k_in, self.proxy_per_class),
                body=lambda s_in, k_in: (s_in.write(k_in, value=self.get_embeddings()),
                                         tf.add(k_in, 1)),
                loop_vars=(s_0, k_0))
            representative_embeddings = s.concat()

        self.class_representatives.assign(representative_embeddings)

        next_proxies = tf.stack(
            tf.split(representative_embeddings,
                     num_or_size_splits=self.proxy_per_class,
                     axis=0),
            axis=1)

        if embedding_pool is None:
            embedding_pool = next_proxies

        return current_proxies, next_proxies, embedding_pool

    @tf.function
    def _updateSolution(self):

        for w_curr, w_next in zip(self.curr_weights, self.best_weights):
            w_curr.assign(w_next)
            if self.perturb_noise > 0.0:
                avg_mag = tf.sqrt(tf.reduce_mean(tf.square(w_next)))
                noise = avg_mag * tf.random.normal(shape=w_next.shape, stddev=self.perturb_noise)
                #noise = noise / tf.sqrt(tf.reduce_sum(tf.square(noise)))
                w_next_noisy = w_next + noise
                w_next.assign(w_next_noisy)

    @tf.function
    def _updateBest(self):

        # current best weights
        [w_best.assign(w_model)
         for w_best, w_model in zip(self.best_weights, self.model_weigths)]

        # current best proxies
        self.best_class_representatives.assign(self.class_representatives)

    @tf.function
    def _updateLambdaCoeff(self):

        updated_lambda = tf.cond(
            tf.greater(self.global_step, 0.1),
            true_fn=lambda: self.ccp_lambda_0 * (self.lambda_decay ** self.global_step),
            false_fn=lambda: 0.0)

        return self.ccp_lambda.assign(updated_lambda)

    @tf.function
    def ccpUpdate(self):

        curr_step = tf.cast(self.global_step.assign_add(1.0), tf.int64)
        self._updateSolution()
        curr_lambda = self._updateLambdaCoeff()

        prev_proxy = None
        next_proxy = None
        embedding_pool = None
        if self.class_representatives is not None:
            prev_proxy, next_proxy, embedding_pool = self._updateRepresentatives()

        return curr_step, curr_lambda, prev_proxy, next_proxy, embedding_pool

    @tf.function
    def ccpInitialize(self):
        curr_lambda = self._updateLambdaCoeff()

        prev_proxy = None
        next_proxy = None
        embedding_pool = None
        if (self.class_representatives is not None) and (not self.random_init):
            print('\nccp: initialize from samples...')
            prev_proxy, next_proxy, embedding_pool = self._updateRepresentatives()

        curr_step = tf.cast(self.global_step, tf.int64)
        return curr_step, curr_lambda, prev_proxy, next_proxy, embedding_pool


    def stepFn(self, step, logs=None):

        self.step += 1

        stop_flag = (self.step >= self.ccp_iterations)
        if self.early_stopping and stop_flag:
            monitored_quantity = logs[self.monitor_to_stop]

            if not self.stop_at_min:
                monitored_quantity = -monitored_quantity

            if monitored_quantity >= (self.best_monitored_so_far - self.min_improvement_margin):
                self.steps_wo_improvement += 1
            else:
                self.steps_wo_improvement = 0
                self.best_monitored_so_far = monitored_quantity
                self._updateBest()

            stop_flag = stop_flag and (self.steps_wo_improvement == self.early_stopping_patience)

        if stop_flag:

            self.ccpUpdate()

            self.step = 0
            self.steps_wo_improvement = 0
            self.best_monitored_so_far = 1e16

    def on_train_begin(self, logs=None):

        self.step = 0
        self.ccpInitialize()

