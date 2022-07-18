import tensorflow as tf

from ..layers import L2Normalization

class GlobalMetric(tf.keras.metrics.Metric):

    @tf.function
    def compute(self, embeddings, labels,
                ref_embeddings=None, ref_labels=None,
                has_self_match=True):
        raise NotImplementedError(
            '''compute(...) is to be implemented in GlobalMetric sub classes
            and should return the computed metric
            Corresponding Callback first calls .compute(...) to compute the result and then passes result to
            update_fn and updates metric result.''')

    def toScalar(self, computed_metric):
        raise NotImplementedError(
            '''toScalar(...) is to be implemented in GlobalMetric sub classes
            and should return a scalar from the computed_metric so that corresponding Callback updates metric result.
            toScalar is useful to convert vector valued metrics such as R@K for [1, 10, 100] to scalar R@1 for
            performance monitoring.''')


    # optional print function
    def printResult(self, metric_result):
        print('Warning: User decorated printing is not implemented.')
        print('{}: {}'.format(self.name, metric_result))

    @tf.function
    def _computeGlobal(self, embeddings, labels, batch_size=None):
        '''assumes compute(embeddings, labels, ref_embeddings, ref_labels) returns averaged result'''

        if self.normalize_embeddings:
            embeddings = self.l2normalization(embeddings)

        split_at = self.split_at
        if split_at > 0:
            print('\033[3;34m' + '\nINFO:Metric:Split: ' + 'Query-Reference split is at {}!\n'.format(split_at) + '\033[0m')
            ref_embeddings = embeddings[:split_at]
            ref_labels = labels[:split_at]
            embeddings = embeddings[split_at:]
            labels = labels[split_at:]
            has_self_match = False

        else:
            ref_embeddings = embeddings
            ref_labels = labels
            has_self_match = True

        if batch_size is None:
            metric_result = self.compute(embeddings, labels,
                                         ref_embeddings, ref_labels,
                                         has_self_match=has_self_match)
        else:
            if split_at > 0:
                total_queires = split_at
            else:
                total_queires = ref_labels.get_shape().as_list()[0]

            num_splits = total_queires // batch_size

            num_residual_queries = total_queires - batch_size * num_splits

            query_emb_list = tf.TensorArray(dtype=tf.float32, size=num_splits,
                                            dynamic_size=False, clear_after_read=False,
                                            element_shape=tf.TensorShape([batch_size, None]))
            query_emb_list = query_emb_list.split(
                ref_embeddings[:total_queires-num_residual_queries], [batch_size] * num_splits)

            query_lbl_list = tf.TensorArray(dtype=tf.int32, size=num_splits,
                                            dynamic_size=False, clear_after_read=False,
                                            element_shape=tf.TensorShape([batch_size, 1])
                                            )
            query_lbl_list = query_lbl_list.split(
                ref_labels[:total_queires-num_residual_queries], [batch_size] * num_splits)


            '''metric_result = 0.0
            if num_residual_queries > 0:
                ref_embeddings = embeddings[:num_residual_queries]
                ref_labels = labels[:num_residual_queries]
                metric_result += self.compute(embeddings, labels, ref_embeddings, ref_labels) * num_residual_queries

            for ref_embeddings, ref_labels in (
                    zip(tf.split(embeddings[num_residual_queries:], num_splits, axis=0),
                        tf.split(labels[num_residual_queries:], num_splits, axis=0))):

                metric_result += self.compute(embeddings, labels, ref_embeddings, ref_labels) * batch_size
            metric_result = metric_result / total_queires'''


            '''k_0 = tf.constant(0, dtype=tf.int32)
            m_0 = tf.constant(0.0, shape=self.metric_shape, dtype=tf.float32)

            cond = lambda k, m: tf.less(k, num_splits)
            body = lambda k, m: (k + 1,
                                 (m +
                                 self.compute(embeddings,
                                              labels,
                                              tf.ensure_shape(x=embeddings[(k * batch_size):((k+1) * batch_size)],
                                                              shape=(batch_size, None)),
                                              tf.ensure_shape(x=labels[(k * batch_size):((k+1) * batch_size)],
                                                              shape=(batch_size, 1)))
                                  )
                                 )

            k, metric_result = tf.nest.map_structure(tf.stop_gradient,
                                                     tf.while_loop(cond=cond,
                                                                   body=body,
                                                                   loop_vars=(k_0, m_0),
                                                                   shape_invariants=(
                                                                   k_0.get_shape(),
                                                                   m_0.get_shape())
                                                                   )
                                                     )
            
            metric_result = metric_result * batch_size'''

            k_0 = tf.constant(0, dtype=tf.int32)
            m_0 = tf.constant(0.0, shape=self.metric_shape, dtype=tf.float32)

            def cond_fn(k, m):
                return tf.less(k, num_splits)

            def body_fn(k, m):

                '''query_emb = tf.reshape(ref_embeddings[(k * batch_size):((k + 1) * batch_size)],
                                       shape=(batch_size, -1))
                query_lbl = tf.reshape(ref_labels[(k * batch_size):((k + 1) * batch_size)],
                                        shape=(batch_size, 1))'''

                query_emb = query_emb_list.read(k)
                query_lbl = query_lbl_list.read(k)

                m = tf.add(
                    m,
                    self.compute(
                        embeddings,
                        labels,
                        query_emb,
                        query_lbl,
                        has_self_match=has_self_match
                    )
                )

                k = tf.add(k, 1)
                #tf.print(m)
                return k, m

            k_out, metric_result = tf.nest.map_structure(tf.stop_gradient,
                                                         tf.while_loop(cond=cond_fn,
                                                                       body=body_fn,
                                                                       loop_vars=(k_0, m_0),
                                                                       shape_invariants=(
                                                                           k_0.get_shape(),
                                                                           m_0.get_shape()),
                                                                       parallel_iterations=5
                                                                       )
                                                         )

            metric_result = metric_result * batch_size

            if num_residual_queries > 0:
                res_embeddings = ref_embeddings[-num_residual_queries:]
                res_labels = ref_labels[-num_residual_queries:]
                metric_result += self.compute(embeddings, labels, res_embeddings, res_labels) * num_residual_queries

            metric_result = metric_result / total_queires

        return metric_result

    def buildGlobalComputeGraph(self, dataset_size, batch_size=None):
        self.computeGlobal = \
            self._computeGlobal.get_concrete_function(embeddings=tf.TensorSpec(shape=[dataset_size, None],
                                                                               dtype=tf.float32),
                                                      labels=tf.TensorSpec(shape=[dataset_size, 1],
                                                                           dtype=tf.int32),
                                                      batch_size=batch_size)

    def computeGlobal(self, embeddings, labels):
        raise ValueError('''Computational graph for global metric computation has not been built yet.
        Call buildGlobalComputeGraph(dataset_size, batch_size, split_at=(optional)) with dataset and batch size parameters first
        to build the graph.''')

    def __init__(self, metric_shape, name, normalize_embeddings=False,
                 lipschitz_cont=False, split_at=0, **kwargs):
        super(GlobalMetric, self).__init__(name=name, **kwargs)
        self.metric_result = self.add_weight(name='{}/metric_result'.format(self.name),
                                             initializer='zeros', dtype=tf.float32)
        self.metric_shape = metric_shape
        self.print_text = ''
        self.plain_print_text = ''

        self.normalize_embeddings = normalize_embeddings
        if lipschitz_cont:
            print('\033[3;34m' + 'INFO:Metric:Normalization: ' + 'Lipschitz cont. L2 Normalization is used.' + '\033[0m')
            self.l2normalization = L2Normalization()
        else:
            print('\033[3;34m' + 'INFO:Metric:Normalization: ' + 'Classical L2 Normalization is used.' + '\033[0m')
            self.l2normalization = lambda x: tf.nn.l2_normalize(x, axis=-1)
        self.split_at = split_at



    def update_state(self, y_true, y_pred, sample_weight=None):
        # do nothing
        pass

    def result(self):

        return self.metric_result

    def reset_state(self):
        # do nothing
        pass

