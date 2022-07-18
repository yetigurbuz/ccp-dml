import tensorflow as tf

class GlobalMetric(tf.keras.callbacks.Callback):
    def __init__(self, metrics, feature_ends, val_datasets,
                 batch_size=None,
                 compute_freq=1,
                 verbose=2):
        '''
        This callback computes the set of metrics for each dataset in val_datasets. The each metric is computed from
        its corresponding feature obtained from the layers in feature_ends.
        Arguments:
        metrics: rank-2 list of metrics to be computed, i.e. a list for each dataset in val_datasets
        feature_ends: rank-2 list of layer names where the features are obtained to compute the corresponding metric
        val_datasets: rank-1 list of validation datasets for metrics to be computed
        batch_size: compute size if global computation is to be perfomed in batches
        compute_freq: compute intervals in epochs (1 means compute at the end of each epoch)
        verbose: 0: no print, 1: print results of the most recent, 2: additionally print best observed so far
        '''

        super(GlobalMetric, self).__init__()
        if not isinstance(metrics, (list, tuple)):
            metrics = [[metrics]]
        elif not isinstance(metrics[0], (list, tuple)):
            metrics = [metrics]

        if not isinstance(feature_ends, (list, tuple)):
            feature_ends = [[feature_ends]]
        elif not isinstance(feature_ends[0], (list, tuple)):
            feature_ends = [feature_ends]

        if not isinstance(val_datasets, (list, tuple)):
            val_datasets = [val_datasets]

        num_datasets = len(val_datasets)
        num_feature_end_sets = len(feature_ends)
        num_metric_sets = len(metrics)

        # check whether each metric has its corresponding feature end
        if num_feature_end_sets != num_metric_sets:
            raise ValueError('''feature and metric sets are incompatible:
            len(feature_ends)!=len(metrics) with {} vs {}'''.format(num_feature_end_sets, num_metric_sets))
        else:
            for k, (f, m) in enumerate(zip(feature_ends, metrics)):
                if len(f) != len(m):
                    raise ValueError('''number of features and metrics are incompatible for set k = {0}:
                                len(feature_ends[{0}])!=len(metrics[{0}]) with {1} vs {2}'''.format(k, len(f), len(m)))

        # check whether each set has its dataset
        if num_feature_end_sets != num_datasets:
            raise ValueError('''datasets and metrics sets are incompatible:
                        len(val_datasets)!=len(metrics) with {} vs {}'''.format(num_datasets, num_metric_sets))

        # if everthing is ok start initialization

        self.metrics = metrics
        self.feature_ends = feature_ends
        self.datasets = val_datasets

        self.compute_freq = compute_freq
        self.step = 0

        self.verbose = verbose

        self.best_monitored = [[-1 for _ in range(len(metrics[k]))] for k in range(len(val_datasets))]
        self.best_so_far = [[None for _ in range(len(metrics[k]))] for k in range(len(val_datasets))]
        self.steps_wo_improvement = [[0 for _ in range(len(metrics[k]))] for k in range(len(val_datasets))]

        labels = [[] for _ in range(len(val_datasets))]
        val_steps = [0 for _ in range(len(val_datasets))]

        for k, dataset in enumerate(self.datasets):
            for _, lbls in dataset:
                labels[k].append(tf.convert_to_tensor(lbls))
                val_steps[k] += 1

        self.val_steps = val_steps

        self.labels = [tf.expand_dims(tf.concat(lbls, axis=0), axis=1) for lbls in labels]

        dataset_size = [lbls.get_shape().as_list()[0] for lbls in self.labels]

        # build compute graph to compute metric with tensorflow graph
        [[metric.buildGlobalComputeGraph(dataset_size[k], batch_size)
          for metric in metric_set]
         for k, metric_set in enumerate(self.metrics)]

        # helper function to use tf.function efficiently
        self.to_tensor = tf.keras.layers.Lambda(lambda x: x)




    def _computeMetric(self, logs=None):


        for k, (dataset, feature_ends, metrics) in enumerate(zip(self.datasets, self.feature_ends, self.metrics)):

            outputs = [self.model.get_layer(feature_end).output for feature_end in feature_ends]

            model = tf.keras.Model(inputs=self.model.input, outputs=outputs)

            features = model.predict(dataset, verbose=1, steps=self.val_steps[k])

            if not isinstance(features, (list, tuple)):
                features = [features]

            for l, (feature, metric) in enumerate(zip(features, metrics)):
                with tf.device('/device:CPU:0'):
                    computed_metric = metric.computeGlobal(self.to_tensor(feature),
                                                           self.labels[k])

                    monitored_metric = metric.toScalar(computed_metric)

                    metric.set_weights([monitored_metric])

                    monitored_metric = monitored_metric.numpy()
                    logs[metric.name] = monitored_metric

                    self.steps_wo_improvement[k][l] += 1
                    if monitored_metric > self.best_monitored[k][l]:
                        self.best_monitored[k][l] = monitored_metric
                        self.best_so_far[k][l] = computed_metric
                        self.steps_wo_improvement[k][l] = 0

                    if self.verbose > 0:
                        metric.printResult(computed_metric)

                    if self.verbose > 1:
                        print('best so far:')
                        metric.printResult(self.best_so_far[k][l])
                        print('steps without improvement: {}'.format(self.steps_wo_improvement[k][l]))
                        print('===================================')


    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.compute_freq == 0:
            self._computeMetric(logs)

        self.step += 1

    def on_train_end(self, logs=None):

        if self.step % self.compute_freq:
            self._computeMetric()

class GlobalMetricGeneral(tf.keras.callbacks.Callback):
    def __init__(self, metrics, feature_ends, val_datasets,
                 batch_size=None,
                 compute_freq=1,
                 verbose=2):
        '''
        This callback computes the set of metrics for each dataset in val_datasets. The each metric is computed from
        its corresponding feature obtained from the layers in feature_ends.
        Arguments:
        metrics: rank-2 list of metrics to be computed, i.e. a list for each dataset in val_datasets
        feature_ends: rank-2 list of layer names where the features are obtained to compute the corresponding metric
        val_datasets: rank-1 list of validation datasets for metrics to be computed
        batch_size: compute size if global computation is to be perfomed in batches
        compute_freq: compute intervals in epochs (1 means compute at the end of each epoch)
        verbose: 0: no print, 1: print results of the most recent, 2: additionally print best observed so far
        '''

        super(GlobalMetricGeneral, self).__init__()
        if not isinstance(metrics, (list, tuple)):
            metrics = [[metrics]]
        elif not isinstance(metrics[0], (list, tuple)):
            metrics = [metrics]

        if not isinstance(feature_ends, (list, tuple)):
            feature_ends = [[feature_ends]]
        elif not isinstance(feature_ends[0], (list, tuple)):
            feature_ends = [feature_ends]

        if not isinstance(val_datasets, (list, tuple)):
            val_datasets = [val_datasets]

        num_datasets = len(val_datasets)
        num_feature_end_sets = len(feature_ends)
        num_metric_sets = len(metrics)

        # check whether each metric has its corresponding feature end
        if num_feature_end_sets != num_metric_sets:
            raise ValueError('''feature and metric sets are incompatible:
            len(feature_ends)!=len(metrics) with {} vs {}'''.format(num_feature_end_sets, num_metric_sets))
        else:
            for k, (f, m) in enumerate(zip(feature_ends, metrics)):
                if len(f) != len(m):
                    raise ValueError('''number of features and metrics are incompatible for set k = {0}:
                                len(feature_ends[{0}])!=len(metrics[{0}]) with {1} vs {2}'''.format(k, len(f), len(m)))

        # check whether each set has its dataset
        if num_feature_end_sets != num_datasets:
            raise ValueError('''datasets and metrics sets are incompatible:
                        len(val_datasets)!=len(metrics) with {} vs {}'''.format(num_datasets, num_metric_sets))

        # if everthing is ok start initialization

        self.metrics = metrics
        self.feature_ends = feature_ends
        self.datasets = val_datasets

        self.compute_freq = compute_freq
        self.step = 0

        self.verbose = verbose

        self.best_monitored = [[-1 for _ in range(len(metrics[k]))] for k in range(len(val_datasets))]
        self.best_so_far = [[None for _ in range(len(metrics[k]))] for k in range(len(val_datasets))]
        self.steps_wo_improvement = [[0 for _ in range(len(metrics[k]))] for k in range(len(val_datasets))]

        labels = [[] for _ in range(len(val_datasets))]
        val_steps = [0 for _ in range(len(val_datasets))]

        for k, dataset in enumerate(self.datasets):
            for _, lbls in dataset:
                labels[k].append(tf.convert_to_tensor(lbls))
                val_steps[k] += 1

        self.val_steps = val_steps

        self.labels = [tf.expand_dims(tf.concat(lbls, axis=0), axis=1) for lbls in labels]

        dataset_size = [lbls.get_shape().as_list()[0] for lbls in self.labels]

        # build compute graph to compute metric with tensorflow graph
        [[metric.buildGlobalComputeGraph(dataset_size[k], batch_size)
          for metric in metric_set]
         for k, metric_set in enumerate(self.metrics)]

        # helper function to use tf.function efficiently
        self.to_tensor = tf.keras.layers.Lambda(lambda x: x)


    def _computeMetric(self, logs=None):


        for k, (dataset, feature_ends, metrics) in enumerate(zip(self.datasets, self.feature_ends, self.metrics)):

            outputs = [self.model.get_layer(feature_end).output for feature_end in feature_ends]

            model = tf.keras.Model(inputs=self.model.input, outputs=outputs)

            features = model.predict(dataset, verbose=1, steps=self.val_steps[k])

            if not isinstance(features, (list, tuple)):
                features = [features]

            for l, (feature, metric) in enumerate(zip(features, metrics)):
                with tf.device('/device:CPU:0'):
                    computed_metric = metric.computeGlobal(self.to_tensor(feature),
                                                           self.labels[k])

                    monitored_metric = metric.toScalar(computed_metric)

                    metric.set_weights([monitored_metric])

                    monitored_metric = monitored_metric.numpy()
                    logs[metric.name] = monitored_metric

                    self.steps_wo_improvement[k][l] += 1
                    if monitored_metric > self.best_monitored[k][l]:
                        self.best_monitored[k][l] = monitored_metric
                        self.best_so_far[k][l] = computed_metric
                        self.steps_wo_improvement[k][l] = 0

                    if self.verbose > 0:
                        metric.printResult(computed_metric)

                    if self.verbose > 1:
                        print('best so far:')
                        metric.printResult(self.best_so_far[k][l])
                        print('steps without improvement: {}'.format(self.steps_wo_improvement[k][l]))
                        print('===================================')


    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.compute_freq == 0:
            self._computeMetric(logs)

        self.step += 1

    def on_train_end(self, logs=None):

        if self.step % self.compute_freq:
            self._computeMetric()

class GlobalMetricEnsembleModel(tf.keras.callbacks.Callback):
    def __init__(self, metrics, feature_ends, val_datasets,
                 batch_size=None,
                 compute_freq=1):
        '''metrics: list of metrics to be computed
        feature_ends: list of layer names where the features are obtained to compute the corresponding metric
        val_datasets: validation datasets for each metric
        batch_size: compute size if global computation is to be perfomed in batches
        compute_freq: compute intervals in epochs (1 means compute at the end of each epoch)
        ========================================
        note that for many-to-many dataset-to-metric case, supports only many-inputs-to-many-metrics.
        does not support many-dataset-to-single-input-to-many-metrics.
        does support single-dataset-to-many-metrics and many-dataset-to-many-input-to-many-metrics'''

        super(GlobalMetric, self).__init__()
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]

        if not isinstance(feature_ends, (list, tuple)):
            feature_ends = [feature_ends]

        if not isinstance(val_datasets, (list, tuple)):
            val_datasets = [val_datasets]

        self.many_to_many = (len(val_datasets) > 1)

        self.metrics = metrics
        self.feature_ends = feature_ends
        self.dataset = tf.data.Dataset.zip(tuple(val_datasets)) if self.many_to_many else val_datasets[0]
        #self.dataset = self.dataset.map(lambda *x: ((x[0][0], x[1][0], x[2][0], x[3][0]), (x[0][1], x[1][1], x[2][1], x[3][1])))
        self.dataset = self.dataset.map(lambda *x: ((x[0][0], x[1][0]), (x[0][1], x[1][1])))

        self.compute_freq = compute_freq
        self.step = 0

        self.best_monitored = [-1 for _ in range(len(val_datasets))]
        self.best_so_far = [None for _ in range(len(val_datasets))]

        labels = [[] for _ in range(len(val_datasets))]
        k = 0
        for items in self.dataset:
            if self.many_to_many:
                for l, lbls in enumerate(items[1]):
                    labels[l].append(tf.convert_to_tensor(lbls))
            else:
                lbls = items[1]
                labels[0].append(tf.convert_to_tensor(lbls))
            k += 1

        # omit last batch possibly due to uneven remainder in many inputs case
        self.val_steps = k - 1 if self.many_to_many else k

        self.labels = [tf.expand_dims(tf.concat(lbls[:self.val_steps], axis=0), axis=1)
                       for lbls in labels]

        dataset_size = self.labels[0].get_shape().as_list()[0]

        for m, metric in enumerate(self.metrics):
            metric.buildGlobalComputeGraph(dataset_size, batch_size)

        self.to_tensor = tf.keras.layers.Lambda(lambda x: x)


    def _computeMetric(self):

        '''outputs = []
        for feature_end in self.feature_ends:
            outputs.append(self.model.get_layer(feature_end).output)

        model = tf.keras.Model(inputs=self.model.input, outputs=outputs)'''

        features = self.model.predict(self.dataset, verbose=1, steps=self.val_steps)

        if not isinstance(features, (list, tuple)):
            features = [features]

        k = 0
        pooled_metric = 0.
        best_pooled_metric = 0.
        for feature, metric in zip(features, self.metrics):
            with tf.device('/device:CPU:0'):
                computed_metric = metric.computeGlobal(self.to_tensor(feature),
                                                       self.labels[k])

            monitored_metric = metric.toScalar(computed_metric)

            metric.set_weights([monitored_metric])
            metric.printResult(computed_metric)

            if monitored_metric > self.best_monitored[k]:
                self.best_monitored[k] = monitored_metric
                self.best_so_far[k] = computed_metric

            print('best so far:')
            metric.printResult(self.best_so_far[k])

            pooled_metric += monitored_metric
            best_pooled_metric += self.best_monitored[k]

            k += 1

        pooled_metric = pooled_metric / k
        best_pooled_metric = best_pooled_metric / k

        print('pooled metric: {}'.format(pooled_metric))
        print('best pooled metric so far: {}'.format(best_pooled_metric))


    def on_epoch_end(self, epoch, logs=None):

        if (epoch + 1) % self.compute_freq == 0:
            self._computeMetric()

        self.step += 1

    def on_train_end(self, logs=None):
        if self.step % self.compute_freq:
            self._computeMetric()
