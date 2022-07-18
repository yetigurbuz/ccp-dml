import tensorflow as tf
from .global_metric import GlobalMetric
from .metric_functions import allRetrievalMetrics


class RecallAtK(GlobalMetric):
    def __init__(self, atK, monitored_metric='map', monitor_k=0, name=None, **kwargs):

        if not isinstance(atK, (list, tuple)):
            atK = [atK]

        if name is None:
            name = ''
        if monitored_metric == 'map' or monitored_metric == 'MAP':
            name += 'MAPat{}'.format(atK[monitor_k])
        else:
            name += 'Rat{}'.format(atK[monitor_k])

        self.atK = atK
        self.monitored_metric = monitored_metric
        self.monitor_k = monitor_k

        self._map_label = 'MAP@{}: '.format(atK)
        self._recall_label = 'R@{}: '.format(atK)

        super(RecallAtK, self).__init__(metric_shape=(2 * len(atK), ), name=name, **kwargs)

    @tf.function
    def compute(self, embeddings, labels, ref_embeddings=None, ref_labels=None):
        return mAPandR(embeddings=embeddings, labels=labels,
                       ref_embeddings=ref_embeddings, ref_labels=ref_labels,
                       atK=tf.convert_to_tensor(self.atK, dtype=tf.int32))

    def toScalar(self, computed_metric):
        [map, recall] = tf.split(computed_metric, 2, axis=0)
        if self.monitored_metric == 'map' or self.monitored_metric == 'MAP':
            return map[self.monitor_k]
        else:
            # assign R@k
            return recall[self.monitor_k]

    def printResult(self, computed_metric):
        computed_metric = tf.round(computed_metric * 1e5) / 1e3
        [map, recall] = tf.split(computed_metric, 2, axis=0)

        map_text = self._map_label + '{} %'.format(map)
        recall_text = self._recall_label + '{} %'.format(recall)
        print('\033[1;35m' + map_text + '\033[0m')
        print('\033[1;34m' + recall_text + '\033[0m')
        self.print_text = map_text + '\n' + recall_text

    def get_config(self):
        config = super(RecallAtK, self).get_config()
        config.update({'atK': self.atK,
                       'monitored_metric': self.monitored_metric,
                       'monitor_k': self.monitor_k
                       })
        return config
