import tensorflow as tf
from .global_metric import GlobalMetric
from .metric_functions import atKMetrics

from ..configs import default
from ..configs.config import CfgNode as CN

AllAtK_cfg = CN()
AllAtK_cfg.atK = [1]
AllAtK_cfg.monitor_k = -1
AllAtK_cfg.monitored_metric = 'MAP'

default.cfg.validation.AllAtK = AllAtK_cfg

class AllAtK(GlobalMetric):
    def __init__(self, atK, monitored_metric='map', monitor_k=-1, name=None, **kwargs):

        if not isinstance(atK, (list, tuple)):
            atK = [atK]

        atK_text = '{}'.format('pN' if monitor_k == -1 else atK[monitor_k])

        if name is None:
            name = ''
        if monitored_metric.lower() == 'precision':
            name += 'Pat{}'.format(atK_text)
        elif monitored_metric.lower() == 'recall':
            name += 'Rat{}'.format(atK_text)
        else:
            name += 'MAPat{}'.format(atK_text)


        self.atK = atK
        self.monitored_metric = monitored_metric.lower()
        self.monitor_k = monitor_k

        atK_list = [k for k in atK] + ['pN']
        self._map_label = 'MAP@{}: '.format(atK_list)
        self._recall_label = 'R@{}: '.format(atK_list)
        self._precison_label = 'P@{}: '.format(atK_list)

        super(AllAtK, self).__init__(metric_shape=(3 * (len(atK) + 1), ), name=name, **kwargs)

    @tf.function
    def compute(self, embeddings, labels, ref_embeddings=None, ref_labels=None):
        return atKMetrics(embeddings=embeddings, labels=labels,
                          ref_embeddings=ref_embeddings, ref_labels=ref_labels,
                          atK=tf.convert_to_tensor(self.atK, dtype=tf.int32))

    def toScalar(self, computed_metric):
        [map, precision, recall] = tf.split(computed_metric, 3, axis=0)
        if self.monitored_metric == 'precison':
            # assign P@k
            return precision[self.monitor_k]
        elif self.monitored_metric == 'recall':
            # assign R@k
            return recall[self.monitor_k]
        else:   # default is map
            return map[self.monitor_k]


    def printResult(self, computed_metric):
        computed_metric = tf.round(computed_metric * 1e5) / 1e3
        [map, precision, recall] = tf.split(computed_metric, 3, axis=0)

        map_text = self._map_label + '{} %'.format(map)
        precision_text = self._precison_label + '{} %'.format(precision)
        recall_text = self._recall_label + '{} %'.format(recall)
        print('\033[1;35m' + map_text + '\033[0m')
        print('\033[1;33m' + precision_text + '\033[0m')
        print('\033[1;34m' + recall_text + '\033[0m')
        self.print_text = map_text + '\n' + precision_text + '\n' + recall_text

    def get_config(self):
        config = super(AllAtK, self).get_config()
        config.update({'atK': self.atK,
                       'monitored_metric': self.monitored_metric,
                       'monitor_k': self.monitor_k
                       })
        return config
