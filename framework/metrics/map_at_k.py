import tensorflow as tf
from .global_metric import GlobalMetric
from .metric_functions import mAP

from ..configs import default
from ..configs.config import CfgNode as CN

MeanAveragePrecisionAtK_cfg = CN()
MeanAveragePrecisionAtK_cfg.atK = [1, 8, 16, 32, 64]
MeanAveragePrecisionAtK_cfg.monitor_k = -1

default.cfg.validation.MeanAveragePrecisionAtK = MeanAveragePrecisionAtK_cfg

class MeanAveragePrecisionAtK(GlobalMetric):
    def __init__(self, atK, monitor_k=0, name=None, **kwargs):

        if not isinstance(atK, (list, tuple)):
            atK = [atK]

        if name is None:
            name = 'MAPat{}'.format(atK[monitor_k])

        self.atK = atK
        self.monitor_k = monitor_k

        self._map_label = 'MAP@{}: '.format(atK)

        super(MeanAveragePrecisionAtK, self).__init__(metric_shape=(len(atK), ), name=name, **kwargs)

    @tf.function
    def compute(self, embeddings, labels, ref_embeddings=None, ref_labels=None):

        return mAP(embeddings=embeddings, labels=labels,
                   ref_embeddings=ref_embeddings, ref_labels=ref_labels,
                   atK=tf.convert_to_tensor(self.atK, dtype=tf.int32))

    def toScalar(self, computed_metric):
        # assign MAP@k
        return computed_metric[self.monitor_k]

    def printResult(self, computed_metric):
        computed_metric = tf.round(computed_metric * 1e5) / 1e3
        self.print_text = self._map_label + '{} \%'.format(computed_metric)
        print('\033[1;35m' + self.print_text + '\033[0m')

    def get_config(self):
        config = super(MeanAveragePrecisionAtK, self).get_config()
        config.update({'atK': self.atK,
                       'monitored_metric': self.monitored_metric,
                       'monitor_k': self.monitor_k
                       })
        return config

