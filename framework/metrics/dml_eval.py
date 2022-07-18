import tensorflow as tf
from .global_metric import GlobalMetric
from .metric_functions import atKMetrics

from ..configs import default
from ..configs.config import CfgNode as CN

DMLEval_cfg = CN()
DMLEval_cfg.monitored_metric = 'MAP'

default.cfg.validation.DMLEval = DMLEval_cfg

class DMLEval(GlobalMetric):
    def __init__(self, monitored_metric='map', name=None, **kwargs):

        atK_text = '{}'.format('R')

        if name is None:
            name = ''
        if monitored_metric.lower() == 'precision':
            name += 'Pat{}'.format(atK_text)
        elif monitored_metric.lower() == 'recall':
            name += 'Rat{}'.format(atK_text)
        else:
            name += 'MAPat{}'.format(atK_text)


        self.monitored_metric = monitored_metric.lower()

        super(DMLEval, self).__init__(metric_shape=(3, ), name=name, **kwargs)

    @tf.function
    def compute(self, embeddings, labels, ref_embeddings=None, ref_labels=None, has_self_match=True):
        m = atKMetrics(
            embeddings=embeddings, labels=labels,
            ref_embeddings=ref_embeddings, ref_labels=ref_labels,
            atK=tf.convert_to_tensor([1], dtype=tf.int32),
            has_self_match=has_self_match
        )

        metrics_to_return = tf.gather(m, [1, 3, 4])  # map, prec, rec

        return metrics_to_return # map, prec, rec

    def toScalar(self, computed_metric):
        [map, precision, recall] = tf.squeeze(tf.split(computed_metric, 3, axis=0))
        if self.monitored_metric == 'precison':
            # assign P@k
            return precision
        elif self.monitored_metric == 'recall':
            # assign R@k
            return recall
        else:   # default is map
            return map


    def printResult(self, computed_metric):

        computed_metric = tf.round(computed_metric * 1e5) / 1e3
        [map, precision, recall] = tf.squeeze(tf.split(computed_metric, 3, axis=0))

        black = lambda s: '\033[1;30m' + s + '\033[0m'
        blue = lambda s: '\033[1;34m' + s + '\033[0m'
        yellow = lambda s: '\033[1;33m' + s + '\033[0m'
        pink = lambda s: '\033[1;35m' + s + '\033[0m'

        metric_text = black('[') + \
                      blue('R@1') + black(', ') + \
                      yellow('P@R') + black(', ') + \
                      pink('MAP@R') + black(']: ') + \
                      black('[') + blue('{:.3f}'.format(recall)) + black(', ') + \
                      yellow('{:.3f}'.format(precision)) + black(', ') + \
                      pink('{:.3f}'.format(map)) + black('] %')


        self.print_text = metric_text

        print(metric_text)

        plain_text = '[R@1, P@R, MAP@R]: [{:.3f}, {:.3f}, {:.3f}] %'.format(recall, precision, map)

        self.plain_print_text = plain_text

        return plain_text

    def get_config(self):
        config = super(DMLEval, self).get_config()
        config.update({'monitored_metric': self.monitored_metric})
        return config
