import tensorflow as tf

from framework.utilities import proxy_utils
from framework import losses


def MDRLoss(model,
            loss_class,
            mdr_weight=1.,
            nu_level_reg=0.01,
            **kwargs):

    class LossWithMDR(getattr(losses, loss_class)):
        def __init__(self, model, weight, **kwargs):
            self.levels = model.add_weight(
                name='mdr/levels',
                shape=(1, 1, 3),
                dtype=tf.float32,
                initializer=proxy_utils.FromTensorInitializer(tf.constant([[[-3., 0., 3.]]], tf.float32)),
                trainable=True)

            self.momentum = 0.9

            self.momented_mean = tf.Variable(initial_value=0.,
                                             trainable=False, shape=(), dtype=tf.float32)
            self.momented_std = tf.Variable(initial_value=0.,
                                            trainable=False, shape=(), dtype=tf.float32)

            # The variable is used to check whether momented_mean and momented_std are initialized
            self.init = tf.Variable(initial_value=0,
                                    trainable=False, shape=(), dtype=tf.int32)

            self.mdr_weight = weight
            self.nu_level_reg = nu_level_reg

            super(LossWithMDR, self).__init__(model=model, **kwargs)

            self.normalize_embeddings = False

        def initStatistics(self, sample_mean, sample_std):
            momented_mean = self.momented_mean.assign(sample_mean)
            momented_std = self.momented_std.assign(sample_std)
            self.init.assign_add(1)

            return momented_mean, momented_std

        def updateStatistics(self, sample_mean, sample_std):
            momented_mean = (1. - self.momentum) * sample_mean + self.momentum * self.momented_mean
            momented_std = (1. - self.momentum) * sample_std + self.momentum * self.momented_std
            momented_mean = self.momented_mean.assign(momented_mean)
            momented_std = self.momented_std.assign(momented_std)

            return momented_mean, momented_std

        def call(self, y_true, y_pred):
            labels, embeddings = y_true, y_pred

            pdists = self.pdist_fn(embeddings=embeddings,
                                   ref_embeddings=None,
                                   squared=self.squared_distance)

            #tf.print('\nmean before: ', self.momented_mean)
            #tf.print('\nstd before: ', self.momented_std)
            sample_mean_0 = tf.reduce_mean(pdists)
            sample_std = tf.stop_gradient(tf.math.reduce_std(pdists))
            embeddings = embeddings / sample_mean_0

            sample_mean = tf.stop_gradient(sample_mean_0)

            momented_mean, momented_std = tf.cond(
                pred=tf.equal(self.init, 0),
                true_fn=lambda: self.initStatistics(sample_mean, sample_std),
                false_fn=lambda: self.updateStatistics(sample_mean, sample_std))

            #tf.print('\nmean after: ', self.momented_mean)
            #tf.print('\nstd after: ', self.momented_std)

            normalized_dist = tf.expand_dims((pdists - momented_mean) / momented_std, axis=-1)
            difference = tf.reduce_min(tf.abs(normalized_dist - self.levels), axis=-1)
            mdr_loss = tf.reduce_mean(difference)

            #tf.print('\n levels: ', self.levels)

            loss = super(LossWithMDR, self).call(y_true, embeddings)

            level_reg_loss =  tf.reduce_sum(tf.square(self.levels))

            return loss + self.mdr_weight * mdr_loss + self.nu_level_reg * level_reg_loss

    return LossWithMDR(model=model, weight=mdr_weight, **kwargs)
