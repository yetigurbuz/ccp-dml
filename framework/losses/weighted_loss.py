import tensorflow as tf


class WeightedLoss(tf.keras.losses.Loss):
    def __init__(self, weight, base_loss, **kwargs):
        super(WeightedLoss, self).__init__(**kwargs)

        self._weight = weight
        self._base_loss = base_loss

    @tf.function
    def call(self, y_true, y_pred):
        loss = self._base_loss(y_true, y_pred)
        return self._weight * loss

'''class AugmentedSparseCategoricalCrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, use_augmented_batch_size=False, name='axe_loss', **kwargs):
        super(AugmentedSparseCategoricalCrossEntropyLoss, self).__init__(name=name, **kwargs)

        self.use_augmented_batch_size = use_augmented_batch_size
        self.xe_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     reduction=tf.keras.losses.Reduction.SUM)

    @tf.function
    def call(self, y_true, y_pred):
        batch_size = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        sum_loss = self.xe_loss(y_true, y_pred)
        if self.use_augmented_batch_size:
            loss = sum_loss / (2.0 * batch_size)
        else:
            loss = sum_loss / batch_size

        return loss
'''
