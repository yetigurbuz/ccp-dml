import tensorflow as tf
from . import miners
from .base_losses import GenericPairBasedLoss

# !TODO: UPDATE IMPLEMENTATION ESPECIALL BETA LR THING
class MarginLoss(GenericPairBasedLoss):
    def __init__(self,
                 alpha_margin=.2,
                 beta_class=1.2,
                 beta_regularization_coeff_nu=0,
                 beta_lr=None,
                 num_classes=0,
                 model=None,
                 **kwargs):

        self.alpha = alpha_margin

        if num_classes > 0:
            if not isinstance(model, tf.keras.models.Model):
                raise ValueError('Expected {} got {}'.format(tf.keras.models.Model, model))

            self.beta = model.add_weight(name='margin_loss/beta'.format(model.name),
                                         shape=(num_classes, 1),
                                         dtype=tf.float32,
                                         initializer=tf.keras.initializers.Constant(beta_class),
                                         trainable=True)
            self.has_beta_per_class = True
        else:
            self.beta = beta_class
            self.has_beta_per_class = False

        self.nu = beta_regularization_coeff_nu
        self.beta_lr = beta_lr

        super(MarginLoss, self).__init__(model=model,
                                         num_classes=num_classes,
                                         **kwargs)

    def _computeLoss(self, pdists, pos_pair_mask, neg_pair_mask, ref_labels):

        if self.has_beta_per_class:

            beta = tf.nn.embedding_lookup(
                self.beta,
                tf.cast(tf.squeeze(ref_labels), dtype=tf.int32)
            )

            '''beta = tf.gather(
                self.beta,
                tf.cast(tf.reshape(ref_labels, shape=(-1, 1)), dtype=tf.int32))'''
                #self.beta[tf.cast(tf.reshape(ref_labels, shape=(self.classes_per_batch,)), dtype=tf.int32)]
        else:
            beta = self.beta

        beta_reg_loss = self.nu * tf.reduce_mean(beta)

        pos_term = (pdists - beta + self.alpha) * pos_pair_mask
        neg_term = (beta + self.alpha - pdists) * neg_pair_mask

        pos_term_mask = miners.nonZeroTerms(pos_term)
        neg_term_mask = miners.nonZeroTerms(neg_term)

        if self.avg_nonzero_only:
            num_pos_term = tf.reduce_sum(pos_term_mask) + 1e-16
            num_neg_term = tf.reduce_sum(neg_term_mask) + 1e-16
            num_terms = num_pos_term + num_neg_term + 1e-16

            margin_loss = tf.add(tf.reduce_sum(pos_term * pos_term_mask) / num_pos_term,
                                 tf.reduce_sum(neg_term * neg_term_mask) / num_neg_term)
        else:
            margin_loss = tf.reduce_mean(pos_term * pos_term_mask + neg_term * neg_term_mask)

        return margin_loss + beta_reg_loss


# MARGIN IDEA
'''
class TransformationLayer(tf.keras.layers.Layer):
    def __init__(self, dim, name='transformation'):
        super(TransformationLayer, self).__init__(name=name)

        self._input_dim = dim
        self._output_dim = 1
        self.initializer = tf.random_uniform_initializer()


    def build(self, input_shape):


        self._transform_vector = self.add_weight(shape=(self._output_dim, self._input_dim),
                                          trainable=True,
                                          dtype=tf.float32,
                                          initializer=self.initializer)


    def call(self, inputs, **kwargs):
        y = tf.reduce_sum(inputs * self._transform_vector, axis=-1, keepdims=True)
        return y

class MetricLearningLoss(tf.keras.losses.Loss):
    def __init__(self, dim=1, b=0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='loss_with_trainable_variable'):
        super(MetricLearningLoss, self).__init__(reduction=reduction, name=name)

        self.b = b

    def call(self, y_true, y_pred):

        y_hat = y_pred + self.b

        err = y_true - y_hat

        return tf.reduce_sum(.5 * err * err, axis=-1)

class TestLossClass(tf.keras.models.Model):

    def __init__(self, dim=2):
        super(TestLossClass, self).__init__()
        self._dim = dim

        self.transform = TransformationLayer(dim=self._dim)



    def call(self, inputs, **kwargs):
        y = self.transform(inputs)
        return y

def createRegressionDataset(dim=2):

    a = 2 * np.random.random(size=(1, dim)) - 1
    b = 2 * np.random.random(size=(1,)) - 1

    x = 20 * np.random.random(size=(1024, dim)) - 10

    sigma = 0.1

    y = np.sum(a * x, axis=-1) + b + sigma * np.random.randn(1024)


    train_data = (x[:512], y[:512])
    val_data = (x[512:], y[512:])

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(512).repeat().batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_data).batch(32)

    return train_dataset, val_dataset, a, b


model = TestLossClass()
model.add_weight(shape=(1, 1),
                 trainable=True,
                 dtype=tf.float32,
                 initializer=tf.zeros_initializer())


loss = MetricLearningLoss(b=model.weights[-1])
model.compile(optimizer='sgd',
              loss=loss)

train_dataset, val_dataset, a, b = createRegressionDataset()

model.fit(train_dataset, epochs=10, steps_per_epoch=512/32,
          validation_data=val_dataset)'''
