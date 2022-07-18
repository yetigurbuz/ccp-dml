import tensorflow as tf

class BaseModel(object):

    def __init__(self, inputs=None, outputs=None, name=None):

        self.ensemble_id = 0
        self.num_classes = 0
        self.training_callbacks = list()
        self.gradient_transformers = list()
        self.warm_up_gradient_transformers = list()
        self.learning_rate_multipliers = dict()
        self.arch = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

