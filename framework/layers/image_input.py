import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class AuxiliaryInput(tf.keras.layers.Layer):

    def __init__(self,
                 image_shape,
                 num_aux_samples=0,
                 seed=6387,
                 name=None,
                 **kwargs):
        super(AuxiliaryInput, self).__init__(name=name, **kwargs)

        self._image_shape = image_shape
        self._num_aux_samples = num_aux_samples
        self._seed = seed

        bbox_base = tf.constant([0.0, 0.0, 1.0, 1.0],
                                dtype=tf.float32,
                                shape=[1, 1, 4])

        self._anchor_boxes = None
        if num_aux_samples > 0:
            self._anchor_boxes = tf.squeeze(
                tf.concat(
                    [tf.image.sample_distorted_bounding_box(
                        image_size=self._image_shape,
                        bounding_boxes=bbox_base,
                        seed=self._seed,
                        min_object_covered=0,
                        aspect_ratio_range=(.75, 1.33),
                        area_range=(.16, 1.),
                        max_attempts=100,
                        use_image_if_no_bounding_boxes=True)[2] for _ in range(self._num_aux_samples)],
                    axis=0)
                )

    def call(self, inputs, training, **kwargs):

        if training:
            return inputs
        elif self._num_aux_samples > 0:
            batch_size = tf.shape(inputs)[0]

            crop_boxes = tf.repeat(self._anchor_boxes, repeats=batch_size, axis=0)
            box_indcs = tf.tile(tf.range(batch_size), multiples=[self._num_aux_samples])

            aux_images = tf.image.crop_and_resize(
                image=inputs,
                boxes=crop_boxes,
                box_indices=box_indcs,
                crop_size=self._image_shape[:2],
                method='bilinear',
                extrapolation_value=0.0
            )

            aux_images = tf.image.random_flip_left_right(aux_images, seed=self._seed)

            return tf.concat([inputs, aux_images], axis=0)

        else:
            return inputs



    def compute_output_shape(self, input_shape):
        output_shape = input_shape

        return output_shape

    def get_config(self):
        config = super(AuxiliaryInput, self).get_config()
        config.update({'image_shape': self._image_shape,
                       'num_aux_samples': self._num_aux_samples,
                       'seed': self._seed
                       }
                      )
        return config