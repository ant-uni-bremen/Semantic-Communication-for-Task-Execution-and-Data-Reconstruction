from keras import layers
import tensorflow as tf

class ConstMultiplierLayer(layers.Layer):
    def __init__(self, shape=(32, 32, 3), k_config=None, **kwargs):
        super().__init__(**kwargs)
        self.k = self.add_weight(
            name='k',
            shape=shape,
            initializer='ones',
            dtype='float32',
            trainable=True, )

    def call(self, x):
        return tf.math.multiply(self.k, x)

    def get_config(self):
        config = super(ConstMultiplierLayer, self).get_config()
        return config