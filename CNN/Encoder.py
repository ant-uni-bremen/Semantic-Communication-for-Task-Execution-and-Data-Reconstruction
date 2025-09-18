import tensorflow as tf
from keras import Model,layers, Sequential, regularizers

class CnnEncoder(Model):
    def __init__(self, split_image_into, enc_out_dec_inp, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.enc_out_dec_inp = enc_out_dec_inp
        self.split_image_into = split_image_into

        # Define the encoder pipeline
        self._encoder = Sequential([
            layers.BatchNormalization(),
            layers.Conv2D(16, (5, 5), padding="same", strides=(2, 2), kernel_regularizer=regularizers.l2(0.0001)),
            layers.PReLU(alpha_initializer="zeros"),

            layers.Conv2D(32, (5, 5), padding="same", strides=(2, 2), kernel_regularizer=regularizers.l2(0.0001)),
            layers.PReLU(alpha_initializer="zeros"),

            layers.Conv2D(32,(5, 5), padding="same", strides=(1, 1), kernel_regularizer=regularizers.l2(0.0001)),
            layers.PReLU(alpha_initializer="zeros"),

            layers.Conv2D(32, (5, 5), padding="same", strides=(1, 1), kernel_regularizer=regularizers.l2(0.0001)),
            layers.PReLU(alpha_initializer="zeros"),
            
            layers.Conv2D(32, (5, 5), padding="same", strides=(1, 1), kernel_regularizer=regularizers.l2(0.0001)),
            layers.PReLU(alpha_initializer="zeros"),

            layers.Conv2D(self.enc_out_dec_inp,(5,5),padding="same", strides=(1, 1), 
                          kernel_regularizer=regularizers.l2(0.0001)),
            layers.PReLU(alpha_initializer="zeros"),

            layers.BatchNormalization(),
            layers.Flatten(),

            layers.Dense(self.enc_out_dec_inp//(self.split_image_into), kernel_regularizer=regularizers.l2(0.0001)),
        ], name='CnnEncoder')

    def build(self, input_shape):
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected input shape (batch_size,height, width, channels), but got {input_shape}")
        super().build(input_shape)

    def call(self, inputs):
        return self._encoder(inputs)

    def get_config(self):
        config = super(CnnEncoder, self).get_config()
        config.update({
            'enc_out_dec_inp': self.enc_out_dec_inp,
            'split_image_into': self.split_image_into
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Reconstruct the encoder from the configuration
        return cls(
            split_image_into=config['split_image_into'],
            enc_out_dec_inp=config['enc_out_dec_inp'],
            name=config['name']
        )