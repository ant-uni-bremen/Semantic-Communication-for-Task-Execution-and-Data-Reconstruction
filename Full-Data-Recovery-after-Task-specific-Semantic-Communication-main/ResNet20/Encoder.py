import tensorflow as tf
from keras import Model,layers, Sequential, regularizers

class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.strides = strides

        # Residual path (main convolutional layers)
        self._residual_block = Sequential([
            layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same',
                          kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization()
        ])

        # Shortcut path (adjust dimensions if necessary)
        if strides != 1:
            self._shortcut = Sequential([
                layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same',
                              kernel_regularizer=regularizers.l2(0.0001)),
                layers.BatchNormalization()
            ])
        else:
            self._shortcut = lambda x: x  # Identity mapping

    def call(self, inputs, *args, **kwargs):
        shortcut = self._shortcut(inputs)
        residual = self._residual_block(inputs)
        return layers.ReLU()(layers.Add()([shortcut, residual]))  # Final activation

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides
        })
        return config

# ResNet20Encoder
class ResNet20Encoder(Model):
    def __init__(self, split_image_into, enc_out_dec_inp, *, activity_regularizer=None,
                 trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable,
                         dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.enc_out_dec_inp = enc_out_dec_inp
        self.split_image_into = split_image_into

        # Build the encoder architecture.
        self._encoder = Sequential([
            # Initial conv
            layers.Conv2D(64, 3, strides=1, padding='same',
                          kernel_regularizer=regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Group 1: 32×32
            ResidualBlock(64),
            ResidualBlock(64),

            # Group 2: 16×16
            ResidualBlock(128, strides=2),
            ResidualBlock(128),
            ResidualBlock(128),

            # Group 3: 8×8
            ResidualBlock(256, strides=2),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Flatten and Dense for bottleneck
            layers.Flatten(),
            layers.Dense(self.enc_out_dec_inp // (self.split_image_into),
                         activation=None,
                         activity_regularizer=regularizers.l2(0.001)),
        ], name='resnet20_encoder')

    def build(self, input_shape):
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected input shape (batch_size,height, width, channels), but got {input_shape}")
        super().build(input_shape)

    def call(self, inputs):
        return self._encoder(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'enc_out_dec_inp': self.enc_out_dec_inp,
            'split_image_into': self.split_image_into
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            split_image_into=config['split_image_into'],
            enc_out_dec_inp=config['enc_out_dec_inp'],
            name=config['name']
        )
