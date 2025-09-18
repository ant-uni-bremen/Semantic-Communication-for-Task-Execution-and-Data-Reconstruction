from keras import Model,layers, Sequential,regularizers
import tensorflow as tf


class ResidualBlock(layers.Layer):
    def __init__(self, filters, use_transpose=False):
        super().__init__()
        self.filters = filters
        self.use_transpose = use_transpose

        # Main path with either Conv2D or Conv2DTranspose
        if use_transpose:
            self.main_path = Sequential([
                layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(filters, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization()
            ])
        else:
            self.main_path = Sequential([
                layers.Conv2D(filters, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(filters, kernel_size=3, strides=1, padding='same'),
                layers.BatchNormalization()
            ])

        # Shortcut path to match dimensions
        self.shortcut = Sequential()
        if use_transpose:
            self.shortcut.add(layers.Conv2DTranspose(filters, kernel_size=1, strides=2, padding='same'))
        else:
            self.shortcut.add(layers.Conv2D(filters, kernel_size=1, strides=1, padding='same'))

    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        main_output = self.main_path(inputs)
        return layers.ReLU()(layers.Add()([shortcut, main_output]))

class ResNet14Decoder(Model):
    def __init__(self, split_image_into, output_channels, enc_out_dec_inp=1024, num_classes=10):
        super(ResNet14Decoder, self).__init__()
        self.split_image_into = split_image_into
        self.enc_out_dec_inp = enc_out_dec_inp
        self.output_channels = output_channels
        self.num_classes = num_classes

        # Image reconstruction path
        self.image_reconstruction_model = self._build_reconstruction_model()

        # Classification path that follows ResNet14 decoder architecture
        self.classification_model = self._build_classification_model()

    def _build_reconstruction_model(self):
        return Sequential([
            layers.Dense(self.enc_out_dec_inp),
            layers.ReLU(),
            layers.Dense(8 * 8 * 256),
            layers.ReLU(),
            layers.Reshape((8, 8, 256)),

            ResidualBlock(filters=256),
            ResidualBlock(filters=128, use_transpose=True),  # Upsample to (16, 16)
            ResidualBlock(filters=64, use_transpose=True),   # Upsample to (32, 32)
            ResidualBlock(filters=32),

            layers.Conv2DTranspose(self.output_channels, kernel_size=3, strides=1, padding='same', activation='sigmoid', name="reconstructed_image")
        ], name="resnet14_image_reconstruction")

    def _build_classification_model(self):
        """Builds a ResNet14-like classification model."""
        return Sequential([
        layers.Dense(self.enc_out_dec_inp, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.5),

        layers.Dense(256, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.3),

        layers.Dense(128, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.01),
        layers.Dropout(0.3),

        layers.Dense(self.num_classes, activation='softmax', name="classification_output",kernel_initializer='he_normal')
    ], name='classification')

    def call(self, inputs):
        # Ensure inputs are properly shaped
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, axis=0)

        # Generate outputs
        reconstructed_image = self.image_reconstruction_model(inputs)
        classification_output = self.classification_model(inputs)

        return {
            "reconstructed_image": reconstructed_image,
            "classification_output": classification_output,
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            'enc_out_dec_inp': self.enc_out_dec_inp,
            'split_image_into': self.split_image_into,
            'num_classes': self.num_classes,
            'output_channels': self.output_channels
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            split_image_into=config['split_image_into'],
            enc_out_dec_inp=config['enc_out_dec_inp'],
            num_classes=config['num_classes'],
            output_channels=config['output_channels'],
            name=config['name']
        )