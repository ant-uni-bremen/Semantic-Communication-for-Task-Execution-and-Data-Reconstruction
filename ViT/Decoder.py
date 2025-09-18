from keras import Model,layers,Sequential,regularizers
import tensorflow as tf

class ResidualBlock(Model):
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


class ViTDecoder(Model):
    def __init__(self, enc_out_dec_inp,split_image_into, num_classes, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.enc_out_dec_inp = enc_out_dec_inp
        self.num_classes = num_classes
        self.output_channels = output_channels
        self.split_image_into = split_image_into

        self.reconstruction_image = self.reconstruction_image_decoder()
        self.classification = self.classification_image()

    def reconstruction_image_decoder(self):
        return Sequential([
            layers.Dense(self.enc_out_dec_inp),
            layers.ReLU(),
            layers.Dense(8* 8* 256),
            layers.ReLU(),
            layers.Reshape((8, 8, 256)),

            ResidualBlock(filters=256),
            ResidualBlock(filters=128, use_transpose=True),  # Upsample to (16, 16)
            ResidualBlock(filters=64,  use_transpose=True),   # Upsample to (32, 32)
            ResidualBlock(filters=32),

            layers.Conv2DTranspose(self.output_channels, kernel_size=3, strides=1, padding='same', activation='sigmoid', name="reconstructed_image")
        ], name="vit_image_reconstruction")

    def classification_image(self):
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
    
    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError("Expected input shape (batch_size, features)")
        self.reconstruction_image.build(input_shape)
        self.classification.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return {
            "reconstructed_image":  self.reconstruction_image(inputs), 
            "classification_output": self.classification(inputs)
        }

    def compute_output_shape(self, input_shape):
        return {
            "reconstructed_image": (None, 32, 32, self.output_channels), 
            "classification_output": (None, self.num_classes)
        }

    def get_config(self):
        config = super(ViTDecoder, self).get_config()
        config.update({
            'enc_out_dec_inp': self.enc_out_dec_inp,
            'split_image_into': self.split_image_into,
            'num_classes':self.num_classes,
            'output_channels':self.output_channels
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            split_image_into=config['split_image_into'],
            enc_out_dec_inp=config['enc_out_dec_inp'],
            num_classes=config['num_classes'],
            output_channels=config['output_channels'],
            name=config['name']  # Ensure no KeyError
        )
