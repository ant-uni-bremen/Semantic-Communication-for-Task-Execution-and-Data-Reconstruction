from keras import Model,layers, Sequential, regularizers

class CnnDecoder(Model):
    def __init__(self, split_image_into, enc_out_dec_inp, num_classes, output_channels, *,
                 activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.enc_out_dec_inp = enc_out_dec_inp
        self.split_image_into = split_image_into
        self.num_classes = num_classes
        self.output_channels = output_channels
        
        # Create sub-models in __init__
        self._classification = self._classification_image()
        self._reconstruction_image_encoder = self._reconstruction_image()
    
    def build(self, input_shape):
        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected input shape (batch_size,enc_out_dec_in), but got {input_shape}")
        super().build(input_shape)
        
    def _reconstruction_image(self):
        return Sequential([
            layers.Dense(self.enc_out_dec_inp),  # Ensure input matches the encoder's output
            layers.PReLU(alpha_initializer='Zeros'),
            
            # Fully connected to prepare for reshaping
            layers.Dense(8 * 8 * 256),  # Units match Reshape dimensions
            layers.PReLU(alpha_initializer='Zeros'),
            
            # Reshape to (8, 8, 256)
            layers.Reshape((8, 8, 256)),  
            
            # First upsampling to (16, 16, 256)
            layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'),
            layers.PReLU(alpha_initializer='Zeros'),
            
            # Second upsampling to (32, 32, 128)
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            layers.PReLU(alpha_initializer='Zeros'),
            
            # Third upsampling to (32, 32, 64) - no stride increase, just reducing filters
            layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'),
            layers.PReLU(alpha_initializer='Zeros'),
            
            # Fourth upsampling to (32, 32, 32) - preparing for final output
            layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='same'),
            layers.PReLU(alpha_initializer='Zeros'),
            
            # Final layer to output shape (32, 32, 3)
            layers.Conv2DTranspose(self.output_channels, kernel_size=3, strides=1, padding='same', 
                                activation='sigmoid', name="reconstructed_image")
        ], name="cnn_image_reconstruction")

    def _classification_image(self):
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
        return {
            "reconstructed_image": self._reconstruction_image_encoder(inputs),
            "classification_output": self._classification(inputs),
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
            name=config.get('name', None)
        )
