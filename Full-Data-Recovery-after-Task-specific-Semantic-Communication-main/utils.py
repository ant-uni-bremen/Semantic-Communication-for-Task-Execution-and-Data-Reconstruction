from keras import layers,Sequential,regularizers
import tensorflow as tf
import numpy as np

class NormalizationLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected input shape (batch_size, enc_out_dec_inp), but got {input_shape}")
        super().build(input_shape)

    def call(self, input):
        def _normalizeEachSignal(signal):
            # Compute mean and variance
            mean_over_whole_batch = tf.reduce_mean(signal, axis=-1, keepdims=True)
            var_over_whole_batch = tf.math.reduce_variance(signal, axis=-1, keepdims=True)
            # Compute power factor
            power_factor = var_over_whole_batch + tf.square(mean_over_whole_batch)
            power_factor_sqrt = tf.sqrt(power_factor)
            # Compute normalization factor
            normalization_factor = tf.math.divide_no_nan(1.0, power_factor_sqrt)
            # Normalize the input signal
            normalized_input = signal * normalization_factor
            return normalized_input

        return _normalizeEachSignal(signal=input)

    def get_config(self):
        config = super(NormalizationLayer, self).get_config()
        config.update({})
        return config

class splitInputImages(layers.Layer):
    def __init__(self, split_image_into, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.split_image_into = np.sqrt(split_image_into)
        
    def get_config(self):
        config = super(splitInputImages, self).get_config()
        config.update({'split_image_into': self.split_image_into})
        return config

    def call(self, input):
        # Dynamically handle input dimensions
        input_shape = tf.shape(input)
        height = input_shape[1]
        width = input_shape[2]

        # Ensure dimensions are divisible by `split_image_into`
        tf.debugging.assert_equal(
            tf.math.floormod(height, self.split_image_into),
            0,
            message="Height is not divisible by split_image_into."
        )
        tf.debugging.assert_equal(
            tf.math.floormod(width, self.split_image_into),
            0,
            message="Width is not divisible by split_image_into."
        )

        # Compute new dimensions
        split_h = height // self.split_image_into
        split_w = width // self.split_image_into

        # Reshape the input tensor to split it into smaller patches
        split_image = tf.reshape(
            input,
            (-1, self.split_image_into, split_h, self.split_image_into, split_w, input.shape[3])
        )

        # Reorder dimensions: (batch_size, num_splits, split_h, split_w, channels)
        split_image = tf.transpose(split_image, perm=[0, 1, 3, 2, 4, 5])
        split_image = tf.reshape(
            split_image,
            (-1, self.split_image_into**2, split_h, split_w, input.shape[3])
        )

        return split_image
    
class CustomImageAugmentation(layers.Layer):
    def __init__(self, *, activity_regularizer=None, trainable=True,
                 dtype=None, autocast=True, name=None, **kwargs):
        super().__init__(activity_regularizer=activity_regularizer,
                         trainable=trainable, dtype=dtype,
                         autocast=autocast, name=name, **kwargs)
        self._imageAugmentation = Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=(0.0, 0.02)),
            layers.RandomZoom(
                height_factor=(0.0, 0.2), 
                width_factor=(0.0, 0.2)
            ),
        ], name="Image_augmentation")
        
    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError('Input shape must be [batch_size, splits, width, height, 3]')
        return super().build(input_shape)
    
    def call(self, inputs):
        images = tf.transpose(inputs, [1, 0, 2, 3, 4])
        augmented_flat = tf.map_fn(lambda img: self._imageAugmentation(img), images)
        return tf.transpose(augmented_flat, [1, 0, 2, 3, 4])

# Basic ResidualBlock with skip connection.
class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1,use_transpose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.strides = strides
        self.use_transpose = use_transpose
        # Main branch: two convolutional layers.
        if use_transpose:
            # Main branch: Conv2DTranspose followed by Conv2D
            self.block = Sequential([
                layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False,
                                    kernel_regularizer=regularizers.l2(0.0001)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001)),
                layers.BatchNormalization()
            ])

            if strides!=1:
                self.shortcut = Sequential([
                    layers.Conv2DTranspose(filters,(1,1),strides=strides,padding='same',use_bias=False,kernel_regularizer=regularizers.L2(1e-4)),
                    layers.BatchNormalization()
                ])
            else:
                self.shortcut = layers.Lambda(lambda x:x)
        else:
            self.block = Sequential([
                layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same',use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same',use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001)),
                layers.BatchNormalization()
            ])
            
            if strides!=1:
                self.shortcut = Sequential([
                    layers.Conv2D(filters,(1,1),strides=strides,padding='same',use_bias=False,kernel_regularizer=regularizers.L2(1e-4)),
                    layers.BatchNormalization()
                ])
            else:
                self.shortcut = layers.Lambda(lambda x:x)

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if self.use_transpose:
            self.shortcut = Sequential([
                layers.Conv2DTranspose(self.filters, kernel_size=(1, 1), strides=self.strides, padding='same', use_bias=False,
                              kernel_regularizer=regularizers.l2(0.0001)),
                layers.BatchNormalization()
            ])
        else:
            if input_channels != self.filters:
                self.shortcut = Sequential([
                    layers.Conv2D(self.filters, kernel_size=(1, 1), strides=self.strides, padding='same', use_bias=False,
                                kernel_regularizer=regularizers.l2(0.0001)),
                    layers.BatchNormalization()
                ])
        super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        _shortcut = self.shortcut(inputs)
        _res_block = self.block(inputs)
        x = layers.Add()([_shortcut,_res_block])
        return layers.ReLU()(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'strides': self.strides
        })
        return config