from keras import layers
import tensorflow as tf
import numpy as np

class AWGNLayer(layers.Layer):
    def __init__(self, snr, **kwargs):
        super(AWGNLayer, self).__init__(**kwargs)
        self.snr = tf.Variable(initial_value=snr, trainable=False, dtype=tf.float32)

    def call(self, input):
        def awgn_eachSplit(signal):
            snr_lin = tf.pow(10.0, self.snr / 10.0)
            signal_power = tf.reduce_mean(tf.square(signal), axis=-1, keepdims=True)
            noise_variance = signal_power / snr_lin
            noise = tf.random.normal(tf.shape(signal), mean=0.0, stddev=tf.sqrt(noise_variance))
            y = signal + noise
            return y
        
        return awgn_eachSplit(input)

    def update_snr(self, new_snr):
        """Update the SNR value dynamically."""
        self.snr.assign(new_snr)

    def get_config(self):
        """Save the current SNR value for model serialization."""
        config = super(AWGNLayer, self).get_config()
        config.update({
            "snr": float(self.snr.numpy())  # Store the current SNR value as a float
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Restore the layer from the saved configuration."""
        snr = config.pop("snr")
        layer = cls(snr=snr, **config)
        return layer

class RayleighFadingChannel(layers.Layer):
    def __init__(self, snr, *, activity_regularizer=None, trainable=True, dtype=None, autocast=True, name=None, **kwargs):
        super(RayleighFadingChannel,self).__init__(activity_regularizer=activity_regularizer, trainable=trainable, dtype=dtype, autocast=autocast, name=name, **kwargs)
        self.snr = tf.Variable(initial_value=snr, trainable=False, dtype=tf.float32)
    
    def call(self, input):
        def Rayleigh_eachSplit(signal):
            snr_lin = tf.pow(10.0, self.snr / 10.0)
            signal_power = tf.reduce_mean(tf.square(signal), axis=-1, keepdims=True)
            noise_variance = signal_power / snr_lin
            
            # Generate Rayleigh fading coefficients using TensorFlow
            real = tf.random.normal(tf.shape(signal), mean=0.0, stddev=1.0)
            imag = tf.random.normal(tf.shape(signal), mean=0.0, stddev=1.0)
            h = tf.complex(real, imag) / tf.complex(tf.sqrt(2.0), 0.0)
            
            # Convert signal to complex if necessary
            signal_complex = tf.complex(signal,signal)
            
            # Generate noise
            noise_real = tf.random.normal(tf.shape(signal), mean=0.0, stddev=tf.sqrt(noise_variance))
            noise_imag = tf.random.normal(tf.shape(signal), mean=0.0, stddev=tf.sqrt(noise_variance))
            noise = tf.complex(noise_real, noise_imag)
            
            y = (h * signal_complex) + noise
            return tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        
        return Rayleigh_eachSplit(input)

    def update_snr(self, new_snr):
        """Update the SNR value dynamically."""
        self.snr.assign(new_snr)

    def get_config(self):
        """Save the current SNR value for model serialization."""
        config = super(RayleighFadingChannel, self).get_config()
        config.update({
            "snr": float(self.snr.numpy())  # Store the current SNR value as a float
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Restore the layer from the saved configuration."""
        snr = config.pop("snr")
        layer = cls(snr=snr, **config)
        return layer

class IdentityChannel(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

    def update_snr(self, snr):
        pass  # No effect since it's identity

    def get_config(self):
        return {}
