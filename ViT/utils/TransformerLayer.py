# -------------------------------------------------------------------
# Transformer MLP (Feed-Forward Network)
# -------------------------------------------------------------------
import tensorflow as tf
import keras
from keras import layers

class TransformerMLP(layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1, **kwargs):
        super(TransformerMLP, self).__init__(**kwargs)
        self.dense1 = layers.Dense(d_ff, activation=keras.activations.gelu)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(d_model)
        self.dropout2 = layers.Dropout(dropout_rate)
            
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x

# -------------------------------------------------------------------
# Single Transformer Encoder Block
# -------------------------------------------------------------------
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = TransformerMLP(d_model=embed_dim, d_ff=mlp_dim, dropout_rate=dropout)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Self-Attention block with residual connection.
        x_norm = self.norm1(inputs)
        attn_output = self.attn(x_norm, x_norm)
        x = inputs + attn_output
        
        # Feed-Forward block with residual connection.
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm, training=training)
        return x + mlp_output