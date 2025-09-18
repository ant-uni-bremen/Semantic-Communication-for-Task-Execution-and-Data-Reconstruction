# -------------------------------------------------------------------
# Vision Transformer (ViT) Encoder
# -------------------------------------------------------------------
import tensorflow as tf
import keras
from keras import layers,Model
from ViT.utils.PatchEmbedded import PatchEmbedding
from ViT.utils.TransformerLayer import TransformerBlock

class ViTEncoder(Model):
    def __init__(self, num_patches, patch_size, embed_dim, num_layers, num_heads, mlp_dim,input_shape,
                 enc_out_dec_inp,split_image_into,dropout_rate=0.1,**kwargs):
        """
        Args:
            num_patches: Total number of patches.
            patch_size: Size of one square patch.
            embed_dim: Dimension of patch embeddings.
            num_layers: Number of transformer blocks.
            num_heads: Number of attention heads.
            mlp_dim: Hidden dimension of the MLP in transformer blocks.
            enc_out_dec_inp: Final output dimension.
            mlp_head_units: List of units for the MLP head.
            dropout_rate: Dropout rate.
            input_shape: Shape of input images.
        """
        super(ViTEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.enc_out_dec_inp = enc_out_dec_inp
        self.input_shape = input_shape
        self.split_image_into = split_image_into
        self.dropout_rate = dropout_rate
        self.vit_encoder = self.build_vit_encoder()

    def build_vit_encoder(self):
        input = keras.Input(shape= self.input_shape)  # (H, W, C)
        x = PatchEmbedding(
            image_size=self.input_shape[0],
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            channels=self.input_shape[-1]
        )(input)

        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embed = layers.Embedding(input_dim=self.num_patches, output_dim=self.embed_dim)(positions)
        x = x + pos_embed

        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim, dropout=self.dropout_rate)(x)

        x = layers.Flatten()(x)
        
        outputs = layers.Dense(self.enc_out_dec_inp // self.split_image_into)(x)

        return keras.Model(inputs=input, outputs=outputs, name="ViTEncoder")

    def build(self, input_shape):
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected input shape (batch_size,height, width, channels), but got {input_shape}")
        super().build(input_shape)
        
    def call(self, inputs, training=False):
        # Extract and project patches.
        return self.vit_encoder(inputs)

    def get_config(self):
        config = super(ViTEncoder, self).get_config()
        config.update({
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'enc_out_dec_inp': self.enc_out_dec_inp,
            'dropout_rate': self.dropout_rate,
            'split_image_into': self.split_image_into,
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_patches=config['num_patches'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            mlp_dim=config['mlp_dim'],
            split_image_into=config['split_image_into'],
            enc_out_dec_inp=config['enc_out_dec_inp'],
            dropout_rate=config['dropout_rate'],
            input_shape=config['input_shape'],
            name=config['name']
        )