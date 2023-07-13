import tensorflow as tf
import numpy as np



'''
    The positional embedding layer.
'''

class PositionalEmbedding(tf.keras.layers.Layer):
    
    # TODO: Change up.
    def __positional_encoding(self, depth, length = 2048):
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis = -1
        )
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.encodings = self.__positional_encoding(self.d_model)
    
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, inputs):
        return [
            (self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))) + self.encodings[tf.newaxis, :tf.shape(x)[1], :]
            for x in inputs
        ]



'''
    The Add & Norm layer of the transformer.
'''
class AddNorm(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.add_layer  = tf.keras.layers.Add()
        self.norm_layer = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        return self.norm_layer(self.add_layer(x))



'''
    Applies attention to the encoding.
    Crosses the encoded input with the output.
'''
class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.addnorm = AddNorm()

    def call(self, input, context):
        output = self.multihead_attention(input, key = context, value = context)
        return self.addnorm([input, output])



'''
    The base self-attention layer.
'''
class SelfAttention(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.addnorm = AddNorm()
        
    def call(self, input):
        output = self.multihead_attention(input, key = input, value = input)
        return self.addnorm([input, output])



'''
    The masked version for the self-attention layer.
'''
class MaskedSelfAttention(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.multihead_attention = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.addnorm = AddNorm()
        
    def call(self, input):
        output = self.multihead_attention(input, key = input, value = input, use_causal_mask = True)
        return self.addnorm([input, output])



'''
    A 2-layer feed-forward neural network, processing MHA outputs.
'''
class FeedForwardLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, dropout_model = 0.1, dropout_ff = 0.1):
        super().__init__()
        self.nn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation='relu'),
            tf.keras.layers.Dropout(dropout_ff),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_model)
        ])
        self.addnorm = AddNorm()
    
    def call(self, input):
        y = self.addnorm([input, self.nn(input)])
        return y



'''
    A layer for the encoder.
'''
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attention = SelfAttention(
            num_heads = num_heads,
            key_dim   = d_model,
            dropout   = dropout
        )
        self.ff = FeedForwardLayer(d_model, d_ff)
    
    def call(self, input):
        y = self.ff(self.self_attention(input))
        return y



'''
    The encoder for the transformer.
    Adds the positional embedding at the beginning to preserve order.
    After positional embedding, it applies a series of encoding layers.
'''
class Encoder(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, num_layers, num_heads, vocab_size, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.layers = [
            EncoderLayer(
                d_model   = d_model,
                d_ff      = d_ff,
                num_heads = num_heads,
                dropout   = dropout
            ) for _ in range(num_layers)
        ]
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
    
    def call(self, input):
        context = self.pos_embedding(input)
        context = self.dropout_layer(context)
        for layer in self.layers:
            context = layer(context)
        return context



'''
    A layer for the decoder.
'''
class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.masked_attention = MaskedSelfAttention(
            num_heads = num_heads,
            key_dim   = d_model,
            dropout   = dropout
        )
        self.mh_attention  = MultiHeadAttention(
            num_heads = num_heads,
            key_dim   = d_model,
            dropout   = dropout
        )
        self.ff = FeedForwardLayer(d_model, d_ff)
    
    def call(self, input, context):
        masked_input = self.masked_attention(input)
        masked_input = self.mh_attention(input = masked_input, context = context)
        output = self.ff(masked_input)
        return output



'''
    The decoder for the transformer.
    First, it decodes the series of encoding layers from the Encoder.
    Then, it removes the positional embedding to get the ordered output.
'''
class Decoder(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, num_layers, num_heads, vocab_size, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model)
        self.layers = [
            DecoderLayer(
                d_model   = d_model,
                d_ff      = d_ff,
                num_heads = num_heads,
                dropout   = dropout
            ) for _ in range(num_layers)
        ]
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
    
    def call(self, input, context):
        output = self.pos_embedding(input)
        output = self.dropout_layer(output)
        for layer in self.layers:
            output = layer(output, context)
        return output



'''
    The transformer model.
'''
class Transformer(tf.keras.Model):

    def __init__(self, d_model, d_ff, num_layers, num_heads, input_vocab_size, output_vocab_size, dropout = 0.1):
        super().__init__()
        self.encoder = Encoder(
            d_model = d_model,
            d_ff = d_ff,
            num_layers = num_layers,
            num_heads = num_heads,
            vocab_size = input_vocab_size,
            dropout = dropout
        )
        self.decoder = Decoder(
            d_model = d_model,
            d_ff = d_ff,
            num_layers = num_layers,
            num_heads = num_heads,
            vocab_size = output_vocab_size,
            dropout = dropout
        )
        self.final_layer = tf.keras.layers.Dense(output_vocab_size)
    
    def call(self, input):
        output = self.encoder(input)
        output = self.decoder(input, output)
        output = self.final_layer(output)
        return list(output.numpy())