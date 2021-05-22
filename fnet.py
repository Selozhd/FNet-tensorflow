""" Implements tf.keras layers for FNet transformer architecture.

For the original paper discussing FNET see:
    Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon, S. (2021).
    FNet: Mixing Tokens with Fourier Transforms. arXiv preprint arXiv:2105.03824.
"""
import tensorflow as tf


class Forward(tf.keras.layers.Layer):

    def __init__(self, units, dropout_rate, **kwargs):
        super(Forward, self).__init__(**kwargs)
        self.units = units
        self.rate = dropout_rate
        self.supports_masking = True
        self.dense1 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(self.rate)
        
    def call(self, inputs, training=False):
        X = self.dense1(inputs)
        X = self.dropout(X, training=training)
        X = self.dense2(X)
        X = self.dropout(X, training=training)
        return X

    def get_config(self):
        config = super(Forward, self).get_config()
        config.update({"units": self.units, "dropout_rate": self.rate})
        return config


class FNetBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_dim):
        super(FNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.supports_masking = True
        self.norm_fourier = tf.keras.layers.BatchNormalization()
        self.norm_ffn = tf.keras.layers.BatchNormalization()
        self.ffn = Forward(self.hidden_dim, 0.1)

    def get_config(self):
        config = super(FNetBlock, self).get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

    def call(self, inputs, training=False):
        X_complex = tf.cast(inputs, tf.complex64)
        X_fft = tf.math.real(tf.signal.fft2d(X_complex))
        X_norm1 = self.norm_fourier(X_fft + inputs, training=training)
        X_dense = self.ffn(X_norm1, training=training)
        X_norm2 = self.norm_ffn(X_dense + X_norm1, training=training)
        return X_norm2