"""
=============================================================================
model_tcn.py  —  Temporal Convolutional Network (TCN)
=============================================================================
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Dropout, Add,
    GlobalAveragePooling1D, ZeroPadding1D, Activation
)
from tensorflow.keras.optimizers import Adam


def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """Single TCN residual block with dilated causal convolutions."""
    padding = (kernel_size - 1) * dilation_rate

    # First causal conv
    x_pad = ZeroPadding1D((padding, 0))(x)
    conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding="valid", activation="relu")(x_pad)
    conv1 = Dropout(dropout_rate)(conv1)

    # Second causal conv
    x_pad2 = ZeroPadding1D((padding, 0))(conv1)
    conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                   padding="valid", activation="relu")(x_pad2)
    conv2 = Dropout(dropout_rate)(conv2)

    # Residual connection — 1×1 conv if dimensions differ
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding="same")(x)

    return Activation("relu")(Add()([x, conv2]))


def build_tcn(seq_len, n_features, filters=64, kernel_size=3,
              num_blocks=4, dropout_rate=0.2, lr=1e-3):
    """
    TCN with exponentially increasing dilation rates:
      Block 0 → dilation 1
      Block 1 → dilation 2
      Block 2 → dilation 4
      Block 3 → dilation 8
    Receptive field = kernel_size * (2^num_blocks - 1)
    """
    inputs = Input(shape=(seq_len, n_features))
    x = inputs

    for i in range(num_blocks):
        dilation = 2 ** i
        x = tcn_residual_block(x, filters, kernel_size, dilation, dropout_rate)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs, name="TCN_Model")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model