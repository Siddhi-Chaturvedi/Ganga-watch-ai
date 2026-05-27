"""
=============================================================================
model_transformer.py  —  Transformer Encoder Model
=============================================================================
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam


def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate=0.1):
    """
    Single Transformer encoder block:
      Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
    """
    # Multi-head self-attention
    key_dim = max(1, x.shape[-1] // num_heads)
    attn    = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn    = Dropout(dropout_rate)(attn)
    out1    = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

    # Position-wise Feed-Forward Network
    ff      = Dense(ff_dim, activation="relu")(out1)
    ff      = Dropout(dropout_rate)(ff)
    ff      = Dense(x.shape[-1])(ff)
    out2    = LayerNormalization(epsilon=1e-6)(Add()([out1, ff]))

    return out2


def build_transformer(seq_len, n_features, d_model=64, num_heads=4,
                       ff_dim=128, num_blocks=2, dropout_rate=0.1, lr=1e-3):
    """
    Transformer encoder for time-series forecasting.

    Parameters
    ----------
    seq_len    : input sequence length
    n_features : number of input features
    d_model    : internal model dimension (projected from n_features)
    num_heads  : number of attention heads
    ff_dim     : feed-forward hidden dimension
    num_blocks : number of encoder blocks stacked
    """
    inputs = Input(shape=(seq_len, n_features))

    # Project input to d_model dimensions
    x = Dense(d_model)(inputs)

    # Stack encoder blocks
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, num_heads=num_heads,
                                      ff_dim=ff_dim, dropout_rate=dropout_rate)

    # Aggregate over time dimension
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs, name="Transformer_Model")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model