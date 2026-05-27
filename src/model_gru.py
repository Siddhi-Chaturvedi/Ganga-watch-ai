"""
=============================================================================
model_gru.py  —  GRU Model
=============================================================================
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def build_gru(seq_len, n_features, units1=128, units2=64, dropout=0.2, lr=1e-3):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        GRU(units1, return_sequences=True),
        Dropout(dropout),
        GRU(units2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="GRU_Model")
    model.compile(optimizer=Adam(lr), loss="mse")
    return model