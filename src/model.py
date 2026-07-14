"""오토인코더 모델 정의 (TensorFlow/Keras, Dense AE)."""
from __future__ import annotations

from typing import Any, Dict

from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_autoencoder(n_features: int, cfg: Dict[str, Any]) -> keras.Model:
    """대칭형 Dense Autoencoder를 구성한다.

    구조: input → hidden_layers → bottleneck → reversed(hidden_layers) → output(linear)
    과용량 방지를 위해 병목 차원을 제한하고 L2/Dropout 옵션을 둔다.
    """
    m_cfg = cfg["model"]
    hidden = list(m_cfg.get("hidden_layers", [32, 16]))
    bottleneck = int(m_cfg.get("bottleneck", 8))
    activation = m_cfg.get("activation", "relu")
    dropout = float(m_cfg.get("dropout", 0.0))
    l2 = float(m_cfg.get("l2", 0.0))
    reg = regularizers.l2(l2) if l2 > 0 else None

    inputs = keras.Input(shape=(n_features,), name="input")
    x = inputs

    # Encoder
    for i, units in enumerate(hidden):
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=reg, name=f"enc_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"enc_drop_{i}")(x)

    # Bottleneck
    x = layers.Dense(bottleneck, activation=activation,
                     kernel_regularizer=reg, name="bottleneck")(x)

    # Decoder (대칭)
    for i, units in enumerate(reversed(hidden)):
        x = layers.Dense(units, activation=activation,
                         kernel_regularizer=reg, name=f"dec_{i}")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dec_drop_{i}")(x)

    outputs = layers.Dense(n_features, activation="linear", name="output")(x)

    model = keras.Model(inputs, outputs, name="phm_autoencoder")
    return model


def build_lstm_autoencoder(n_features: int, window: int, cfg: Dict[str, Any]) -> keras.Model:
    """시계열 윈도우 오토인코더(LSTM-AE).

    입력: (window, n_features) 시퀀스. 인코더 LSTM으로 압축 후 RepeatVector로
    시퀀스를 재생성하고 디코더 LSTM + TimeDistributed Dense로 복원한다.
    시간 의존성이 중요한 데이터에서 행 단위 Dense AE보다 정상 패턴을 잘 학습한다.
    """
    m_cfg = cfg["model"]
    latent = int(m_cfg.get("bottleneck", 8))
    hidden = int(list(m_cfg.get("hidden_layers", [32]))[0])

    inputs = keras.Input(shape=(window, n_features), name="seq_input")
    x = layers.LSTM(hidden, activation="tanh", return_sequences=True, name="enc_lstm1")(inputs)
    x = layers.LSTM(latent, activation="tanh", return_sequences=False, name="enc_lstm2")(x)
    x = layers.RepeatVector(window, name="repeat")(x)
    x = layers.LSTM(latent, activation="tanh", return_sequences=True, name="dec_lstm1")(x)
    x = layers.LSTM(hidden, activation="tanh", return_sequences=True, name="dec_lstm2")(x)
    outputs = layers.TimeDistributed(layers.Dense(n_features), name="seq_output")(x)

    return keras.Model(inputs, outputs, name="phm_lstm_autoencoder")
