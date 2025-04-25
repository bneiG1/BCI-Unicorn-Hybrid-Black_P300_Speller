import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense, SpatialDropout2D
from tensorflow.keras.models import Model

def build_deeegnet(
    n_channels: int,
    n_samples: int,
    n_classes: int = 2,
    dropout_rate: float = 0.5,
    dilation_rates: list = [1, 2, 4, 8]
) -> Model:
    """
    D-EEGNet: Dilated Convolutional Neural Network for single-trial P300 detection.
    Args:
        n_channels: Number of EEG channels
        n_samples: Number of time samples per epoch
        n_classes: Output classes (default 2)
        dropout_rate: Dropout rate
        dilation_rates: List of dilation rates for temporal convolutions
    Returns:
        Keras Model
    """
    inputs = Input(shape=(n_channels, n_samples, 1))
    x = inputs
    # Block 1: Temporal convolutions with dilation
    for dilation in dilation_rates:
        x = Conv2D(
            filters=8,
            kernel_size=(1, 8),
            dilation_rate=(1, dilation),
            padding='same',
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = SpatialDropout2D(dropout_rate)(x)
    # Block 2: Depthwise spatial convolution
    x = Conv2D(
        filters=16,
        kernel_size=(n_channels, 1),
        groups=n_channels,
        padding='valid',
        use_bias=False
    )(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(32, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
