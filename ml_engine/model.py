"""
Final Enhanced IDS Model
-----------------------
CNN + Residual Block + BiLSTM + Attention

Includes:
  Projection Layer
  Residual CNN Block
  BatchNormalization
  Dropout
  Bidirectional LSTM
  Custom Additive Attention
  Label Smoothing
  Backward compatibility with existing training scripts
"""

import os
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    LSTM,
    Bidirectional,
    Add,
    Activation,
    Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



# Custom Additive Attention Layer

@tf.keras.utils.register_keras_serializable(package="Custom")
class AdditiveAttention(Layer):
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer="glorot_uniform",
            name="att_weight"
        )
        self.b = self.add_weight(
            shape=(self.attention_dim,),
            initializer="zeros",
            name="att_bias"
        )
        self.v = self.add_weight(
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            name="att_score"
        )
        super().build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        attention_weights = K.softmax(K.dot(score, self.v), axis=1)
        context = attention_weights * inputs
        return K.sum(context, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "attention_dim": self.attention_dim
        })
        return config



# Enhanced CNN + BiLSTM + Attention Model

def build_cnn_bilstm_attention_model(input_shape, num_classes):

   
    # Input Layer
   
    inputs = Input(shape=input_shape, name="network_traffic_input")

   
    # Projection Layer
   
    x = Dense(128, activation="relu", name="projection")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    
    # Residual CNN Block (FIXED)
    
    shortcut = x

    # Main path
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv1D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)

    # Projection shortcut to match channels
    shortcut = Conv1D(64, kernel_size=1, padding="same")(shortcut)

    # Residual merge
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)


    
    # Bidirectional LSTM Encoder
    
    x = Bidirectional(
        LSTM(64, return_sequences=True),
        name="bilstm_encoder"
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    
    # Attention Layer
    
    context_vector = AdditiveAttention(
        attention_dim=64,
        name="attention"
    )(x)

    
    # Decoder / Classifier
    
    x = Dense(64, activation="relu")(context_vector)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="intrusion_output"
    )(x)

   
    # Model Definition
   
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Enhanced_IDS_Model"
    )

    
    # Compile with Label Smoothing
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model



# Backward Compatibility Wrapper

def build_cnn_lstm_attention_model(input_shape, num_classes):
    """
    Wrapper for backward compatibility.
    Existing training scripts can continue using this function.
    """
    return build_cnn_bilstm_attention_model(input_shape, num_classes)