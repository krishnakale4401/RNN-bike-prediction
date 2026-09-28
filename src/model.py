# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.losses import MeanSquaredError

def build_model(input_shape):
    model = Sequential()

    model.add(SimpleRNN(64, return_sequences=True, input_shape=input_shape)) ## return_sequences means pass the sequence to next RNN
    model.add(SimpleRNN(32))
    model.add(Dense(1))  # We want to predict a single value of "Bike count"

    

    model.compile(optimizer="adam", loss=MeanSquaredError())
    return model