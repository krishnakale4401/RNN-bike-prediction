# src/predict.py

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from preprocess import load_data, FEATURES

def predict():

    model = load_model(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\model.keras",compile=False)
    x_scaler = joblib.load(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\x_scaler.pkl")
    y_scaler = joblib.load(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\y_scaler.pkl")

    df = load_data()

    # last 7 days
    last_data = df[FEATURES].tail(7)

    X = x_scaler.transform(last_data)
    X = X.reshape(1, X.shape[0], X.shape[1])  ## (1,7,11)

    pred = model.predict(X)

    pred = y_scaler.inverse_transform(pred)

    print("Next day prediction:", pred[0][0])

def predict_from_array(arr):
    import joblib
    import numpy as np
    from tensorflow.keras.models import load_model

    from preprocess import FEATURES

    # load model + scalers
    model = load_model(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\model.h5", compile=False)
    x_scaler = joblib.load(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\x_scaler.pkl")
    y_scaler = joblib.load(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models/y_scaler.pkl")

    # reshape properly
    arr = arr.reshape(-1, arr.shape[-1])

    # scale input
    arr = x_scaler.transform(arr)

    # reshape back to RNN format
    arr = arr.reshape(1, 7, len(FEATURES))

    # predict
    pred = model.predict(arr)

    # inverse scale
    pred = y_scaler.inverse_transform(pred)

    return float(pred[0][0])
    
if __name__ == "__main__":
    predict()