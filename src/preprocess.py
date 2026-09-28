import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

FEATURES = [
    "season","yr","mnth","holiday","weekday",
    "workingday","weathersit","temp","atemp","hum","windspeed"
]

def load_data(path=r"D:\Jen AI\Deep Learning\RNN-bike-prediction\data\bike.csv"):
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.sort_values("dteday")
    return df


def create_sequences(X, y, window=7):
    Xs, ys = [], []  ## Sample store to store the sequence
    for i in range(len(X) - window):  ## Sliding window
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)


def preprocess():
    df = load_data()

    # ✅ split first (NO leakage)
    split = int(len(df) * 0.8)
    train_df = df[:split]
    test_df = df[split:]

    # ✅ scale X
    x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(train_df[FEATURES])
    X_test = x_scaler.transform(test_df[FEATURES])

    # ✅ scale y
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(train_df[["cnt"]])
    y_test = y_scaler.transform(test_df[["cnt"]])

    # ✅ create sequences AFTER split
    X_train, y_train = create_sequences(X_train, y_train)
    X_test, y_test = create_sequences(X_test, y_test)

    # ✅ save scalers
    joblib.dump(x_scaler, r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\x_scaler.pkl")
    joblib.dump(y_scaler, r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\y_scaler.pkl")

    return X_train, X_test, y_train, y_test