# src/train.py

from preprocess import preprocess
from model import build_model
import numpy as np
import joblib

def main():
    X_train, X_test, y_train, y_test = preprocess()

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    model = build_model((X_train.shape[1], X_train.shape[2]))

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32
    )

    # save model
    model.save(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\model.keras")

    # evaluation
    pred = model.predict(X_test)

    y_scaler = joblib.load(r"D:\Jen AI\Deep Learning\RNN-bike-prediction\Models\y_scaler.pkl")
    pred = y_scaler.inverse_transform(pred)
    y_test = y_scaler.inverse_transform(y_test)

    rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    print("RMSE:", rmse)


if __name__ == "__main__":
    main()