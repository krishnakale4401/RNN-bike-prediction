# Bike Demand Prediction (RNN)

Simple end-to-end project to predict the next timestep bike demand using a stacked SimpleRNN model.

Project structure:

- src/
  - preprocess.py
  - model.py
  - train.py
  - predict.py
  - app.py (Streamlit)
- data/bike.csv

How to run:

1. Install dependencies (recommended in a venv):

   pip install -r requirements.txt

2. Train model:

   python src/train.py

   This will create `models/rnn_model.h5`.

3. Predict for new data:
   python src/predict.py