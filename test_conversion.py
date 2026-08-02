"""
test_conversion.py
Vérifie que les nouveaux .tflite donnent les mêmes
embeddings que les .h5 sources, sur quelques exemples réels.
"""

import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import json

MODEL_DIR = "models"

# Charger le scaler et un échantillon réel
scaler = joblib.load(f"{MODEL_DIR}/scaler_toniot.joblib")
df = pd.read_csv("data/eval_test_full.csv").sample(5, random_state=1)

features = json.loads(open(f"{MODEL_DIR}/feature_names_toniot.json").read())
X = df[features].fillna(0.0).values.astype(np.float32)
X_scaled = scaler.transform(X).astype(np.float32)

def test_encoder(h5_path, tflite_path, X_input, label):
    print(f"\n--- {label} ---")
    model = tf.keras.models.load_model(h5_path, compile=False)
    z_h5 = model.predict(X_input, verbose=0)

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    z_tflite = []
    for i in range(len(X_input)):
        x = X_input[i:i+1].reshape(inp[0]['shape']).astype(np.float32)
        interp.set_tensor(inp[0]['index'], x)
        interp.invoke()
        z_tflite.append(interp.get_tensor(out[0]['index']).flatten())
    z_tflite = np.array(z_tflite)

    diff = np.abs(z_h5 - z_tflite).max()
    print(f"  Écart max h5 vs tflite : {diff:.6f}")
    print("  ✅ OK" if diff < 1e-3 else "  ❌ PROBLÈME — écart trop grand")

test_encoder(
    f"{MODEL_DIR}/encoder_ae_only.h5",
    f"{MODEL_DIR}/encoder_ae_toniot.tflite",
    X_scaled, "AutoEncoder"
)
test_encoder(
    f"{MODEL_DIR}/encoder_lstm_only.h5",
    f"{MODEL_DIR}/encoder_lstm_toniot.tflite",
    X_scaled.reshape(-1, 1, 16), "LSTM"
)
test_encoder(
    f"{MODEL_DIR}/encoder_cnn_only.h5",
    f"{MODEL_DIR}/encoder_cnn_toniot.tflite",
    X_scaled.reshape(-1, 16, 1), "CNN"
)