"""
fix_lstm_and_convert.py
Reconstruit le LSTM avec unroll=True (élimine les ops
dynamiques TensorListReserve) puis reconvertit en .tflite
SANS avoir besoin du Flex delegate.
"""

import os
import json
import tensorflow as tf

MODEL_DIR = "models"
H5_PATH = os.path.join(MODEL_DIR, "encoder_lstm_only.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "encoder_lstm_toniot.tflite")

print("=" * 60)
print("CORRECTIF LSTM — unroll=True")
print("=" * 60)

# 1. Charger le modèle original
model = tf.keras.models.load_model(H5_PATH, compile=False)
print(f"\nModèle chargé : {H5_PATH}")
model.summary()

# 2. Récupérer la config et forcer unroll=True sur toutes les
#    couches LSTM (récursivement, au cas où il y a des sous-modèles)
def forcer_unroll(config):
    if isinstance(config, dict):
        if config.get("class_name") in ("LSTM", "GRU", "SimpleRNN"):
            config["config"]["unroll"] = True
        for v in config.values():
            forcer_unroll(v)
    elif isinstance(config, list):
        for item in config:
            forcer_unroll(item)

config = model.get_config()
forcer_unroll(config)

# 3. Reconstruire le modèle depuis la config modifiée
model_unrolled = tf.keras.Model.from_config(config)
model_unrolled.set_weights(model.get_weights())
print("\n✅ Modèle reconstruit avec unroll=True et poids copiés")

# 4. Figer le batch_size à 1 (nécessaire pour TFLite edge)
input_shape = model.input_shape  # ex (None, 1, 16)
fixed_input = tf.keras.Input(batch_shape=(1,) + tuple(input_shape[1:]))
fixed_output = model_unrolled(fixed_input)
fixed_model = tf.keras.Model(fixed_input, fixed_output)

# 5. Conversion TFLite SANS Flex (doit passer nativement maintenant)
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

taille_ko = len(tflite_model) / 1024
print(f"\n✅ Sauvegardé : {TFLITE_PATH} ({taille_ko:.1f} KB)")
print("   (sans Flex ops — compatible interpréteur standard)")