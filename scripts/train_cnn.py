"""
Script d'entraînement CNN 1D (classifieur supervisé → encoder)
================================================================
Entraîne un CNN 1D en classification binaire (normal/attaque)
sur les données NETTOYÉES de la Semaine 3 (sans fuite avec
eval_test_full.csv), puis extrait la sous-partie "encoder"
(jusqu'à dense_embed) utilisée pour la fusion en Semaine 4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split

print("=" * 70)
print("CNN 1D - ENTRAÎNEMENT CLASSIFIEUR + EXTRACTION ENCODER")
print("=" * 70)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ====================== CHARGEMENT DES DONNÉES (Semaine 3, nettoyées) ======================
print("\nChargement des données nettoyées et normalisées (Semaine 3)...")
X_train_scaled = np.load(f"{DATA_DIR}/X_train_scaled.npy").astype(np.float32)
X_eval_scaled  = np.load(f"{DATA_DIR}/X_eval_scaled.npy").astype(np.float32)
y_train_full   = np.load(f"{DATA_DIR}/y_train.npy")
y_eval         = np.load(f"{DATA_DIR}/y_eval.npy")

print(f"Train shape : {X_train_scaled.shape} | Eval shape : {X_eval_scaled.shape}")

# Split train/validation pour l'entraînement supervisé
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)

# Reshape pour CNN 1D : (n_samples, n_features=16, channels=1)
X_tr_cnn   = X_tr.reshape(-1, X_tr.shape[1], 1)
X_val_cnn  = X_val.reshape(-1, X_val.shape[1], 1)
X_eval_cnn = X_eval_scaled.reshape(-1, X_eval_scaled.shape[1], 1)

INPUT_DIM = X_train_scaled.shape[1]

# ====================== ARCHITECTURE (identique à celle inspectée) ======================
print(f"\nConstruction du CNN 1D classifieur (input=({INPUT_DIM},1))")

cnn_input = keras.Input(shape=(INPUT_DIM, 1), name="cnn_input")
x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu', name="conv1")(cnn_input)
x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)
x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', name="conv2")(x)
x = layers.GlobalMaxPooling1D(name="global_pool")(x)
x = layers.Dropout(0.3, name="dropout")(x)
embed = layers.Dense(16, activation='relu', name="dense_embed")(x)
output = layers.Dense(1, activation='sigmoid', name="output")(embed)

cnn_classifier = keras.Model(cnn_input, output, name="cnn_classifier")
cnn_classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
cnn_classifier.summary()

# ====================== ENTRAÎNEMENT ======================
print("\nDébut de l'entraînement...")

n_normal  = int((y_tr == 0).sum())
n_attaque = int((y_tr == 1).sum())
class_weight = {
    0: (n_normal + n_attaque) / (2 * n_normal),
    1: (n_normal + n_attaque) / (2 * n_attaque)
}
print(f"class_weight : {class_weight}")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)

history = cnn_classifier.fit(
    X_tr_cnn, y_tr,
    epochs=30,
    batch_size=256,
    validation_data=(X_val_cnn, y_val),
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# ====================== ÉVALUATION RAPIDE ======================
print("\nÉvaluation sur eval_test_full.csv (nettoyé) :")
loss, acc = cnn_classifier.evaluate(X_eval_cnn, y_eval, verbose=0)
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  Loss     : {loss:.4f}")

# ====================== SAUVEGARDE ======================
cnn_classifier.save(f"{MODEL_DIR}/cnn_classifier_full.h5")
print(f"\n✅ Modèle complet sauvegardé : {MODEL_DIR}/cnn_classifier_full.h5")

# Extraction de l'encoder seul (jusqu'à dense_embed, sans la couche output)
encoder_cnn = keras.Model(
    inputs=cnn_classifier.input,
    outputs=cnn_classifier.get_layer("dense_embed").output,
    name="cnn_encoder"
)
encoder_cnn.save(f"{MODEL_DIR}/encoder_cnn_only.h5")
print(f"✅ Encoder seul sauvegardé  : {MODEL_DIR}/encoder_cnn_only.h5")

# ====================== GRAPHIQUE ======================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CNN 1D Classifieur — Entraînement (sans fuite de données)', fontsize=14)

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Courbe de perte')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Crossentropy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Courbe d\'accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/cnn_results.png", dpi=150)
print(f"✅ Graphique sauvegardé : {RESULTS_DIR}/cnn_results.png")

print("\n✅ CNN 1D terminé avec succès (sans fuite de données) !")