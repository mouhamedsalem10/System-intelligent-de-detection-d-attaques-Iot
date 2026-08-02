"""
Script d'entraînement AutoEncoder (CORRIGÉ)
=============================================
CORRECTIF : utilise désormais data/X_train_scaled.npy et
data/X_eval_scaled.npy (Semaine 3, nettoyés de toute fuite
de données avec eval_test_full.csv), déjà normalisés par le
scaler final. Plus de double-transform, plus d'incohérence
avec les données brutes de la Semaine 2.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from sklearn.model_selection import train_test_split

print("=" * 70)
print("AUTOENCODER - EXTRACTION DE CARACTÉRISTIQUES (S3, corrigé)")
print("=" * 70)

# ====================== CONFIG ======================
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Charger les données de la Semaine 3 (déjà nettoyées + normalisées)
print("\nChargement des données nettoyées et normalisées (Semaine 3)...")
X_train_scaled = np.load(f"{DATA_DIR}/X_train_scaled.npy").astype(np.float32)
X_test_scaled  = np.load(f"{DATA_DIR}/X_eval_scaled.npy").astype(np.float32)
y_train = pd.Series(np.load(f"{DATA_DIR}/y_train.npy"))
y_test  = pd.Series(np.load(f"{DATA_DIR}/y_eval.npy"))

print(f"Train shape : {X_train_scaled.shape} | Eval shape : {X_test_scaled.shape}")

# Isoler seulement le trafic normal pour l'entraînement
X_normal = X_train_scaled[y_train.values == 0]
print(f"Trafic normal pour entraînement : {len(X_normal)} flux")

# Split train/validation (sur le trafic normal uniquement)
X_train_ae, X_val_ae = train_test_split(X_normal, test_size=0.2, random_state=42)

# ====================== ARCHITECTURE ======================
INPUT_DIM = X_train_scaled.shape[1]
LATENT_DIM = 8   # compression (8 au lieu de 16)

print(f"\nConstruction de l'AutoEncoder (input={INPUT_DIM} → latent={LATENT_DIM})")

encoder_input = keras.Input(shape=(INPUT_DIM,))
x = layers.Dense(64, activation='relu')(encoder_input)
x = layers.Dense(32, activation='relu')(x)
latent = layers.Dense(LATENT_DIM, activation='relu', name='latent_space')(x)
encoder = keras.Model(encoder_input, latent, name="encoder")

decoder_input = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(32, activation='relu')(decoder_input)
x = layers.Dense(64, activation='relu')(x)
decoder_output = layers.Dense(INPUT_DIM, activation='linear')(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

# AutoEncoder complet
ae_input = keras.Input(shape=(INPUT_DIM,))
encoded = encoder(ae_input)
decoded = decoder(encoded)
autoencoder = keras.Model(ae_input, decoded, name="autoencoder")

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# ====================== ENTRAÎNEMENT ======================
print("\nDébut de l'entraînement...")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

history = autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=50,
    batch_size=256,
    validation_data=(X_val_ae, X_val_ae),
    callbacks=[early_stop],
    verbose=1
)

# ====================== SAUVEGARDE ======================
encoder.save(f"{MODEL_DIR}/encoder_ae_toniot.h5")
autoencoder.save(f"{MODEL_DIR}/autoencoder_full.h5")
print(f"\nModèles sauvegardés dans {MODEL_DIR}/")

print("\n✅ AutoEncoder terminé avec succès !")

print("\nÉvaluation de l'AutoEncoder...")

# Reconstructions
reconstructions_train = autoencoder.predict(X_train_scaled, verbose=0)
mse_train = np.mean(np.power(X_train_scaled - reconstructions_train, 2), axis=1)

reconstructions_test = autoencoder.predict(X_test_scaled, verbose=0)
mse_test = np.mean(np.power(X_test_scaled - reconstructions_test, 2), axis=1)

# Erreurs par classe (sur le train)
mse_normal = mse_train[y_train.values == 0]
mse_attaque = mse_train[y_train.values == 1]

print(f"\nErreur de reconstruction (MSE) :")
print(f"  Normal   : moyenne = {mse_normal.mean():.5f}")
print(f"  Attaque  : moyenne = {mse_attaque.mean():.5f}")

# Graphiques
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Résultats AutoEncoder - Semaine 3 (corrigé, sans fuite)', fontsize=14)

# Loss curve
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Courbe de perte pendant l\'entraînement')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogramme des erreurs
axes[1].hist(mse_normal[:5000], bins=50, alpha=0.7, color='green', label='Normal')
axes[1].hist(mse_attaque[:5000], bins=50, alpha=0.7, color='red', label='Attaque')
axes[1].set_title('Distribution des erreurs de reconstruction')
axes[1].set_xlabel('MSE')
axes[1].set_ylabel('Fréquence')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/autoencoder_results.png", dpi=150)
print(f"✅ Graphique sauvegardé : {RESULTS_DIR}/autoencoder_results.png")

print("\n✅ AutoEncoder terminé avec succès (sans fuite de données) !")