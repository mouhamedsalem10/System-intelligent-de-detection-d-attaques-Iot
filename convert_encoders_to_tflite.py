"""
convert_encoders_to_tflite.py
Reconvertit les 3 nouveaux encodeurs .h5 vers .tflite
et écrase les anciens fichiers dans models/
"""

import os
import shutil
import tensorflow as tf

MODEL_DIR = "models"
BACKUP_DIR = "models/backup_old_tflite"

os.makedirs(BACKUP_DIR, exist_ok=True)

# ============================================================
# ÉTAPE 0 — Backup des anciens .tflite (sécurité)
# ============================================================
anciens = [
    "encoder_ae_toniot.tflite",
    "encoder_lstm_toniot.tflite",
    "encoder_cnn_toniot.tflite",
]

print("=" * 60)
print("BACKUP DES ANCIENS FICHIERS TFLITE")
print("=" * 60)
for nom in anciens:
    src = os.path.join(MODEL_DIR, nom)
    if os.path.exists(src):
        dst = os.path.join(BACKUP_DIR, nom)
        shutil.copy2(src, dst)
        print(f"  ✅ Backup : {nom} → {BACKUP_DIR}/")
    else:
        print(f"  ⚠️  Introuvable (pas grave) : {nom}")


# ============================================================
# Fonction générique de conversion
# ============================================================
def convertir_en_tflite(h5_path, tflite_path, input_shape, is_lstm=False):
    """
    h5_path      : chemin du modèle .h5 source
    tflite_path  : chemin du .tflite de sortie
    input_shape  : shape figée pour batch=1, ex (1, 16)
    is_lstm      : True pour activer le correctif LSTM
                   (nécessaire car TFLite ne supporte pas
                   nativement certaines ops dynamiques des LSTM)
    """
    print(f"\n--- Conversion : {h5_path} ---")
    model = tf.keras.models.load_model(h5_path, compile=False)

    if tuple(model.input_shape[1:]) != tuple(input_shape[1:]):
        raise ValueError(
            f"Shape du modèle {model.input_shape} "
            f"différente de celle attendue {input_shape} "
            f"— vérifier l'architecture avant de continuer."
        )

    # Reconstruire le modèle avec un batch_size FIXE (1)
    # → nécessaire pour figer les shapes internes (surtout LSTM)
    fixed_input = tf.keras.Input(
        batch_shape=(1,) + tuple(input_shape[1:])
    )
    fixed_output = model(fixed_input)
    fixed_model = tf.keras.Model(fixed_input, fixed_output)

    converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)

    if is_lstm:
        # Correctif : autoriser les ops TF natives non convertibles
        # nativement (nécessaire pour LSTM avec TensorListReserve)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    taille_ko = len(tflite_model) / 1024
    print(f"  ✅ Sauvegardé : {tflite_path} ({taille_ko:.1f} KB)")


# ============================================================
# ÉTAPE 1 — Conversion des 3 encodeurs
# ============================================================
print("\n" + "=" * 60)
print("CONVERSION DES NOUVEAUX ENCODEURS")
print("=" * 60)

# AutoEncoder : entrée (batch, 16)
convertir_en_tflite(
    h5_path=os.path.join(MODEL_DIR, "encoder_ae_only.h5"),
    tflite_path=os.path.join(MODEL_DIR, "encoder_ae_toniot.tflite"),
    input_shape=(1, 16),
    is_lstm=False
)

# LSTM : entrée (batch, 1, 16) — avec correctif
convertir_en_tflite(
    h5_path=os.path.join(MODEL_DIR, "encoder_lstm_only.h5"),
    tflite_path=os.path.join(MODEL_DIR, "encoder_lstm_toniot.tflite"),
    input_shape=(1, 1, 16),
    is_lstm=True
)

# CNN : entrée (batch, 16, 1)
convertir_en_tflite(
    h5_path=os.path.join(MODEL_DIR, "encoder_cnn_only.h5"),
    tflite_path=os.path.join(MODEL_DIR, "encoder_cnn_toniot.tflite"),
    input_shape=(1, 16, 1),
    is_lstm=False
)

print("\n" + "=" * 60)
print("✅ CONVERSION TERMINÉE")
print("=" * 60)
print("""
Prochaine étape :
  1. Vérifier avec test_conversion.py que les scores
     match entre .h5 et .tflite sur quelques exemples
  2. Relancer semaine6_script1_iid.py (et 2, 3, 4)
  3. Comparer le nouveau F1 IID à celui de la Semaine 4
     (devrait être très proche maintenant, ~0.99 et non 0.91)
""")