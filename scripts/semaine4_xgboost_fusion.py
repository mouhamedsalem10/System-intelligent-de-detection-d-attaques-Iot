"""
XGBoost + Fusion + Évaluation finale

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix,
    classification_report, accuracy_score
)
import joblib
import json
import os
import time

os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)

print("=" * 60)
print("SEMAINE 4 — XGBOOST + FUSION + ÉVALUATION")
print("=" * 60)

# ============================================================
# ÉTAPE 1 — Charger les données prétraitées (de S3)
# ============================================================
print("\n📂 ÉTAPE 1 — Chargement des données (S3)")
print("-" * 50)

X_train_sc = np.load("data/X_train_scaled.npy")
y_train    = np.load("data/y_train.npy")
X_eval_sc  = np.load("data/X_eval_scaled.npy")
y_eval     = np.load("data/y_eval.npy")
types_eval = np.load("data/types_eval.npy",
                     allow_pickle=True)

with open("models/feature_names_toniot.json") as f:
    features = json.load(f)

print(f"  X_train : {X_train_sc.shape}")
print(f"  X_eval  : {X_eval_sc.shape}")

# ============================================================
# ÉTAPE 2 — Charger les encodeurs entraînés
# ============================================================
print("\n🧠 ÉTAPE 2 — Chargement des encodeurs")
print("-" * 50)

def charger_encodeur(chemin):
    if os.path.exists(chemin.replace('.h5', '.tflite')):
        interp = tf.lite.Interpreter(
            model_path=chemin.replace('.h5', '.tflite')
        )
        interp.allocate_tensors()
        return ('tflite', interp)
    elif os.path.exists(chemin):
        model = tf.keras.models.load_model(
            chemin, compile=False
        )
        return ('keras', model)
    else:
        raise FileNotFoundError(f"Modèle introuvable : {chemin}")

type_ae,   enc_ae   = charger_encodeur("models/encoder_ae_only.h5")
type_lstm, enc_lstm = charger_encodeur("models/encoder_lstm_only.h5")
type_cnn,  enc_cnn  = charger_encodeur("models/encoder_cnn_only.h5")

print(f"  ✅ AutoEncoder chargé ({type_ae})")
print(f"  ✅ LSTM chargé        ({type_lstm})")
print(f"  ✅ CNN 1D chargé      ({type_cnn})")

# ============================================================
# ÉTAPE 3 — Générer les embeddings et fusionner
# ============================================================
print("\n🔄 ÉTAPE 3 — Génération des embeddings")
print("-" * 50)

def get_embedding_keras(model, X, reshape=None):
    if reshape:
        X = X.reshape(reshape)
    return model.predict(X, verbose=0, batch_size=1024)

def get_embedding_tflite(interp, X_single):
    inp = interp.get_input_details()
    out = interp.get_output_details()
    X_r = X_single.reshape(inp[0]['shape'])
    interp.set_tensor(inp[0]['index'], X_r)
    interp.invoke()
    return interp.get_tensor(out[0]['index']).flatten()

def fusionner(X_sc, enc_ae, enc_lstm, enc_cnn,
              type_ae, type_lstm, type_cnn):
    n = len(X_sc)
    print(f"  Fusion de {n} flux...")

    if type_ae == 'keras':
        z_ae = get_embedding_keras(
            enc_ae, X_sc,
            reshape=(n, X_sc.shape[1])
        )
    else:
        z_ae = np.array([
            get_embedding_tflite(enc_ae, X_sc[i:i+1])
            for i in range(n)
        ])

    if type_lstm == 'keras':
        z_lstm = get_embedding_keras(
            enc_lstm, X_sc,
            reshape=(n, 1, X_sc.shape[1])
        )
    else:
        z_lstm = np.array([
            get_embedding_tflite(
                enc_lstm, X_sc[i:i+1].reshape(1,1,-1)
            )
            for i in range(n)
        ])

    if type_cnn == 'keras':
        z_cnn = get_embedding_keras(
            enc_cnn, X_sc,
            reshape=(n, X_sc.shape[1], 1)
        )
    else:
        z_cnn = np.array([
            get_embedding_tflite(
                enc_cnn, X_sc[i:i+1].reshape(1,-1,1)
            )
            for i in range(n)
        ])

    V_fused = np.concatenate(
        [X_sc, z_ae, z_lstm, z_cnn], axis=1
    ).astype(np.float32)

    print(f"  Vecteur fusionné : {V_fused.shape}")
    print(f"  = X_scaled({X_sc.shape[1]}) + "
          f"z_AE({z_ae.shape[1]}) + "
          f"z_LSTM({z_lstm.shape[1]}) + "
          f"z_CNN({z_cnn.shape[1]})")
    return V_fused

print("  Train...")
V_train = fusionner(X_train_sc, enc_ae, enc_lstm, enc_cnn,
                    type_ae, type_lstm, type_cnn)
print("  Eval...")
V_eval  = fusionner(X_eval_sc, enc_ae, enc_lstm, enc_cnn,
                    type_ae, type_lstm, type_cnn)

# ============================================================
# ÉTAPE 4 — Entraîner XGBoost
# ============================================================
print("\n🌲 ÉTAPE 4 — Entraînement XGBoost")
print("-" * 50)

X_tr, X_val, y_tr, y_val = train_test_split(
    V_train, y_train,
    test_size=0.2, random_state=42, stratify=y_train
)

dtrain = xgb.DMatrix(X_tr,   label=y_tr)
dval   = xgb.DMatrix(X_val,  label=y_val)
deval  = xgb.DMatrix(V_eval, label=y_eval)

# ============================================================
# JUSTIFICATION DES HYPERPARAMÈTRES XGBOOST :
#   n_estimators=150  : 150 arbres → bon équilibre
#   max_depth=6       : profondeur max → évite surapprentissage
#   learning_rate=0.1 : pas d'apprentissage standard
#   subsample=0.8     : 80% données par arbre → régularisation
#   colsample=0.8     : 80% features par arbre → diversité
#   scale_pos_weight  : gestion déséquilibre classes
# ============================================================
n_normal  = int((y_train == 0).sum())
n_attaque = int((y_train == 1).sum())
scale_pos = n_normal / n_attaque

params = {
    'objective'        : 'binary:logistic',
    'eval_metric'      : ['logloss', 'auc'],
    'max_depth'        : 6,
    'learning_rate'    : 0.1,
    'n_estimators'     : 150,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'scale_pos_weight' : scale_pos,
    'seed'             : 42,
    'device'           : 'cpu'
}

print(f"  scale_pos_weight : {scale_pos:.2f}")
print(f"  (compense déséquilibre normal/attaque)")

t_start = time.time()
model_xgb = xgb.train(
    params,
    dtrain,
    num_boost_round=150,
    evals=[(dtrain,'train'), (dval,'val')],
    early_stopping_rounds=15,
    verbose_eval=10
)
t_train = time.time() - t_start

print(f"\n  ✅ XGBoost entraîné en {t_train:.1f}s")
print(f"  Meilleur round : {model_xgb.best_iteration}")

# ============================================================
# ÉTAPE 5 — Évaluation finale
# ============================================================
print("\n📊 ÉTAPE 5 — ÉVALUATION FINALE")
print("-" * 50)

probas = model_xgb.predict(deval)
y_pred = (probas >= 0.5).astype(int)

accuracy  = accuracy_score(y_eval, y_pred)
precision = precision_score(y_eval, y_pred,
                            zero_division=0)
recall    = recall_score(y_eval, y_pred,
                         zero_division=0)
f1        = f1_score(y_eval, y_pred, zero_division=0)
auc       = roc_auc_score(y_eval, probas)
cm        = confusion_matrix(y_eval, y_pred)
TN, FP    = int(cm[0,0]), int(cm[0,1])
FN, TP    = int(cm[1,0]), int(cm[1,1])

print(f"""
  Résultats sur eval_test_full.csv :
  ─────────────────────────────────
  Accuracy  : {accuracy*100:.2f}%
  Precision : {precision*100:.2f}%
  Recall    : {recall*100:.2f}%
  F1-Score  : {f1:.4f}
  AUC-ROC   : {auc:.4f}

  Matrice de confusion :
  ┌──────────┬──────────┬──────────┐
  │          │ Prédit N │ Prédit A │
  ├──────────┼──────────┼──────────┤
  │ Réel N   │ TN={TN:5d} │ FP={FP:5d} │
  │ Réel A   │ FN={FN:5d} │ TP={TP:5d} │
  └──────────┴──────────┴──────────┘
""")

# Métriques par type
print("  Métriques par type d'attaque :")
print(f"  {'Type':<15} {'N':>6} {'Recall':>8} {'F1':>8}")
print("  " + "-" * 42)
resultats_types = {}
for t in sorted(set(types_eval)):
    mask = types_eval == t
    yt   = y_eval[mask]
    yp   = y_pred[mask]
    if yt.sum() > 0:
        r = recall_score(yt, yp, zero_division=0)
        f = f1_score(yt, yp, zero_division=0)
        status = "✅" if r >= 0.95 else "⚠️ "
        print(f"  {status} {t:<13} "
              f"{mask.sum():>6} {r:>8.4f} {f:>8.4f}")
        resultats_types[t] = {
            'recall': round(float(r), 4),
            'f1'    : round(float(f), 4),
            'n'     : int(mask.sum())
        }

# ============================================================
# ÉTAPE 6 — Sauvegarde
# ============================================================
print("\n💾 ÉTAPE 6 — SAUVEGARDE")
print("-" * 50)

model_xgb.save_model("models/xgb_toniot.json")
print("  ✅ models/xgb_toniot.json")

metriques = {
    "modele"    : "Pipeline hybride AE+LSTM+CNN+XGBoost",
    "dataset"   : "eval_test_full.csv",
    "n_flux"    : int(len(y_eval)),
    "accuracy"  : round(float(accuracy), 4),
    "precision" : round(float(precision), 4),
    "recall"    : round(float(recall), 4),
    "f1_score"  : round(float(f1), 4),
    "auc_roc"   : round(float(auc), 4),
    "confusion_matrix": {
        "TN":TN,"FP":FP,"FN":FN,"TP":TP
    },
    "par_type"  : resultats_types
}
with open("results/semaine4_metriques.json",
          "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=False)
print("  ✅ results/semaine4_metriques.json")

# ============================================================
# ÉTAPE 7 — Figures scientifiques finales
# ============================================================
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "Semaine 4 — Pipeline AE+LSTM+CNN+XGBoost\n"
    f"Évaluation finale | F1={f1:.4f} | "
    f"AUC={auc:.4f} | Accuracy={accuracy*100:.2f}%",
    fontsize=13, fontweight='bold'
)

# Figure 1 : Matrice confusion
ax1 = axes[0, 0]
cm_norm = cm.astype(float)/cm.sum(axis=1,keepdims=True)
im = ax1.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax1)
ax1.set_xticks([0,1])
ax1.set_yticks([0,1])
ax1.set_xticklabels(['Normal','Attaque'])
ax1.set_yticklabels(['Normal','Attaque'])
ax1.set_xlabel('Classe prédite')
ax1.set_ylabel('Classe réelle')
ax1.set_title('Matrice de Confusion (normalisée)')
for i in range(2):
    for j in range(2):
        color = 'white' if cm_norm[i,j]>0.5 else 'black'
        ax1.text(j, i,
                 f'{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)',
                 ha='center', va='center',
                 color=color, fontsize=12,
                 fontweight='bold')

# Figure 2 : Distribution scores
ax2 = axes[0, 1]
ax2.hist(probas[y_eval==0], bins=60, alpha=0.6,
         color='#2ecc71', label='Normal', density=True)
ax2.hist(probas[y_eval==1], bins=60, alpha=0.6,
         color='#e74c3c', label='Attaque', density=True)
ax2.axvline(x=0.3, color='#f39c12', linewidth=2,
            linestyle='--', label='MEDIUM (0.30)')
ax2.axvline(x=0.8, color='#c0392b', linewidth=2,
            linestyle='--', label='HIGH (0.80)')
ax2.set_title('Distribution des scores de probabilité')
ax2.set_xlabel('Score p')
ax2.set_ylabel('Densité')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Figure 3 : Recall par type
ax3 = axes[1, 0]
if resultats_types:
    t_names  = list(resultats_types.keys())
    t_recall = [resultats_types[t]['recall']
                for t in t_names]
    t_colors = ['#2ecc71' if r >= 0.95 else '#e74c3c'
                for r in t_recall]
    bars = ax3.bar(range(len(t_names)), t_recall,
                   color=t_colors, edgecolor='black')
    ax3.set_xticks(range(len(t_names)))
    ax3.set_xticklabels(t_names, rotation=45,
                        ha='right', fontsize=9)
    ax3.set_title('Recall par type d\'attaque')
    ax3.set_ylabel('Recall')
    ax3.set_ylim(0, 1.15)
    ax3.axhline(y=0.95, color='black',
                linestyle='--', label='Seuil 95%')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, t_recall):
        ax3.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.01,
                 f'{val:.3f}', ha='center', fontsize=8)

# Figure 4 : Courbe ROC
ax4 = axes[1, 1]
fpr, tpr, _ = roc_curve(y_eval, probas)
ax4.plot(fpr, tpr, color='#2980b9', linewidth=2.5,
         label=f'Pipeline hybride (AUC={auc:.4f})')
ax4.plot([0,1],[0,1], 'k--', linewidth=1,
         label='Aléatoire (AUC=0.5)')
ax4.fill_between(fpr, tpr, alpha=0.1, color='#2980b9')
ax4.set_title('Courbe ROC')
ax4.set_xlabel('Taux Faux Positifs (FPR)')
ax4.set_ylabel('Taux Vrais Positifs (TPR)')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/semaine4_evaluation_finale.png',
            dpi=150, bbox_inches='tight')
print("  ✅ results/semaine4_evaluation_finale.png")

print(f"""
✅ SEMAINE 4 TERMINÉE !

Modèles entraînés :
  📁 models/encoder_ae_only.h5    (AutoEncoder)
  📁 models/encoder_lstm_only.h5  (LSTM)
  📁 models/encoder_cnn_only.h5   (CNN 1D)
  📁 models/xgb_toniot.json       (XGBoost)

Résultats :
  Accuracy  : {accuracy*100:.2f}%
  Recall    : {recall*100:.2f}%
  F1-Score  : {f1:.4f}
  AUC-ROC   : {auc:.4f}

Figures :
  📊 results/semaine4_evaluation_finale.png
  📄 results/semaine4_metriques.json

Semaine 5 : Pipeline MQTT + Docker + SHAP
""")