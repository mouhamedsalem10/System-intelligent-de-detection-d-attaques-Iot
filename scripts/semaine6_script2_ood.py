"""
 Évaluation OOD

"""

import sys, os
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from app.inference import EdgeInference

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("SEMAINE 6 — SCRIPT 2 : ÉVALUATION OOD")
print("Dataset : eval_ood_dst_ip_balanced.csv")
print("=" * 60)

engine = EdgeInference(model_dir="models")

# Charger le dataset OOD
ood_path = "data/eval_ood_dst_ip_balanced.csv"
if not os.path.exists(ood_path):
    print(f"❌ Fichier non trouvé : {ood_path}")
    exit(1)

df     = pd.read_csv(ood_path, sep=',')
y_true = df['label'].astype(int).values
types  = df['type'].astype(str).values \
         if 'type' in df.columns else None

print(f"Dataset OOD : {len(df)} flux")
print(f"Normal (0)  : {(y_true==0).sum()}")
print(f"Attaque (1) : {(y_true==1).sum()}")

# Inférence
print("\nInférence en cours...")
probas  = []
t_start = time.time()

for i, (_, row) in enumerate(df.iterrows()):
    feats = {}
    for k in engine.feature_names:
        v = row.get(k, 0.0)
        try:
            feats[k] = 0.0 if pd.isna(v) else float(v)
        except Exception:
            feats[k] = 0.0
    probas.append(
        engine.predict_proba_from_feature_dict(feats)
    )
    if (i+1) % 2000 == 0:
        print(f"  {i+1}/{len(df)} flux traités...")

t_total = time.time() - t_start
probas  = np.array(probas)
y_pred  = (probas >= 0.5).astype(int)

# Métriques
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_true, probas)
except Exception:
    auc = None
cm      = confusion_matrix(y_true, y_pred)
TN, FP  = int(cm[0,0]), int(cm[0,1])
FN, TP  = int(cm[1,0]), int(cm[1,1])

print(f"""
MÉTRIQUES GLOBALES — Scénario OOD
  Accuracy  : {accuracy*100:.2f}%
  Precision : {precision*100:.2f}%
  Recall    : {recall*100:.2f}%
  F1-Score  : {f1:.4f}
  AUC-ROC   : {f'{auc:.4f}' if auc else 'N/A'}

  Matrice de confusion :
  TN={TN:,} | FP={FP:,}
  FN={FN:,} | TP={TP:,}
""")

# Métriques par type
resultats_types = {}
if types is not None:
    print("MÉTRIQUES PAR TYPE :")
    for t in sorted(set(types)):
        mask = types == t
        yt   = y_true[mask]
        yp   = y_pred[mask]
        if yt.sum() > 0:
            r = recall_score(yt, yp, zero_division=0)
            f = f1_score(yt, yp, zero_division=0)
            s = "✅" if r >= 0.95 else "⚠️ "
            print(f"  {s} {t:<15} : "
                  f"Recall={r:.4f} F1={f:.4f}")
            resultats_types[t] = {
                'recall': round(float(r), 4),
                'f1'    : round(float(f), 4)
            }

# Figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks([0,1])
ax.set_yticks([0,1])
ax.set_xticklabels(['Normal','Attaque'])
ax.set_yticklabels(['Normal','Attaque'])
ax.set_xlabel('Prédit')
ax.set_ylabel('Réel')
ax.set_title(
    f'Scénario OOD — F1={f1:.4f}\n'
    f'(IPs de destination inconnues)',
    fontsize=12
)
for i in range(2):
    for j in range(2):
        c = 'white' if cm_norm[i,j] > 0.5 else 'black'
        ax.text(j, i,
                f'{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)',
                ha='center', va='center',
                color=c, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/s6_ood.png', dpi=150,
            bbox_inches='tight')
print("✅ results/s6_ood.png")

# Sauvegarder
with open("results/s6_metriques_ood.json", "w") as f:
    json.dump({
        "scenario"  : "OOD",
        "dataset"   : "eval_ood_dst_ip_balanced.csv",
        "n_flux"    : int(len(df)),
        "accuracy"  : round(float(accuracy), 4),
        "precision" : round(float(precision), 4),
        "recall"    : round(float(recall), 4),
        "f1_score"  : round(float(f1), 4),
        "auc_roc"   : round(float(auc), 4) if auc else None,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP,
        "par_type"  : resultats_types
    }, f, indent=2, ensure_ascii=False)
print(" results/s6_metriques_ood.json")
print(f"\n SCRIPT 2 TERMINÉ — F1={f1:.4f}")