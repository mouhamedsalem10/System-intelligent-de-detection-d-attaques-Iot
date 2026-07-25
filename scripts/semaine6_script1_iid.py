"""
 Évaluation IID

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
    recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    roc_curve
)
from app.inference import EdgeInference

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("SEMAINE 6 — SCRIPT 1 : ÉVALUATION IID")
print("Dataset : eval_test_full.csv")
print("=" * 60)

# Charger le pipeline
engine = EdgeInference(model_dir="models")

# Charger le dataset
df     = pd.read_csv("data/eval_test_full.csv", sep=',')
y_true = df['label'].astype(int).values
types  = df['type'].astype(str).values \
         if 'type' in df.columns else None

print(f"\nDataset chargé : {len(df)} flux")
print(f"Normal (0)     : {(y_true==0).sum()}")
print(f"Attaque (1)    : {(y_true==1).sum()}")

# Inférence
print(f"\nInférence en cours...")
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
    if (i+1) % 5000 == 0:
        print(f"  {i+1}/{len(df)} flux traités...")

t_total = time.time() - t_start
probas  = np.array(probas)
y_pred  = (probas >= 0.5).astype(int)

print(f"\n Inférence terminée en {t_total:.1f}s")

# Métriques globales
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
auc       = roc_auc_score(y_true, probas)
cm        = confusion_matrix(y_true, y_pred)
TN, FP    = int(cm[0,0]), int(cm[0,1])
FN, TP    = int(cm[1,0]), int(cm[1,1])

print(f"""
MÉTRIQUES GLOBALES — Scénario IID
  Accuracy  : {accuracy*100:.2f}%
  Precision : {precision*100:.2f}%
  Recall    : {recall*100:.2f}%
  F1-Score  : {f1:.4f}
  AUC-ROC   : {auc:.4f}

  Matrice de confusion :
  TN={TN:,} | FP={FP:,}
  FN={FN:,} | TP={TP:,}
""")

# Métriques par type
resultats_types = {}
if types is not None:
    print("MÉTRIQUES PAR TYPE :")
    print(f"  {'Type':<15} {'N':>6} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 42)
    for t in sorted(set(types)):
        mask = types == t
        yt   = y_true[mask]
        yp   = y_pred[mask]
        if yt.sum() > 0:
            r = recall_score(yt, yp, zero_division=0)
            f = f1_score(yt, yp, zero_division=0)
            s = "✅" if r >= 0.95 else "⚠️ "
            print(f"  {s} {t:<13} {mask.sum():>6} "
                  f"{r:>8.4f} {f:>8.4f}")
            resultats_types[t] = {
                'recall': round(float(r), 4),
                'f1'    : round(float(f), 4),
                'n'     : int(mask.sum())
            }

# Figures scientifiques
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Scénario IID — eval_test_full.csv\n"
    f"F1={f1:.4f} | AUC={auc:.4f} | "
    f"Accuracy={accuracy*100:.2f}%",
    fontsize=13, fontweight='bold'
)

# Matrice de confusion
ax1 = axes[0]
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax1.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax1)
ax1.set_xticks([0,1])
ax1.set_yticks([0,1])
ax1.set_xticklabels(['Normal','Attaque'])
ax1.set_yticklabels(['Normal','Attaque'])
ax1.set_xlabel('Prédit')
ax1.set_ylabel('Réel')
ax1.set_title('Matrice de Confusion (normalisée)')
for i in range(2):
    for j in range(2):
        c = 'white' if cm_norm[i,j] > 0.5 else 'black'
        ax1.text(j, i,
                 f'{cm[i,j]:,}\n({cm_norm[i,j]*100:.1f}%)',
                 ha='center', va='center',
                 color=c, fontsize=11, fontweight='bold')

# Recall par type
ax2 = axes[1]
if resultats_types:
    noms    = list(resultats_types.keys())
    recalls = [resultats_types[t]['recall'] for t in noms]
    colors  = ['#2ecc71' if r >= 0.95 else '#e74c3c'
               for r in recalls]
    bars = ax2.bar(range(len(noms)), recalls,
                   color=colors, edgecolor='black')
    ax2.set_xticks(range(len(noms)))
    ax2.set_xticklabels(noms, rotation=45,
                        ha='right', fontsize=9)
    ax2.set_title('Recall par type d\'attaque')
    ax2.set_ylabel('Recall')
    ax2.set_ylim(0, 1.12)
    ax2.axhline(y=0.95, color='orange',
                linestyle='--', label='Seuil 95%')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, recalls):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/s6_iid.png', dpi=150,
            bbox_inches='tight')
print("\n results/s6_iid.png")

# Sauvegarder
metriques_iid = {
    "scenario"  : "IID",
    "dataset"   : "eval_test_full.csv",
    "n_flux"    : int(len(df)),
    "accuracy"  : round(float(accuracy), 4),
    "precision" : round(float(precision), 4),
    "recall"    : round(float(recall), 4),
    "f1_score"  : round(float(f1), 4),
    "auc_roc"   : round(float(auc), 4),
    "TN": TN, "FP": FP, "FN": FN, "TP": TP,
    "par_type"  : resultats_types,
    "vitesse"   : round(len(df)/t_total, 1)
}
with open("results/s6_metriques_iid.json", "w") as f:
    json.dump(metriques_iid, f, indent=2, ensure_ascii=False)
print(" results/s6_metriques_iid.json")
print(f"\n SCRIPT 1 TERMINÉ — F1={f1:.4f}")