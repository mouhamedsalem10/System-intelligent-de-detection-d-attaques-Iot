"""
SEMAINE 3 — Prétraitement COMPLÉMENTAIRE (CORRIGÉ)
====================================================
CORRECTIF CRITIQUE : une analyse a révélé que eval_test_full.csv
est un sous-ensemble EXACT de train_test_networks.csv (100% de
chevauchement sur les 16 features + label + type). Ce script
retire désormais ces flux de l'ensemble d'entraînement avant
tout fit de scaler ou d'entraînement de modèle, afin d'éliminer
toute fuite de données entre train et évaluation.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import joblib
import json
import os

os.makedirs("results", exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("data",    exist_ok=True)

FEATURES_16 = [
    'src_port', 'dst_port', 'duration',
    'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qclass', 'dns_qtype', 'dns_rcode',
    'http_request_body_len', 'http_response_body_len',
    'http_status_code'
]

print("=" * 60)
print("SEMAINE 3 — PRÉTRAITEMENT COMPLÉMENTAIRE (CORRIGÉ)")
print("=" * 60)

# ============================================================
# ÉTAPE 1 — Charger les deux fichiers
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 1 — CHARGEMENT DES DEUX FICHIERS")
print("=" * 60)

df_train_full = pd.read_csv("data/train_test_networks.csv", sep=';', low_memory=False)
df_eval       = pd.read_csv("data/eval_test_full.csv",      sep=',', low_memory=False)

print(f"\n  train_test_networks.csv (brut) : {len(df_train_full):>7} lignes")
print(f"  eval_test_full.csv             : {len(df_eval):>7} lignes")

# ============================================================
# ÉTAPE 2 — CORRECTIF CRITIQUE : retirer de train les flux
#           présents dans eval (fuite de données)
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 2 — SUPPRESSION DES FLUX EVAL PRÉSENTS DANS TRAIN")
print("=" * 60)

key_cols = FEATURES_16 + ['label', 'type']

df_train_full['_key'] = df_train_full[key_cols].apply(tuple, axis=1)
df_eval['_key']       = df_eval[key_cols].apply(tuple, axis=1)

eval_keys_set = set(df_eval['_key'])
mask_leak     = df_train_full['_key'].isin(eval_keys_set)

n_leak = int(mask_leak.sum())
print(f"\n  ⚠️  Flux identifiés comme fuite (présents aussi dans eval) : {n_leak}")
print(f"      ({n_leak/len(df_eval)*100:.1f}% du jeu eval, soit "
      f"{n_leak/len(df_train_full)*100:.1f}% du jeu train brut)")

df_train_clean = df_train_full[~mask_leak].drop(columns=['_key']).copy()
df_eval        = df_eval.drop(columns=['_key'])

print(f"\n  Train AVANT nettoyage : {len(df_train_full):>7} lignes")
print(f"  Train APRÈS nettoyage : {len(df_train_clean):>7} lignes")
print(f"  ✅ Plus aucun chevauchement possible avec eval_test_full.csv")

# Vérification finale (doit afficher 0)
verif_keys = set(df_train_clean[key_cols].apply(tuple, axis=1))
verif_overlap = len(verif_keys & eval_keys_set)
print(f"\n  🔍 Vérification post-nettoyage : chevauchement résiduel = {verif_overlap}")
if verif_overlap == 0:
    print("     ✅ CONFIRMÉ — aucune fuite résiduelle")
else:
    print("     ❌ ATTENTION — chevauchement encore présent, ne pas continuer")

df_train = df_train_clean  # à partir d'ici, df_train = train PROPRE

# ============================================================
# ÉTAPE 3 — Extraction et nettoyage des features
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 3 — EXTRACTION DES 16 FEATURES")
print("=" * 60)

features_train = [f for f in FEATURES_16 if f in df_train.columns]
features_eval  = [f for f in FEATURES_16 if f in df_eval.columns]

X_train = df_train[features_train].replace(
    [np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y_train  = df_train['label'].astype(np.int32)
typ_train = df_train['type'].copy()

X_eval  = df_eval[features_eval].replace(
    [np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y_eval  = df_eval['label'].astype(np.int32)
typ_eval = df_eval['type'].copy() if 'type' in df_eval.columns else None

print(f"\n  X_train (nettoyé, sans fuite) : {X_train.shape}")
print(f"  X_eval                         : {X_eval.shape}")

print(f"\n  Distribution train (après nettoyage) :")
print(f"    Normal  : {(y_train==0).sum()}")
print(f"    Attaque : {(y_train==1).sum()}")

# ============================================================
# ÉTAPE 4 — Normalisation (scaler fit sur train nettoyé UNIQUEMENT)
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 4 — NORMALISATION (train nettoyé → eval)")
print("=" * 60)

scaler = StandardScaler()
scaler.fit(X_train)  # uniquement sur le train nettoyé, sans fuite

X_train_sc = scaler.transform(X_train).astype(np.float32)
X_eval_sc  = scaler.transform(X_eval).astype(np.float32)

print(f"\n  Scaler entraîné sur train nettoyé ({len(X_train)} flux)")
print(f"  Appliqué sur train : {X_train_sc.shape}")
print(f"  Appliqué sur eval  : {X_eval_sc.shape}")

idx = {f: i for i, f in enumerate(features_train)}
print(f"\n  Vérification (moyenne ≈ 0, std ≈ 1) :")
print(f"    src_bytes → moy={X_train_sc[:,idx['src_bytes']].mean():.4f}, "
      f"std={X_train_sc[:,idx['src_bytes']].std():.4f}")
print(f"    duration  → moy={X_train_sc[:,idx['duration']].mean():.4f}, "
      f"std={X_train_sc[:,idx['duration']].std():.4f}")

joblib.dump(scaler, "models/scaler_toniot.joblib")
print(f"\n  ✅ Scaler sauvegardé (models/scaler_toniot.joblib)")

# ============================================================
# ÉTAPE 5 — Sauvegarde .npy pour les modèles (Semaine 4)
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 5 — SAUVEGARDE .NPY POUR SEMAINE 4")
print("=" * 60)

np.save("data/X_train_scaled.npy", X_train_sc)
np.save("data/X_eval_scaled.npy",  X_eval_sc)
np.save("data/y_train.npy",        y_train.values)
np.save("data/y_eval.npy",         y_eval.values)
np.save("data/types_train.npy",    typ_train.values.astype(str))
if typ_eval is not None:
    np.save("data/types_eval.npy", typ_eval.values.astype(str))

with open("models/feature_names_toniot.json", "w") as f:
    json.dump(features_train, f, indent=2)

print(f"  ✅ data/X_train_scaled.npy  {X_train_sc.shape}")
print(f"  ✅ data/X_eval_scaled.npy   {X_eval_sc.shape}")
print(f"  ✅ data/y_train.npy         {y_train.shape}")
print(f"  ✅ data/y_eval.npy          {y_eval.shape}")
print(f"  ✅ models/feature_names_toniot.json")

# ============================================================
# ÉTAPE 6 — Analyse corrélation features vs label
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 6 — CORRÉLATION FEATURES vs LABEL")
print("=" * 60)

df_c = pd.DataFrame(X_train_sc, columns=features_train)
df_c['label'] = y_train.values
corrs = df_c.corr()['label'].drop('label').sort_values(
    key=abs, ascending=False
)

print(f"\n  {'Feature':<28} {'Corrélation':>12}  Niveau")
print("  " + "-" * 58)
for feat, corr in corrs.items():
    niveau = ("🔴 FORTE"   if abs(corr) > 0.3 else
              "🟡 MOYENNE" if abs(corr) > 0.1 else
              "🟢 FAIBLE")
    print(f"  {feat:<28} {corr:>+12.4f}  {niveau}")

sorted_shap = None
shap_path = "models/shap_importance.json"
if os.path.exists(shap_path):
    with open(shap_path) as f:
        shap_data = json.load(f)
    shap_imp = shap_data["feature_importance_mean_abs_shap"]
    sorted_shap = sorted(shap_imp.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 features selon SHAP :")
    for rang, (feat, val) in enumerate(sorted_shap[:5], 1):
        print(f"    {rang}. {feat:<28} SHAP={val:.4f}")
else:
    print("\n  ⚠️  shap_importance.json non trouvé (sera généré plus tard)")

# ============================================================
# ÉTAPE 7 — SMOTE pour MitM (sur train nettoyé)
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 7 — SMOTE POUR CLASSE MitM (sur train nettoyé)")
print("=" * 60)

le = LabelEncoder()
typ_enc = le.fit_transform(typ_train)
mitm_label  = le.transform(['mitm'])[0]
TARGET_MITM = 5000

print(f"\n  Avant SMOTE :")
dist_av = typ_train.value_counts()
total_av = len(typ_train)
for t, c in dist_av.items():
    pct = c / total_av * 100
    flag = " ⚠️ RARE" if pct < 2 else ""
    print(f"    {t:<15} : {c:>6} ({pct:.1f}%){flag}")

n_mitm_avant = int((typ_enc == mitm_label).sum())
if n_mitm_avant < TARGET_MITM:
    print(f"\n  Application SMOTE (MitM : {n_mitm_avant} → {TARGET_MITM})...")
    smote = SMOTE(
        sampling_strategy={mitm_label: TARGET_MITM},
        k_neighbors=5,
        random_state=42
    )
    X_res, y_res = smote.fit_resample(X_train.values, typ_enc)

    df_res         = pd.DataFrame(X_res, columns=features_train)
    df_res['type'] = le.inverse_transform(y_res)
    df_res['label'] = (df_res['type'] != 'normal').astype(int)
    df_res.to_csv("data/train_resampled_mitm.csv", index=False)

    print(f"\n  Après SMOTE :")
    dist_ap = df_res['type'].value_counts()
    total_ap = len(df_res)
    for t, c in dist_ap.items():
        pct = c / total_ap * 100
        print(f"    {t:<15} : {c:>6} ({pct:.1f}%)")
    print(f"\n  ✅ data/train_resampled_mitm.csv sauvegardé")
else:
    dist_ap = dist_av
    print(f"  MitM déjà >= {TARGET_MITM}, SMOTE non nécessaire")

# ============================================================
# ÉTAPE 8 — Figures scientifiques
# ============================================================
print("\n" + "=" * 60)
print("ÉTAPE 8 — FIGURES SCIENTIFIQUES")
print("=" * 60)

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Semaine 3 — Prétraitement Complémentaire TON-IoT (corrigé)\n"
        f"Train (sans fuite) : {len(X_train):,} flux | "
        f"Eval : {len(X_eval):,} flux | "
        f"Flux retirés (fuite) : {n_leak:,}",
        fontsize=13, fontweight='bold'
    )

    ax1 = axes[0, 0]
    colors = ['#2ecc71' if t == 'normal' else '#e74c3c' for t in dist_av.index]
    ax1.bar(range(len(dist_av)), dist_av.values, color=colors, edgecolor='black')
    ax1.set_xticks(range(len(dist_av)))
    ax1.set_xticklabels(dist_av.index, rotation=45, ha='right', fontsize=8)
    ax1.set_title('Distribution par type (train nettoyé)', fontsize=11)
    ax1.set_ylabel('Nombre de flux')

    ax2 = axes[0, 1]
    cv = corrs.values
    cn = [n[:14] for n in corrs.index]
    cc = ['#e74c3c' if abs(v) > 0.3 else '#f39c12' if abs(v) > 0.1 else '#2ecc71' for v in cv]
    ax2.barh(range(len(cv)), cv, color=cc, edgecolor='black')
    ax2.set_yticks(range(len(cn)))
    ax2.set_yticklabels(cn, fontsize=8)
    ax2.set_title('Corrélation features vs label', fontsize=11)
    ax2.set_xlabel('Corrélation de Pearson')
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    if sorted_shap:
        s_names = [n[:14] for n, _ in reversed(sorted_shap)]
        s_vals  = [v for _, v in reversed(sorted_shap)]
        s_col   = ['#e74c3c' if v > 1.0 else '#f39c12' if v > 0.3 else '#2ecc71' for v in s_vals]
        ax3.barh(range(len(s_names)), s_vals, color=s_col, edgecolor='black')
        ax3.set_yticks(range(len(s_names)))
        ax3.set_yticklabels(s_names, fontsize=8)
        ax3.set_title('Importance SHAP (XGBoost)', fontsize=11)
        ax3.set_xlabel('Mean |SHAP value|')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
        ax3.set_title('SHAP non disponible', fontsize=11)

    ax4 = axes[1, 0]
    seuil = np.percentile(X_train.values[:, idx['src_bytes']], 95)
    pd.Series(X_train.values[:, idx['src_bytes']]).clip(upper=seuil).hist(
        ax=ax4, bins=60, color='#3498db', edgecolor='black')
    ax4.set_title('src_bytes AVANT normalisation', fontsize=11)
    ax4.set_xlabel('Valeur brute (octets)')
    ax4.set_ylabel('Fréquence')

    ax5 = axes[1, 1]
    seuil_s = np.percentile(X_train_sc[:, idx['src_bytes']], 95)
    pd.Series(X_train_sc[:, idx['src_bytes']]).clip(upper=seuil_s).hist(
        ax=ax5, bins=60, color='#9b59b6', edgecolor='black')
    ax5.set_title('src_bytes APRÈS normalisation\n(StandardScaler)', fontsize=11)
    ax5.set_xlabel('Valeur normalisée')
    ax5.set_ylabel('Fréquence')

    ax6 = axes[1, 2]
    labels_comp = ['Avant SMOTE', 'Après SMOTE']
    mitm_counts = [dist_av.get('mitm', 0), dist_ap.get('mitm', 0)]
    bars6 = ax6.bar(labels_comp, mitm_counts, color=['#e74c3c', '#2ecc71'],
                    edgecolor='black', width=0.5)
    ax6.set_title('SMOTE — Classe MitM\n(Chawla et al., 2002)', fontsize=11)
    ax6.set_ylabel("Nombre d'exemples MitM")
    for bar, val in zip(bars6, mitm_counts):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, max(mitm_counts) * 1.2)

    plt.tight_layout()
    plt.savefig('results/semaine3_complet.png', dpi=150, bbox_inches='tight')
    print(f"  ✅ results/semaine3_complet.png")

except Exception as e:
    print(f"⚠️  Erreur génération figures : {e}")
    print("   (pas grave — les données sont déjà sauvegardées)")

# ============================================================
# RÉSUMÉ FINAL + MÉTRIQUES JSON
# ============================================================
print("\n" + "=" * 60)
print("RÉSUMÉ SEMAINE 3 (CORRIGÉ)")
print("=" * 60)
print(f"""
CORRECTIF APPLIQUÉ :
  ⚠️  {n_leak} flux de train_test_networks.csv étaient identiques
      (16 features + label + type) à des flux de eval_test_full.csv
  ✅ Ces flux ont été RETIRÉS du train avant tout entraînement
  ✅ Vérification post-nettoyage : chevauchement résiduel = {verif_overlap}

Fichiers créés :
  📁 data/X_train_scaled.npy  {X_train_sc.shape}
  📁 data/X_eval_scaled.npy   {X_eval_sc.shape}
  📁 data/y_train.npy
  📁 data/y_eval.npy
  📁 data/train_resampled_mitm.csv
  📁 models/scaler_toniot.joblib
  📁 models/feature_names_toniot.json
  📊 results/semaine3_complet.png

🚀 Prochaine étape : ré-entraîner TOUT depuis ces nouvelles données
   (train_autoencoder.py → LSTM → CNN → semaine4_xgboost_fusion.py
   → reconversion .tflite → réévaluation Semaine 6)
""")

metrics = {
    "week": 3,
    "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "train_size_before_cleaning": int(len(df_train_full)),
    "train_size_after_cleaning": int(len(df_train)),
    "leaked_rows_removed": int(n_leak),
    "eval_size": int(len(X_eval)),
    "features": len(features_train),
    "normal_train": int(y_train.value_counts().get(0, 0)),
    "attack_train": int(y_train.value_counts().get(1, 0)),
    "mitm_before_smote": int(dist_av.get('mitm', 0)),
    "mitm_after_smote": int(dist_ap.get('mitm', 0)),
    "correlation_top5": {k: round(float(v), 4) for k, v in corrs.head(5).items()},
    "scaler": "StandardScaler (fit sur train nettoyé uniquement)",
    "smote_target": TARGET_MITM,
    "data_leakage_check": "eval_test_full.csv retiré de train_test_networks.csv avant tout traitement"
}

with open("results/metrics_semaine3.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métriques sauvegardées dans results/metrics_semaine3.json")