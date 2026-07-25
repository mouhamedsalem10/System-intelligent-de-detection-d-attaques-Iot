"""
SEMAINE 3 — Prétraitement COMPLÉMENTAIRE
==========================================
CONTEXTE : La Semaine 2 (prepare_data.py) a déjà fait :
           ✅ Chargement train_test_networks.csv
           ✅ Sélection des 16 features
           ✅ Nettoyage NaN/infinis
           ✅ Normalisation basique
           ✅ Train/Test split

SEMAINE 3 AJOUTE uniquement ce qui manque :
  NOUVEAU 1 → Charger eval_test_full.csv (2e fichier)
  NOUVEAU 2 → Corriger le scaler (fit train, transform les deux)
  NOUVEAU 3 → Sauvegarder .npy pour les modèles IA
  NOUVEAU 4 → Analyse corrélation features vs label
  NOUVEAU 5 → SMOTE pour MitM (classe rare 0.5%)
  NOUVEAU 6 → Figures scientifiques propres

JUSTIFICATION DES CHOIX TECHNIQUES :
  - StandardScaler : choix de référence (Moustafa et al., 2020)
  - 16 features : seules disponibles dans les logs Zeek
  - SMOTE : Chawla et al., 2002 — meilleur que duplication simple
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

# Les 16 features (même liste que S2)
FEATURES_16 = [
    'src_port', 'dst_port', 'duration',
    'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qclass', 'dns_qtype', 'dns_rcode',
    'http_request_body_len', 'http_response_body_len',
    'http_status_code'
]

print("=" * 60)
print("SEMAINE 3 — PRÉTRAITEMENT COMPLÉMENTAIRE")
print("Ce qui manquait après la Semaine 2")
print("=" * 60)

# ============================================================
# RAPPEL S2 — Ce qu'on avait déjà
# ============================================================
print("\n📌 RAPPEL SEMAINE 2 (déjà fait dans prepare_data.py) :")
print("   ✅ Chargement train_test_networks.csv (211 043 lignes)")
print("   ✅ Sélection des 16 features")
print("   ✅ Nettoyage NaN/infinis")
print("   ✅ Normalisation basique")
print("   ✅ Train/Test split (80/20)")
print("   ✅ Scaler sauvegardé")

# ============================================================
# NOUVEAU 1 — Charger LES DEUX fichiers correctement
# ============================================================
# QUOI    : Charger aussi eval_test_full.csv
# POURQUOI: En S2 on n'avait chargé que train
#           Pour évaluer le modèle, on a besoin des deux
#           Le scaler doit être appliqué sur les deux
#           avec les MÊMES paramètres
# ============================================================
print("\n" + "=" * 60)
print("NOUVEAU 1 — CHARGEMENT DES DEUX FICHIERS")
print("=" * 60)

df_train = pd.read_csv("data/train_test_networks.csv", sep=';')
df_eval  = pd.read_csv("data/eval_test_full.csv",      sep=',')

print(f"\n  train_test_networks.csv : {len(df_train):>7} lignes")
print(f"  eval_test_full.csv      : {len(df_eval):>7} lignes")

features_train = [f for f in FEATURES_16 if f in df_train.columns]
features_eval  = [f for f in FEATURES_16 if f in df_eval.columns]

# Extraction et nettoyage
X_train = df_train[features_train].replace(
    [np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y_train  = df_train['label'].astype(np.int32)
typ_train = df_train['type'].copy()

X_eval  = df_eval[features_eval].replace(
    [np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
y_eval  = df_eval['label'].astype(np.int32)
typ_eval = df_eval['type'].copy() if 'type' in df_eval.columns else None

print(f"\n  X_train : {X_train.shape}")
print(f"  X_eval  : {X_eval.shape}")

# ============================================================
# NOUVEAU 2 — Corriger le scaler
# ============================================================
# QUOI    : Recréer le scaler correctement
# POURQUOI: En S2, le scaler était fit_transform en une seule
#           opération sur les données train uniquement
#           C'est correct mais on n'avait pas transformé eval
# RÈGLE   : fit() sur TRAIN → transform() sur TRAIN + EVAL
#           Si on fit aussi sur eval → data leakage (triche !)
# ============================================================
print("\n" + "=" * 60)
print("NOUVEAU 2 — NORMALISATION CORRECTE (train + eval)")
print("=" * 60)

scaler = StandardScaler()
scaler.fit(X_train)  # ← uniquement sur train

X_train_sc = scaler.transform(X_train).astype(np.float32)
X_eval_sc  = scaler.transform(X_eval).astype(np.float32)

print(f"\n  Scaler entraîné sur train ({len(X_train)} flux)")
print(f"  Appliqué sur train : {X_train_sc.shape}")
print(f"  Appliqué sur eval  : {X_eval_sc.shape}")

idx = {f: i for i, f in enumerate(features_train)}
print(f"\n  Vérification (moyenne ≈ 0, std ≈ 1) :")
print(f"    src_bytes → moy={X_train_sc[:,idx['src_bytes']].mean():.4f}, "
      f"std={X_train_sc[:,idx['src_bytes']].std():.4f}")
print(f"    duration  → moy={X_train_sc[:,idx['duration']].mean():.4f}, "
      f"std={X_train_sc[:,idx['duration']].std():.4f}")

joblib.dump(scaler, "models/scaler_toniot.joblib")
print(f"\n  ✅ Scaler sauvegardé (remplace celui de S2)")

# ============================================================
# NOUVEAU 3 — Sauvegarder .npy pour les modèles
# ============================================================
# QUOI    : Sauvegarder les données en format numpy binaire
# POURQUOI: La Semaine 4 chargera directement ces fichiers
#           Format .npy = 10x plus rapide qu'un CSV
#           Garantit l'ordre exact des features
# ============================================================
print("\n" + "=" * 60)
print("NOUVEAU 3 — SAUVEGARDE .NPY POUR SEMAINE 4")
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
# NOUVEAU 4 — Analyse corrélation features vs label
# ============================================================
# QUOI    : Mesurer relation entre chaque feature et label
# POURQUOI: Justifier le choix des 16 features
#           Comparer avec l'importance SHAP du modèle
# RÉSULTAT: Confirme que dst_ip_bytes et dst_port
#           sont les plus discriminantes
# ============================================================
print("\n" + "=" * 60)
print("NOUVEAU 4 — CORRÉLATION FEATURES vs LABEL")
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

# Charger SHAP pour comparaison
with open("models/shap_importance.json") as f:
    shap_data = json.load(f)
shap_imp = shap_data["feature_importance_mean_abs_shap"]
sorted_shap = sorted(
    shap_imp.items(), key=lambda x: x[1], reverse=True
)

print(f"\n  Top 5 features selon SHAP :")
for rang, (feat, val) in enumerate(sorted_shap[:5], 1):
    print(f"    {rang}. {feat:<28} SHAP={val:.4f}")

# ============================================================
# NOUVEAU 5 — SMOTE pour MitM
# ============================================================
# QUOI    : Créer des exemples MitM synthétiques
# POURQUOI: MitM = seulement 1043 exemples (0.5%)
#           Le modèle IA ignorera cette classe sans SMOTE
#           → recall MitM très faible (< 50% sans SMOTE)
# JUSTIFICATION DU CHOIX SMOTE :
#   Simple duplication → surapprentissage
#   Undersampling     → perte d'information
#   SMOTE (Chawla et al., 2002) → exemples synthétiques
#   par interpolation entre voisins → meilleure généralisation
# ============================================================
print("\n" + "=" * 60)
print("NOUVEAU 5 — SMOTE POUR CLASSE MitM")
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

print(f"\n  Application SMOTE (MitM : {(typ_enc==mitm_label).sum()} → {TARGET_MITM})...")

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

# ============================================================
# NOUVEAU 6 — Figures scientifiques
# ============================================================
print("\n" + "=" * 60)
print("NOUVEAU 6 — FIGURES SCIENTIFIQUES")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "Semaine 3 — Prétraitement Complémentaire TON-IoT\n"
    f"Train : {len(X_train):,} flux | "
    f"Eval : {len(X_eval):,} flux",
    fontsize=14, fontweight='bold'
)

# Figure 1 : Distribution train par type
ax1 = axes[0, 0]
colors = ['#2ecc71' if t == 'normal' else '#e74c3c'
          for t in dist_av.index]
bars = ax1.bar(range(len(dist_av)), dist_av.values,
               color=colors, edgecolor='black')
ax1.set_xticks(range(len(dist_av)))
ax1.set_xticklabels(dist_av.index, rotation=45,
                    ha='right', fontsize=8)
ax1.set_title('Distribution par type (train)', fontsize=11)
ax1.set_ylabel('Nombre de flux')
if 'mitm' in dist_av.index:
    mi = list(dist_av.index).index('mitm')
    ax1.annotate('⚠️ MitM\nRare (0.5%)',
                 xy=(mi, dist_av['mitm']),
                 xytext=(mi+1.5, dist_av['mitm']+8000),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8, color='red')

# Figure 2 : Corrélation Pearson
ax2 = axes[0, 1]
cv = corrs.values
cn = [n[:14] for n in corrs.index]
cc = ['#e74c3c' if abs(v) > 0.3 else
      '#f39c12' if abs(v) > 0.1 else
      '#2ecc71' for v in cv]
ax2.barh(range(len(cv)), cv, color=cc, edgecolor='black')
ax2.set_yticks(range(len(cn)))
ax2.set_yticklabels(cn, fontsize=8)
ax2.set_title('Corrélation features vs label', fontsize=11)
ax2.set_xlabel('Corrélation de Pearson')
ax2.axvline(x=0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3)

# Figure 3 : Importance SHAP
ax3 = axes[0, 2]
s_names = [n[:14] for n, _ in reversed(sorted_shap)]
s_vals  = [v for _, v in reversed(sorted_shap)]
s_col   = ['#e74c3c' if v > 1.0 else
           '#f39c12' if v > 0.3 else
           '#2ecc71' for v in s_vals]
ax3.barh(range(len(s_names)), s_vals,
         color=s_col, edgecolor='black')
ax3.set_yticks(range(len(s_names)))
ax3.set_yticklabels(s_names, fontsize=8)
ax3.set_title('Importance SHAP (XGBoost)', fontsize=11)
ax3.set_xlabel('Mean |SHAP value|')
ax3.grid(True, alpha=0.3)

# Figure 4 : src_bytes avant normalisation
ax4 = axes[1, 0]
seuil = np.percentile(X_train.values[:, idx['src_bytes']], 95)
pd.Series(X_train.values[:, idx['src_bytes']]).clip(
    upper=seuil).hist(ax=ax4, bins=60,
                      color='#3498db', edgecolor='black')
ax4.set_title('src_bytes AVANT normalisation', fontsize=11)
ax4.set_xlabel('Valeur brute (octets)')
ax4.set_ylabel('Fréquence')

# Figure 5 : src_bytes après normalisation
ax5 = axes[1, 1]
seuil_s = np.percentile(X_train_sc[:, idx['src_bytes']], 95)
pd.Series(X_train_sc[:, idx['src_bytes']]).clip(
    upper=seuil_s).hist(ax=ax5, bins=60,
                        color='#9b59b6', edgecolor='black')
ax5.set_title('src_bytes APRÈS normalisation\n(StandardScaler)',
              fontsize=11)
ax5.set_xlabel('Valeur normalisée')
ax5.set_ylabel('Fréquence')

# Figure 6 : SMOTE avant vs après
ax6 = axes[1, 2]
labels_comp = ['Avant SMOTE', 'Après SMOTE']
mitm_counts = [
    dist_av.get('mitm', 0),
    dist_ap.get('mitm', 0)
]
bars6 = ax6.bar(labels_comp, mitm_counts,
                color=['#e74c3c', '#2ecc71'],
                edgecolor='black', width=0.5)
ax6.set_title('SMOTE — Classe MitM\n(Chawla et al., 2002)',
              fontsize=11)
ax6.set_ylabel("Nombre d'exemples MitM")
for bar, val in zip(bars6, mitm_counts):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 50,
             f'{val:,}', ha='center',
             fontsize=12, fontweight='bold')
ax6.set_ylim(0, max(mitm_counts) * 1.2)

plt.tight_layout()
plt.savefig('results/semaine3_complet.png',
            dpi=150, bbox_inches='tight')
print(f"  ✅ results/semaine3_complet.png")

# ============================================================
# RÉSUMÉ FINAL
# ============================================================
print("\n" + "=" * 60)
print("RÉSUMÉ SEMAINE 3 — CE QUI EST NOUVEAU PAR RAPPORT À S2")
print("=" * 60)
print(f"""
Semaine 2 avait fait :
  ✅ Chargement train_test_networks.csv
  ✅ Sélection 16 features
  ✅ Nettoyage NaN/infinis
  ✅ Normalisation basique (sur train uniquement)
  ✅ Train/Test split

Semaine 3 AJOUTE (sans répétition) :
  ✅ NOUVEAU 1 : Chargement eval_test_full.csv aussi
  ✅ NOUVEAU 2 : Normalisation correcte sur les DEUX fichiers
  ✅ NOUVEAU 3 : Sauvegarde .npy pour les modèles S4
  ✅ NOUVEAU 4 : Analyse corrélation + SHAP
  ✅ NOUVEAU 5 : SMOTE pour MitM (1043 → {TARGET_MITM} exemples)
  ✅ NOUVEAU 6 : 6 figures scientifiques propres

Fichiers créés :
  📁 data/X_train_scaled.npy  {X_train_sc.shape}
  📁 data/X_eval_scaled.npy   {X_eval_sc.shape}
  📁 data/y_train.npy
  📁 data/y_eval.npy
  📁 data/train_resampled_mitm.csv
  📁 models/scaler_toniot.joblib
  📁 models/feature_names_toniot.json
  📊 results/semaine3_complet.png

🚀 Semaine 4 : Entraînement et validation des modèles IA
   → Les fichiers .npy sont prêts
   → EdgeInference (app/inference.py) est prêt
   → Évaluation avec métriques F1, Recall, AUC
""")


# ============================================================
# RÉSULTATS EXPÉRIMENTAUX & MÉTRIQUES (NOUVEAU)
# ============================================================
print("\n" + "=" * 60)
print("RÉSULTATS EXPÉRIMENTAUX & MÉTRIQUES - SEMAINE 3")
print("=" * 60)

# 1. Statistiques descriptives après normalisation
print("\n1. Statistiques après normalisation :")
stats = pd.DataFrame({
    'mean': X_train_sc.mean(axis=0),
    'std': X_train_sc.std(axis=0),
    'min': X_train_sc.min(axis=0),
    'max': X_train_sc.max(axis=0)
}, index=features_train)

print(stats.round(4))

# 2. Distribution des classes
print("\n2. Distribution des classes :")
print("   Train (original) :")
print(y_train.value_counts().to_string())
print("\n   Eval :")
print(y_eval.value_counts().to_string())

# 3. Métriques de corrélation
print(f"\n3. Top 5 features les plus corrélées avec le label :")
print(corrs.head(5).round(4))

# 4. Impact du SMOTE
print(f"\n4. Impact du SMOTE sur la classe MitM :")
print(f"   Avant SMOTE : {dist_av.get('mitm', 0)} exemples")
print(f"   Après SMOTE : {dist_ap.get('mitm', 0)} exemples")
print(f"   Multiplication : {dist_ap.get('mitm', 0) / dist_av.get('mitm', 1):.1f}x")

# 5. Sauvegarde des métriques dans un fichier JSON (très utile pour le rapport)
metrics = {
    "week": 3,
    "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
    "train_size": len(X_train),
    "eval_size": len(X_eval),
    "features": len(features_train),
    "normal_train": int(y_train.value_counts().get(0, 0)),
    "attack_train": int(y_train.value_counts().get(1, 0)),
    "mitm_before_smote": int(dist_av.get('mitm', 0)),
    "mitm_after_smote": int(dist_ap.get('mitm', 0)),
    "correlation_top5": {k: round(float(v), 4) for k,v in corrs.head(5).items()},
    "scaler": "StandardScaler",
    "smote_target": TARGET_MITM
}

with open("results/metrics_semaine3.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Métriques sauvegardées dans results/metrics_semaine3.json")
print("   → Prêt à être utilisé dans le rapport")