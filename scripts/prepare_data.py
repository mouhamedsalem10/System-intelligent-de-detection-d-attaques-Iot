import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


# CHARGER LES DONNÉES

print("=" * 60)
print("ÉTAPE 1 — CHARGEMENT DES DONNÉES")
print("=" * 60)


DATA_PATH = "data/train_test_networks.csv"

df = pd.read_csv(
    DATA_PATH,
    sep=';',
    low_memory=False
)

print(f" Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("Premières colonnes :", df.columns.tolist()[:10])


#  DÉFINIR LES 16 FEATURES ET LES LABELS

print("\n" + "=" * 60)
print("ÉTAPE 2 — SÉPARATION FEATURES (X) ET LABELS (y)")
print("=" * 60)

# Les 16 features que le modèle va analyser
FEATURES = [
    'src_port', 'dst_port', 'duration',
    'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qclass', 'dns_qtype', 'dns_rcode',
    'http_request_body_len', 'http_response_body_len',
    'http_status_code'
]


X = df[FEATURES].copy()

y = df['label'].copy()

type_col = df['type'].copy()

print(f"\nX (features) : {X.shape[0]} lignes × {X.shape[1]} colonnes")
print(f"y (labels)   : {y.shape[0]} valeurs")
print(f"\nValeurs possibles de y :")
print(f"  0 = trafic normal  → {(y==0).sum()} exemples")
print(f"  1 = attaque        → {(y==1).sum()} exemples")


#  NETTOYER LES DONNÉES

print("\n" + "=" * 60)
print("ÉTAPE 3 — NETTOYAGE (valeurs infinies et NaN)")
print("=" * 60)

nan_count = X.isnull().sum().sum()
inf_count = np.isinf(X.values).sum()
print(f"Avant nettoyage :")
print(f"  Valeurs NaN      : {nan_count}")
print(f"  Valeurs infinies : {inf_count}")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

nan_after = X.isnull().sum().sum()
print(f"\nAprès nettoyage :")
print(f"  Valeurs NaN restantes : {nan_after}")
print(f"   Données propres !")


# ANALYSER LA DISTRIBUTION PAR TYPE D'ATTAQUE

print("\n" + "=" * 60)
print("ÉTAPE 4 — DISTRIBUTION PAR TYPE D'ATTAQUE")
print("=" * 60)

df_temp = X.copy()
df_temp['label'] = y
df_temp['type']  = type_col

distribution = type_col.value_counts()
total = len(df_temp)

print(f"\n{'Type':<15} {'Nombre':>8} {'Pourcentage':>12} {'Barre'}")
print("-" * 55)
for attack_type, count in distribution.items():
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    label_val = df_temp[df_temp['type'] == attack_type]['label'].iloc[0]
    flag = " RARE" if pct < 2 else ""
    print(f"{attack_type:<15} {count:>8} {pct:>11.1f}%  {bar} {flag}")


# ============================================================
# ÉTAPE 5 — SÉPARATION TRAIN / TEST (AVANT normalisation)
# ============================================================
# CORRECTIF MÉTHODOLOGIQUE : le split doit se faire AVANT le
# fit du scaler, pour que le test set reste totalement "invisible"
# du pipeline d'entraînement (aucune fuite de données / data leakage).
# ============================================================

print("\n" + "=" * 60)
print("ÉTAPE 5 — SÉPARATION TRAIN / TEST (AVANT normalisation)")
print("=" * 60)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nDonnées d'entraînement (Train) : {len(X_train_raw)} lignes")
print(f"  → Normal   : {(y_train==0).sum()} exemples")
print(f"  → Attaque  : {(y_train==1).sum()} exemples")
print(f"\nDonnées de test (Test)         : {len(X_test_raw)} lignes")
print(f"  → Normal   : {(y_test==0).sum()} exemples")
print(f"  → Attaque  : {(y_test==1).sum()} exemples")

print(f"\n Explication :")
print(f"   Le modèle apprendra sur {len(X_train_raw)} exemples")
print(f"   On le testera sur {len(X_test_raw)} exemples qu'il n'a jamais vus")


# ============================================================
# ÉTAPE 6 — NORMALISATION (StandardScaler fit sur TRAIN uniquement)
# ============================================================

print("\n" + "=" * 60)
print("ÉTAPE 6 — NORMALISATION (StandardScaler fit sur TRAIN uniquement)")
print("=" * 60)

print("\nAvant normalisation (exemples de valeurs brutes, TRAIN) :")
print(f"  src_bytes  : min={X_train_raw['src_bytes'].min():.0f}, max={X_train_raw['src_bytes'].max():.0f}")
print(f"  duration   : min={X_train_raw['duration'].min():.4f}, max={X_train_raw['duration'].max():.2f}")
print(f"  src_pkts   : min={X_train_raw['src_pkts'].min():.0f}, max={X_train_raw['src_pkts'].max():.0f}")

scaler = StandardScaler()

# FIT uniquement sur X_train — le scaler ne "voit" jamais le test
scaler.fit(X_train_raw)

# TRANSFORM appliqué séparément sur train et test
X_train = pd.DataFrame(
    scaler.transform(X_train_raw),
    columns=FEATURES,
    index=X_train_raw.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test_raw),
    columns=FEATURES,
    index=X_test_raw.index
)

print(f"\nAprès normalisation (TRAIN) :")
print(f"  src_bytes  : min={X_train['src_bytes'].min():.3f}, max={X_train['src_bytes'].max():.3f}")
print(f"  duration   : min={X_train['duration'].min():.3f}, max={X_train['duration'].max():.3f}")
print(f"  src_pkts   : min={X_train['src_pkts'].min():.3f}, max={X_train['src_pkts'].max():.3f}")
print(f"\n Toutes les features sont maintenant sur la même échelle !")
print(f" Scaler fit UNIQUEMENT sur le train — aucune fuite de données")

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler_toniot.joblib")
print(f" Scaler sauvegardé dans models/scaler_toniot.joblib")


# ============================================================
# ÉTAPE 7 — SAUVEGARDE DES DONNÉES POUR LES ÉTAPES SUIVANTES (S3, S4)
# ============================================================

print("\n" + "=" * 60)
print("ÉTAPE 7 — SAUVEGARDE DES DONNÉES TRAIN/TEST")
print("=" * 60)

os.makedirs("data", exist_ok=True)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_frame(name='label').to_csv("data/y_train.csv", index=False)
y_test.to_frame(name='label').to_csv("data/y_test.csv", index=False)

# Sauvegarde du type d'attaque correspondant au test set
# (utile pour les métriques "recall par type d'attaque")
type_col.loc[X_test_raw.index].to_frame(name='type').to_csv(
    "data/type_test.csv", index=False
)

print(" data/X_train.csv")
print(" data/X_test.csv")
print(" data/y_train.csv")
print(" data/y_test.csv")
print(" data/type_test.csv")


# ============================================================
# ÉTAPE 8 — VISUALISATIONS (protégée : ne bloque jamais le pipeline)
# ============================================================

print("\n" + "=" * 60)
print("ÉTAPE 8 — CRÉATION DES GRAPHIQUES")
print("=" * 60)

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analyse du Dataset TON-IoT — Semaine 2', fontsize=16, fontweight='bold')

    # Graphique 1 : Distribution par type d'attaque
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if t == 'normal' else '#e74c3c' for t in distribution.index]
    distribution.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
    ax1.set_title('Distribution par type de trafic')
    ax1.set_xlabel('Type')
    ax1.set_ylabel('Nombre de flux')
    ax1.tick_params(axis='x', rotation=45)

    # Graphique 2 : Normal vs Attaque
    ax2 = axes[0, 1]
    label_counts = y.value_counts()
    ax2.pie(
        label_counts.values,
        labels=['Attaque (1)', 'Normal (0)'],
        colors=['#e74c3c', '#2ecc71'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('Normal vs Attaque')

    # Graphique 3 : Distribution de src_bytes (avant normalisation, TRAIN)
    ax3 = axes[1, 0]
    threshold = X_train_raw['src_bytes'].quantile(0.95)
    X_train_raw['src_bytes'].clip(upper=threshold).hist(ax=ax3, bins=50, color='#3498db', edgecolor='black')
    ax3.set_title('Distribution de src_bytes (Train, 95e percentile)')
    ax3.set_xlabel('src_bytes')
    ax3.set_ylabel('Fréquence')

    # Graphique 4 : Distribution de src_bytes après normalisation (TRAIN)
    ax4 = axes[1, 1]
    threshold_scaled = X_train['src_bytes'].quantile(0.95)
    X_train['src_bytes'].clip(upper=threshold_scaled).hist(ax=ax4, bins=50, color='#9b59b6', edgecolor='black')
    ax4.set_title('src_bytes APRÈS normalisation (Train)')
    ax4.set_xlabel('src_bytes normalisé')
    ax4.set_ylabel('Fréquence')

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig('results/analyse_semaine2.png', dpi=150)
    print(" Graphiques sauvegardés dans results/analyse_semaine2.png")

except Exception as e:
    print(f"⚠️  Erreur lors de la génération des graphiques : {e}")
    print("   (pas grave — les données sont déjà sauvegardées, on continue)")


print("\n" + "=" * 60)
print("✅ SEMAINE 2 TERMINÉE — split AVANT scaler, sans fuite de données")
print("=" * 60)