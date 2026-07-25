

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
print("Premières colonnes :", df.columns.tolist()[:10])  # Pour vérifier


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


# NORMALISATION avec StandardScaler

print("\n" + "=" * 60)
print("ÉTAPE 5 — NORMALISATION (StandardScaler)")
print("=" * 60)

print("\nAvant normalisation (exemples de valeurs brutes) :")
print(f"  src_bytes  : min={X['src_bytes'].min():.0f}, max={X['src_bytes'].max():.0f}")
print(f"  duration   : min={X['duration'].min():.4f}, max={X['duration'].max():.2f}")
print(f"  src_pkts   : min={X['src_pkts'].min():.0f}, max={X['src_pkts'].max():.0f}")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

print(f"\nAprès normalisation (les mêmes features) :")
print(f"  src_bytes  : min={X_scaled['src_bytes'].min():.3f}, max={X_scaled['src_bytes'].max():.3f}")
print(f"  duration   : min={X_scaled['duration'].min():.3f}, max={X_scaled['duration'].max():.3f}")
print(f"  src_pkts   : min={X_scaled['src_pkts'].min():.3f}, max={X_scaled['src_pkts'].max():.3f}")
print(f"\n Toutes les features sont maintenant sur la même échelle !")


os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler_toniot.joblib")
print(f" Scaler sauvegardé dans models/scaler_toniot.joblib")


#  SÉPARATION TRAIN / TEST

print("\n" + "=" * 60)
print("ÉTAPE 6 — SÉPARATION TRAIN / TEST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      
    random_state=42,    
    stratify=y           
)

print(f"\nDonnées d'entraînement (Train) : {len(X_train)} lignes")
print(f"  → Normal   : {(y_train==0).sum()} exemples")
print(f"  → Attaque  : {(y_train==1).sum()} exemples")
print(f"\nDonnées de test (Test)         : {len(X_test)} lignes")
print(f"  → Normal   : {(y_test==0).sum()} exemples")
print(f"  → Attaque  : {(y_test==1).sum()} exemples")

print(f"\n Explication :")
print(f"   Le modèle apprendra sur {len(X_train)} exemples")
print(f"   On le testera sur {len(X_test)} exemples qu'il n'a jamais vus")


#  VISUALISATIONS

print("\n" + "=" * 60)
print("ÉTAPE 7 — CRÉATION DES GRAPHIQUES")
print("=" * 60)

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

# Graphique 3 : Distribution de src_bytes (avant normalisation)
ax3 = axes[1, 0]
# On prend seulement les valeurs < 95e percentile pour éviter les outliers
threshold = X['src_bytes'].quantile(0.95)
X['src_bytes'].clip(upper=threshold).hist(ax=ax3, bins=50, color='#3498db', edgecolor='black')
ax3.set_title('Distribution de src_bytes (95e percentile)')
ax3.set_xlabel('src_bytes')
ax3.set_ylabel('Fréquence')

# Graphique 4 : Distribution de src_bytes après normalisation
ax4 = axes[1, 1]
threshold_scaled = X_scaled['src_bytes'].quantile(0.95)
X_scaled['src_bytes'].clip(upper=threshold_scaled).hist(ax=ax4, bins=50, color='#9b59b6', edgecolor='black')
ax4.set_title('src_bytes APRÈS normalisation')
ax4.set_xlabel('src_bytes normalisé')
ax4.set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('results/analyse_semaine2.png', dpi=150)
print(" Graphiques sauvegardés dans results/analyse_semaine2.png")

