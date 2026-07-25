

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# CHARGER LE DATASET

print("=" * 60)
print("CHARGEMENT DU DATASET")
print("=" * 60)

df = pd.read_csv("data/eval_test_full.csv")

print(f"\nNombre de lignes    : {len(df)}")
print(f"Nombre de colonnes  : {len(df.columns)}")
print(f"\nColonnes disponibles :")
for col in df.columns:
    print(f"  - {col}")


# DISTRIBUTION DES CLASSES
print("\n" + "=" * 60)
print("DISTRIBUTION DES TYPES DE TRAFIC")
print("=" * 60)

if 'label' in df.columns:
    label_col = 'label'
elif 'type' in df.columns:
    label_col = 'type'
else:
    
    label_col = df.columns[-1]

counts = df[label_col].value_counts()
print(f"\nColonne de labels : '{label_col}'")
print("\nRépartition :")
for label, count in counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<15} : {count:6d} flux ({pct:5.1f}%) {bar}")


# LES 16 FEATURES RÉSEAU

print("\n" + "=" * 60)
print("LES 16 FEATURES RÉSEAU")
print("=" * 60)

features_16 = [
    'src_port', 'dst_port', 'duration',
    'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qclass', 'dns_qtype', 'dns_rcode',
    'http_request_body_len', 'http_response_body_len', 'http_status_code'
]

features_presentes = [f for f in features_16 if f in df.columns]
features_absentes  = [f for f in features_16 if f not in df.columns]

print(f"\nFeatures présentes ({len(features_presentes)}/16) :")
for f in features_presentes:
    print(f"  ✓ {f}")

if features_absentes:
    print(f"\nFeatures absentes ({len(features_absentes)}) :")
    for f in features_absentes:
        print(f"  ✗ {f}")


# STATISTIQUES DESCRIPTIVES

print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES DES FEATURES")
print("=" * 60)

if features_presentes:
    print(df[features_presentes].describe().to_string())

# VALEURS MANQUANTES

print("\n" + "=" * 60)
print("VALEURS MANQUANTES")
print("=" * 60)

missing = df[features_presentes].isnull().sum()
print("\nNombre de valeurs manquantes par feature :")
for feat, count in missing.items():
    if count > 0:
        print(f"  {feat:<35} : {count} manquantes")
    else:
        print(f"  {feat:<35} : OK")


# SAUVEGARDER UN GRAPHIQUE

plt.figure(figsize=(12, 6))
counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribution des types de trafic — TON-IoT', fontsize=14)
plt.xlabel('Type de trafic')
plt.ylabel('Nombre de flux')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/distribution_classes.png', dpi=150)
print("\n Graphique sauvegardé dans results/distribution_classes.png")

print("\n Exploration terminée !")