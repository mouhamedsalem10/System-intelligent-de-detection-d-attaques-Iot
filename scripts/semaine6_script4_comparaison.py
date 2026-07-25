"""
 Comparaison des 3 scénarios

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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("SEMAINE 6 — SCRIPT 4 : COMPARAISON 3 SCÉNARIOS")
print("=" * 60)

# Charger les métriques des 3 scénarios
fichiers = {
    "IID"        : "results/s6_metriques_iid.json",
    "OOD"        : "results/s6_metriques_ood.json",
    "Bruité ±20%": "results/s6_metriques_noisy.json"
}

resultats = {}
for nom, chemin in fichiers.items():
    if os.path.exists(chemin):
        with open(chemin) as f:
            resultats[nom] = json.load(f)
        print(f"✅ {nom} chargé")
    else:
        print(f"❌ {nom} non trouvé : lance d'abord "
              f"les scripts 1, 2 et 3")

if len(resultats) < 2:
    print("Lance d'abord les scripts 1, 2 et 3 !")
    exit(1)

# Tableau comparatif
print("\n" + "=" * 60)
print("TABLEAU COMPARATIF")
print("=" * 60)
print(f"\n{'Métrique':<15}", end="")
for nom in resultats:
    print(f"  {nom:>12}", end="")
print()
print("-" * (15 + 14 * len(resultats)))

for metrique in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
    print(f"{metrique:<15}", end="")
    for nom, data in resultats.items():
        val = data.get(metrique)
        if val is not None:
            print(f"  {val:>12.4f}", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()

# Analyse de dégradation
print("\n" + "=" * 60)
print("ANALYSE DE ROBUSTESSE")
print("=" * 60)

if "IID" in resultats:
    f1_iid = resultats["IID"]["f1_score"]
    for nom, data in resultats.items():
        if nom != "IID":
            f1_sc = data["f1_score"]
            diff  = f1_iid - f1_sc
            pct   = diff / f1_iid * 100
            if diff < 0.02:
                eval_txt = "✅ Très robuste"
            elif diff < 0.05:
                eval_txt = "✅ Robuste"
            elif diff < 0.10:
                eval_txt = "⚠️  Dégradation modérée"
            else:
                eval_txt = "❌ Forte dégradation"
            print(f"\n  IID → {nom} :")
            print(f"    F1 : {f1_iid:.4f} → {f1_sc:.4f} "
                  f"(-{diff:.4f}, -{pct:.1f}%)")
            print(f"    {eval_txt}")

# Figure comparative
scenarios = list(resultats.keys())
metriques = ['accuracy', 'precision', 'recall', 'f1_score']
colors    = ['#2ecc71', '#3498db', '#e74c3c'][:len(scenarios)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Comparaison des 3 scénarios d'évaluation\n"
    "Pipeline AE+LSTM+CNN+XGBoost",
    fontsize=13, fontweight='bold'
)

# Figure 1 : F1 par scénario
ax1 = axes[0]
f1s = [resultats[s]["f1_score"] for s in scenarios]
bars = ax1.bar(range(len(scenarios)), f1s,
               color=colors, edgecolor='black')
ax1.set_xticks(range(len(scenarios)))
ax1.set_xticklabels(scenarios, rotation=10, fontsize=10)
ax1.set_title('F1-Score par scénario', fontsize=12)
ax1.set_ylabel('F1-Score')
ax1.set_ylim(0, 1.1)
ax1.axhline(y=0.95, color='orange',
            linestyle='--', label='Seuil 95%')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, f1s):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.4f}', ha='center',
             fontsize=11, fontweight='bold')

# Figure 2 : Radar des métriques
ax2 = axes[1]
x     = np.arange(len(metriques))
width = 0.25
for i, (nom, data) in enumerate(resultats.items()):
    vals = [data.get(m, 0) for m in metriques]
    ax2.bar(x + i*width, vals, width,
            label=nom, color=colors[i],
            edgecolor='black', alpha=0.8)
ax2.set_xticks(x + width)
ax2.set_xticklabels(
    ['Accuracy', 'Precision', 'Recall', 'F1'],
    fontsize=9
)
ax2.set_title('Métriques par scénario', fontsize=12)
ax2.set_ylabel('Valeur')
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/s6_comparaison.png',
            dpi=150, bbox_inches='tight')
print("\n✅ results/s6_comparaison.png")

# Rapport JSON final
rapport_final = {
    "pipeline"  : "AE+LSTM+CNN+XGBoost",
    "scenarios" : resultats,
    "conclusion": {
        "meilleur_scenario" : "IID",
        "f1_iid"    : resultats.get("IID", {}).get("f1_score"),
        "f1_ood"    : resultats.get("OOD", {}).get("f1_score"),
        "f1_noisy"  : resultats.get(
            "Bruité ±20%", {}
        ).get("f1_score"),
    }
}
with open("results/s6_rapport_final.json", "w",
          encoding="utf-8") as f:
    json.dump(rapport_final, f, indent=2,
              ensure_ascii=False)
print("✅ results/s6_rapport_final.json")
print("\n✅ SCRIPT 4 TERMINÉ — Semaine 6 complète !")