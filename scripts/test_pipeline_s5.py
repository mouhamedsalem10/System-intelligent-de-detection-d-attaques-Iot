"""
Test du pipeline complet SANS MQTT

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import pandas as pd
from app.inference import EdgeInference
from app.prevention import decide
# ... reste du code

print("=" * 55)
print("TEST PIPELINE SEMAINE 5 — MODE OFFLINE")
print("=" * 55)

# Charger le pipeline
engine = EdgeInference(model_dir="models")

# ============================================================
# TEST 1 : Flux fictif normal
# ============================================================
print("\n--- TEST 1 : Flux normal ---")
flux_normal = {
    "src_port": 52341, "dst_port": 80,
    "duration": 0.5,   "src_bytes": 1200,
    "dst_bytes": 4500, "missed_bytes": 0,
    "src_pkts": 8,     "src_ip_bytes": 1520,
    "dst_pkts": 6,     "dst_ip_bytes": 4820,
    "dns_qclass": 0,   "dns_qtype": 0,
    "dns_rcode": 0,
    "http_request_body_len": 0,
    "http_response_body_len": 0,
    "http_status_code": 200
}

t = time.time()
p = engine.predict_proba_from_feature_dict(flux_normal)
lat = (time.time() - t) * 1000
d = decide(p)
print(f"  Score   : {p:.4f}")
print(f"  Décision: {d['level']} — {d['action']}")
print(f"  Latence : {lat:.1f} ms")

# ============================================================
# TEST 2 : Flux fictif DDoS
# ============================================================
print("\n--- TEST 2 : Flux DDoS simulé ---")
flux_ddos = {
    "src_port": 12345,      "dst_port": 80,
    "duration": 0.001,      "src_bytes": 5000000,
    "dst_bytes": 100,       "missed_bytes": 0,
    "src_pkts": 10000,      "src_ip_bytes": 5200000,
    "dst_pkts": 50,         "dst_ip_bytes": 6500000,
    "dns_qclass": 0,        "dns_qtype": 0,
    "dns_rcode": 0,
    "http_request_body_len": 0,
    "http_response_body_len": 0,
    "http_status_code": 0
}

t = time.time()
p = engine.predict_proba_from_feature_dict(flux_ddos)
lat = (time.time() - t) * 1000
d = decide(p)
print(f"  Score   : {p:.4f}")
print(f"  Décision: {d['level']} — {d['action']}")
print(f"  Latence : {lat:.1f} ms")

# ============================================================
# TEST 3 : Évaluation sur 100 vrais flux
# ============================================================
print("\n--- TEST 3 : 100 flux réels ---")
df = pd.read_csv("data/eval_test_full.csv", sep=',')
sample = df.sample(100, random_state=42)

correct = 0
latences = []

for _, row in sample.iterrows():
    feats = {k: float(row.get(k, 0.0))
             for k in engine.feature_names}
    y_true = int(row['label'])

    t = time.time()
    p = engine.predict_proba_from_feature_dict(feats)
    latences.append((time.time() - t) * 1000)

    y_pred = 1 if p >= 0.5 else 0
    if y_true == y_pred:
        correct += 1

print(f"  Accuracy sur 100 flux : {correct}%")
print(f"  Latence médiane       : "
      f"{np.median(latences):.1f} ms")
print(f"  Latence max           : "
      f"{np.max(latences):.1f} ms")

print("\n✅ Pipeline opérationnel !")
print("   Prochaine étape : Docker")