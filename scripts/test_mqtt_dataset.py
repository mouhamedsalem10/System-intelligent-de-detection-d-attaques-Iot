"""
Test MQTT avec vrais flux du dataset TON-IoT

"""

import json
import time
import pandas as pd
import numpy as np
import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC_IN    = "iot/flows"
TOPIC_ALERTS= "iot/alerts"

FEATURES_16 = [
    'src_port', 'dst_port', 'duration',
    'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    'dns_qclass', 'dns_qtype', 'dns_rcode',
    'http_request_body_len', 'http_response_body_len',
    'http_status_code'
]

# Compteurs pour les statistiques
stats = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "total": 0}
latences = []

def on_message(client, userdata, msg):
    """Réception des alertes"""
    try:
        alerte = json.loads(msg.payload.decode())
        level  = alerte.get("level", "LOW")
        stats[level] = stats.get(level, 0) + 1
        stats["total"] += 1
        lat = alerte.get("latency_ms", 0)
        latences.append(lat)

        emoji = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(level,"⚪")
        print(
            f"{emoji} [{alerte.get('device_id')}] "
            f"p={alerte.get('prob_attack',0):.4f} "
            f"→ {level:<6} | {lat:.1f}ms"
        )
    except Exception as e:
        print(f"Erreur : {e}")

# Connexion MQTT
client = mqtt.Client(client_id="dataset-publisher")
client.on_message = on_message
client.connect(BROKER_HOST, BROKER_PORT)
client.subscribe(TOPIC_ALERTS)
client.loop_start()
time.sleep(1)

# Charger le dataset
print("Chargement eval_test_full.csv...")
df = pd.read_csv("data/eval_test_full.csv", sep=',')

# Prendre 10 normaux + 10 attaques + 5 MitM
normaux   = df[df['label'] == 0].head(10)
attaques  = df[df['label'] == 1].head(10)
mitm      = df[df['type'] == 'mitm'].head(5) \
            if 'type' in df.columns else pd.DataFrame()

echantillon = pd.concat([normaux, attaques, mitm])
echantillon = echantillon.sample(
    frac=1, random_state=42
).reset_index(drop=True)

total = len(echantillon)
print(f"Envoi de {total} flux réels...\n")
print("=" * 55)

for i, (_, row) in enumerate(echantillon.iterrows()):
    features = {}
    for f in FEATURES_16:
        v = row.get(f, 0.0)
        try:
            features[f] = 0.0 if pd.isna(v) else float(v)
        except Exception:
            features[f] = 0.0

    label = int(row.get('label', 0))
    typ   = str(row.get('type', 'unknown')) \
            if 'type' in row else 'unknown'

    message = {
        "device_id": f"device_{typ}_{i}",
        "ts_ms"    : int(time.time() * 1000),
        "features" : features,
        "true_label": label
    }

    client.publish(TOPIC_IN, json.dumps(message), qos=0)
    time.sleep(0.3)

# Attendre les dernières alertes
time.sleep(3)

# Résumé final
print("\n" + "=" * 55)
print("RÉSUMÉ — FLUX RÉELS TON-IoT")
print("=" * 55)
print(f"\n  Total flux envoyés : {total}")
print(f"  Alertes reçues     : {stats['total']}")
print(f"\n  🟢 LOW    : {stats.get('LOW',0)}")
print(f"  🟡 MEDIUM : {stats.get('MEDIUM',0)}")
print(f"  🔴 HIGH   : {stats.get('HIGH',0)}")

if latences:
    print(f"\n  Latence médiane : {np.median(latences):.1f} ms")
    print(f"  Latence max     : {np.max(latences):.1f} ms")
    print(f"  Latence min     : {np.min(latences):.1f} ms")

client.loop_stop()
client.disconnect()
print("\n✅ Test avec données réelles terminé !")