"""
Test de latence du pipeline

"""

import json
import time
import numpy as np
import pandas as pd
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

latences = []

def on_message(client, userdata, msg):
    try:
        alerte  = json.loads(msg.payload.decode())
        lat     = alerte.get("latency_ms", 0)
        latences.append(lat)
        nb = len(latences)
        if nb % 20 == 0:
            print(f"  {nb}/100 flux traités — "
                  f"médiane={np.median(latences):.1f}ms")
    except Exception:
        pass

client = mqtt.Client(client_id="latency-tester")
client.on_message = on_message
client.connect(BROKER_HOST, BROKER_PORT)
client.subscribe(TOPIC_ALERTS)
client.loop_start()
time.sleep(1)

# Charger 100 flux réels
df      = pd.read_csv("data/eval_test_full.csv", sep=',')
sample  = df.sample(100, random_state=42)

print("Test de latence — 100 flux consécutifs...")
print("=" * 50)

for _, row in sample.iterrows():
    features = {
        f: float(row.get(f, 0.0) or 0.0)
        for f in FEATURES_16
    }
    message = {
        "device_id": "latency-test",
        "ts_ms"    : int(time.time() * 1000),
        "features" : features
    }
    client.publish(TOPIC_IN, json.dumps(message), qos=0)
    time.sleep(0.1)  # 10 flux par seconde

time.sleep(3)

# Résultats
print("\n" + "=" * 50)
print("RÉSULTATS DE LATENCE")
print("=" * 50)

if latences:
    print(f"\n  Flux mesurés    : {len(latences)}")
    print(f"\n  Latence médiane : {np.median(latences):.1f} ms")
    print(f"  Latence moyenne : {np.mean(latences):.1f} ms")
    print(f"  Latence min     : {np.min(latences):.1f} ms")
    print(f"  Latence max     : {np.max(latences):.1f} ms")
    print(f"  Latence P95     : {np.percentile(latences,95):.1f} ms")
    print(f"  Latence P99     : {np.percentile(latences,99):.1f} ms")

    # Évaluation
    med = np.median(latences)
    print(f"\n  Évaluation :")
    if med < 10:
        print(f"  ✅ EXCELLENT — Médiane {med:.1f}ms < 10ms")
        print(f"     Compatible avec les systèmes IoT temps réel")
    elif med < 50:
        print(f"  ✅ BON — Médiane {med:.1f}ms < 50ms")
        print(f"     Acceptable pour la détection IoT")
    elif med < 100:
        print(f"  ⚠️  MOYEN — Médiane {med:.1f}ms < 100ms")
        print(f"     Optimisation recommandée en production")
    else:
        print(f"  ❌ LENT — Médiane {med:.1f}ms > 100ms")
        print(f"     Micro-batching conseillé")

client.loop_stop()
client.disconnect()
print("\n✅ Test de latence terminé !")