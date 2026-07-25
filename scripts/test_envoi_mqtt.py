"""
Test d'envoi de flux réseau via MQTT

"""

import json
import time
import paho.mqtt.client as mqtt

# ============================================================
# CONFIGURATION
# ============================================================
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC_IN    = "iot/flows"
TOPIC_ALERTS= "iot/alerts"

# ============================================================
# FLUX DE TEST
# ============================================================
flux_tests = [
    {
        "nom"      : "Trafic NORMAL",
        "device_id": "camera01",
        "features" : {
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
    },
    {
        "nom"      : "Attaque DDoS simulée",
        "device_id": "sensor01",
        "features" : {
            "src_port": 12345,    "dst_port": 80,
            "duration": 0.001,    "src_bytes": 5000000,
            "dst_bytes": 100,     "missed_bytes": 0,
            "src_pkts": 10000,    "src_ip_bytes": 5200000,
            "dst_pkts": 50,       "dst_ip_bytes": 6500000,
            "dns_qclass": 0,      "dns_qtype": 0,
            "dns_rcode": 0,
            "http_request_body_len": 0,
            "http_response_body_len": 0,
            "http_status_code": 0
        }
    },
    {
        "nom"      : "Attaque Scanning simulée",
        "device_id": "thermostat01",
        "features" : {
            "src_port": 54321,  "dst_port": 22,
            "duration": 0.002,  "src_bytes": 100,
            "dst_bytes": 0,     "missed_bytes": 50,
            "src_pkts": 500,    "src_ip_bytes": 200,
            "dst_pkts": 0,      "dst_ip_bytes": 150000,
            "dns_qclass": 1,    "dns_qtype": 1,
            "dns_rcode": 3,
            "http_request_body_len": 0,
            "http_response_body_len": 0,
            "http_status_code": 0
        }
    }
]

# ============================================================
# CALLBACK — Réception des alertes
# ============================================================
alertes_recues = []

def on_message(client, userdata, msg):
    """Appelé quand une alerte arrive"""
    try:
        alerte = json.loads(msg.payload.decode())
        alertes_recues.append(alerte)
        emoji = {
            "LOW"   : "🟢",
            "MEDIUM": "🟡",
            "HIGH"  : "🔴"
        }.get(alerte.get("level", ""), "⚪")
        print(f"\n  {emoji} ALERTE REÇUE :")
        print(f"     Device    : {alerte.get('device_id')}")
        print(f"     Score     : {alerte.get('prob_attack', 0):.4f}")
        print(f"     Niveau    : {alerte.get('level')}")
        print(f"     Action    : {alerte.get('action')}")
        print(f"     Latence   : {alerte.get('latency_ms', 0):.1f} ms")
    except Exception as e:
        print(f"  Erreur lecture alerte : {e}")

# ============================================================
# CONNEXION MQTT
# ============================================================
client = mqtt.Client(client_id="test-publisher")
client.on_message = on_message

print("Connexion au broker MQTT...")
client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
client.subscribe(TOPIC_ALERTS)
client.loop_start()
time.sleep(1)
print("✅ Connecté !\n")

# ============================================================
# ENVOI DES FLUX DE TEST
# ============================================================
print("=" * 50)
print("ENVOI DES FLUX DE TEST")
print("=" * 50)

for flux in flux_tests:
    print(f"\n📤 Envoi : {flux['nom']}")
    print(f"   Device  : {flux['device_id']}")

    # Construire le message
    message = {
        "device_id": flux["device_id"],
        "ts_ms"    : int(time.time() * 1000),
        "features" : flux["features"]
    }

    # Publier sur iot/flows
    client.publish(
        TOPIC_IN,
        json.dumps(message),
        qos=0
    )

    # Attendre la réponse (2 secondes)
    time.sleep(2)

# ============================================================
# RÉSUMÉ
# ============================================================
print("\n" + "=" * 50)
print("RÉSUMÉ DES ALERTES REÇUES")
print("=" * 50)

if alertes_recues:
    for i, a in enumerate(alertes_recues, 1):
        emoji = {
            "LOW"   : "🟢",
            "MEDIUM": "🟡",
            "HIGH"  : "🔴"
        }.get(a.get("level", ""), "⚪")
        print(f"\n  Flux {i} : {emoji} {a.get('level')}")
        print(f"    Score   : {a.get('prob_attack', 0):.4f}")
        print(f"    Latence : {a.get('latency_ms', 0):.1f} ms")
else:
    print("  Aucune alerte reçue")

client.loop_stop()
client.disconnect()
print("\n✅ Test terminé !")