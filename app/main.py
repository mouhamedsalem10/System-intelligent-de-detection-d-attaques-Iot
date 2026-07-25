"""
app/main.py — Point d'entrée principal
========================================
QUOI    : Lance le système de détection IoT
POURQUOI: Connecte tous les composants :
          MQTT → Pipeline IA → Alertes
COMMENT :
  1. Charge la configuration (config.yaml)
  2. Initialise le pipeline d'inférence
  3. Se connecte au broker MQTT
  4. Pour chaque message reçu sur iot/flows :
     a. Extraire les 16 features
     b. Lancer l'inférence → score p
     c. Décider le niveau (LOW/MEDIUM/HIGH)
     d. Publier l'alerte sur iot/alerts
     e. Publier l'action sur iot/actions
"""

import json
import socket
import time
import yaml
from app.inference   import EdgeInference
from app.mqtt_client import MqttService
from app.prevention  import decide


# ============================================================
# CHARGEMENT DE LA CONFIGURATION
# ============================================================
# POURQUOI : Séparer la config du code
#            On peut changer host, port, seuils
#            sans modifier le code source
# ============================================================
print("Chargement de la configuration...")
with open("config/config.yaml", "r",
          encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mqtt_cfg  = cfg["mqtt"]
inf_cfg   = cfg["inference"]
prev_cfg  = cfg["prevention"]
xai_cfg   = cfg.get("xai", {})

# ============================================================
# CHARGEMENT DU PIPELINE IA
# ============================================================
print("Chargement du pipeline d'inférence...")
engine = EdgeInference(
    model_dir=inf_cfg.get("model_dir", "models")
)

# ============================================================
# CHARGEMENT DES VALEURS SHAP
# ============================================================
# QUOI    : Importance pré-calculée de chaque feature
# POURQUOI: SHAP en temps réel est trop lent
#           On utilise les valeurs globales pré-calculées
#           et on les publie toutes les 5 secondes
# ============================================================
import os
shap_importance = {}
shap_path = os.path.join(
    inf_cfg.get("model_dir", "models"),
    "shap_importance.json"
)
if os.path.exists(shap_path):
    with open(shap_path, encoding="utf-8") as f:
        shap_data = json.load(f)
    shap_importance = shap_data.get(
        "feature_importance_mean_abs_shap", {}
    )
    # Top K features les plus importantes
    top_k = xai_cfg.get("top_k", 5)
    top_features = dict(
        sorted(shap_importance.items(),
               key=lambda x: x[1],
               reverse=True)[:top_k]
    )
    print(f"✅ SHAP chargé — top {top_k} features : "
          f"{list(top_features.keys())}")
else:
    top_features = {}
    print("⚠️ shap_importance.json non trouvé")

# Identifiant unique de ce nœud edge
client_id = f"edge-{socket.gethostname()}"

# Dernière publication SHAP
_last_xai_ts = 0
_xai_period  = xai_cfg.get("expl_period_ms", 5000) / 1000


# ============================================================
# FONCTION DE TRAITEMENT DES MESSAGES
# ============================================================
def on_message(client, userdata, msg):
    """
    QUOI    : Traite chaque message MQTT reçu
    QUAND   : Appelé automatiquement à chaque flux reçu
    COMMENT :
      1. Parser le JSON du message
      2. Extraire les features
      3. Lancer le pipeline d'inférence
      4. Décider le niveau de risque
      5. Publier les alertes
    """
    global _last_xai_ts

    try:
        t_reception = time.time()

        # 1. Parser le message JSON
        payload   = json.loads(msg.payload.decode("utf-8"))
        device_id = payload.get("device_id", "unknown")
        features  = payload.get("features", {})
        ts_ms     = payload.get(
            "ts_ms", int(t_reception * 1000)
        )

        # 2. Pipeline d'inférence
        proba  = engine.predict_proba_from_feature_dict(
            features
        )
        result = decide(
            proba,
            low=prev_cfg.get("low",  0.30),
            high=prev_cfg.get("high", 0.80)
        )
        latency_ms = (time.time() - t_reception) * 1000

        # 3. Construire l'alerte
        alerte = {
            "ts_ms"      : int(time.time() * 1000),
            "device_id"  : device_id,
            "prob_attack": round(proba, 6),
            "level"      : result["level"],
            "action"     : result["action"],
            "latency_ms" : round(latency_ms, 2)
        }

        # 4. Publier sur iot/alerts
        mqtt_svc.publish(
            mqtt_cfg["topic_alerts"],
            json.dumps(alerte)
        )

        # 5. Publier sur iot/actions
        action_msg = {
            "device_id" : device_id,
            "level"     : result["level"],
            "action"    : result["action"]
        }
        mqtt_svc.publish(
            mqtt_cfg["topic_actions"],
            json.dumps(action_msg)
        )

        # 6. Publier SHAP toutes les 5 secondes
        now = time.time()
        if (xai_cfg.get("enabled", True)
                and top_features
                and now - _last_xai_ts >= _xai_period
                and proba >= prev_cfg.get("low", 0.30)):

            xai_msg = {
                "ts_ms"       : int(now * 1000),
                "device_id"   : device_id,
                "prob_attack" : round(proba, 6),
                "level"       : result["level"],
                "xai_type"    : "shap_global",
                "top_features": top_features,
                "period_ms"   : xai_cfg.get(
                    "expl_period_ms", 5000
                )
            }
            mqtt_svc.publish(
                mqtt_cfg["topic_explanations"],
                json.dumps(xai_msg)
            )
            _last_xai_ts = now

        # 7. Afficher dans la console
        emoji = {
            "LOW"   : "🟢",
            "MEDIUM": "🟡",
            "HIGH"  : "🔴"
        }.get(result["level"], "⚪")

        print(
            f"{emoji} [{device_id}] "
            f"p={proba:.4f} → {result['level']:<6} "
            f"| {latency_ms:.1f}ms"
        )

    except Exception as e:
        print(f"❌ Erreur traitement message : {e}")


# ============================================================
# INITIALISATION MQTT
# ============================================================
mqtt_svc = MqttService(
    host         = mqtt_cfg["host"],
    port         = mqtt_cfg["port"],
    client_id    = client_id,
    on_message_cb= on_message,
    username     = mqtt_cfg.get("username"),
    password     = mqtt_cfg.get("password"),
    tls          = mqtt_cfg.get("tls", False),
    ca_cert      = mqtt_cfg.get("ca_cert"),
    insecure     = mqtt_cfg.get("insecure", False)
)

mqtt_svc.subscribe(mqtt_cfg["topic_in"])

print(f"\n🚀 Edge Detector démarré")
print(f"   ID          : {client_id}")
print(f"   Écoute sur  : {mqtt_cfg['topic_in']}")
print(f"   Alertes sur : {mqtt_cfg['topic_alerts']}")
print(f"   Actions sur : {mqtt_cfg['topic_actions']}")
print(f"   SHAP sur    : {mqtt_cfg['topic_explanations']}")
print(f"\n   En attente de flux réseau...\n")

mqtt_svc.connect()
mqtt_svc.loop_forever()