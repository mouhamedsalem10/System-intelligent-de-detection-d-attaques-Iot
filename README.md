#  Détection et Prévention des Attaques IoT par Intelligence Artificielle

> Système intelligent de détection d'attaques réseau IoT en temps réel,
> basé sur une architecture hybride **AE + LSTM + CNN 1D + XGBoost**,
> déployé en Edge Computing via **MQTT/TLS** et **Docker**.

---

##  Résultats

| Scénario | F1-Score | Recall | Accuracy |
|----------|----------|--------|----------|
| IID (conditions normales) | **0.9136** | 94.73% | 86.18% |
| OOD (nouveaux réseaux) | 0.7756 | 95.15% | 72.47% |
| Bruité ±20% | 0.8820 | 93.40% | 80.74% |

**Latence médiane :** 12.8 ms | **Docker déployé** | **SHAP intégré**

---

##  Architecture du système

```
Appareils IoT (caméras, capteurs...)
        │ trafic réseau
        ▼
     ZEEK 6.0  (analyse passive)
        │ conn.log + http.log + dns.log
        ▼
  16 features extraites
        │ JSON via MQTT/TLS (port 8883)
        ▼
   EDGE DETECTOR (Docker)
   ┌──────────────────────────────────┐
   │  StandardScaler (normalisation)  │
   │           │                      │
   │    ┌──────┼──────┐               │
   │    ▼      ▼      ▼               │
   │   AE    LSTM   CNN 1D            │
   │    │      │      │               │
   │    └──────┴──────┘               │
   │      Fusion (64 dimensions)      │
   │              │                   │
   │          XGBoost                 │
   │              │                   │
   │     p ∈ [0,1] → LOW/MEDIUM/HIGH  │
   └──────────────────────────────────┘
        │
   iot/alerts + iot/actions + iot/explanations (SHAP)
```

---

##  Modèles IA utilisés

| Modèle | Rôle | Paramètres | Performance |
|--------|------|-----------|-------------|
| **AutoEncoder** | Détection d'anomalies statistiques | 7 392 | Ratio MSE attaque/normal = 9.2× |
| **LSTM** | Dépendances temporelles entre features | 6 817 | Accuracy 91.33% |
| **CNN 1D** | Patterns locaux entre features | 7 393 | Accuracy 96.87% |
| **XGBoost** | Classification finale (64 dims) | 138 arbres | AUC 0.9999 |

---

##  Structure du projet

```
IoT_Detection_Project/
│
├── app/                          # Pipeline de détection
│   ├── __init__.py
│   ├── inference.py              # Pipeline AE+LSTM+CNN+XGBoost
│   ├── main.py                   # Orchestrateur MQTT
│   ├── mqtt_client.py            # Client MQTT/TLS
│   ├── prevention.py             # Logique LOW/MEDIUM/HIGH
│   └── utils.py                  # Utilitaires
│
├── config/
│   ├── config.yaml               # Configuration MQTT et seuils
│   ├── mosquitto.conf            # Configuration broker
│   └── certs/                    # Certificats TLS
│
├── data/
│   ├── train_test_networks.csv   # Dataset entraînement (211 043 flux)
│   ├── eval_test_full.csv        # Dataset évaluation (36 595 flux)
│   ├── noisy20.csv               # Dataset bruité ±20%
│   └── eval_ood_dst_ip_balanced.csv  # Dataset OOD
│
├── docker/
│   └── Dockerfile                # Image edge-detector
│
├── models/
│   ├── encoder_ae_toniot.tflite  # AutoEncoder (5.7 KB)
│   ├── encoder_lstm_toniot.tflite# LSTM (12.5 KB)
│   ├── encoder_cnn_toniot.tflite # CNN 1D (3.4 KB)
│   ├── xgb_toniot.json           # XGBoost (712 KB)
│   ├── scaler_toniot.joblib      # StandardScaler
│   ├── feature_names_toniot.json # Ordre des 16 features
│   ├── meta.json                 # Seuils LOW=0.30 / HIGH=0.80
│   └── shap_importance.json      # Importance SHAP des features
│
├── scripts/
│   ├── semaine3_pretraitement.py # Prétraitement S3
│   ├── semaine4_xgboost_fusion.py# Fusion + XGBoost S4
│   ├── semaine6_script1_iid.py   # Évaluation IID
│   ├── semaine6_script2_ood.py   # Évaluation OOD
│   ├── semaine6_script3_noisy.py # Évaluation bruité
│   ├── semaine6_script4_comparaison.py # Comparaison
│   ├── test_mqtt_dataset.py      # Test MQTT temps réel
│   └── test_latence_s5.py        # Test de latence
│
├── results/                      # Figures et métriques
├── zeek_out/                     # Logs Zeek (conn.log, http.log)
├── docker-compose.yml            # Déploiement 2 conteneurs
├── requirements.txt              # Dépendances Python
└── README.md
```

---

##  Installation et lancement

### Prérequis

- Python 3.11+
- Docker Desktop
- Google Colab (pour l'entraînement des modèles)

### 1. Cloner le projet

```bash
git clone https://github.com/votre-username/IoT_Detection_Project.git
cd IoT_Detection_Project
```

### 2. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

**requirements.txt :**
```
numpy
pandas
scikit-learn==1.6.1
xgboost==3.2.0
tensorflow==2.20.0
paho-mqtt==1.6.1
pyyaml==6.0
joblib==1.3.2
imbalanced-learn
matplotlib
shap
```

### 3. Lancer via Docker (recommandé)

```bash
# Lancer les 2 conteneurs (MQTT + Edge Detector)
docker-compose up --build
```

Vous devriez voir :
```
mqtt           | mosquitto version 2.1.2 running
edge-detector  | ✅ EdgeInference chargé — Features : 16
edge-detector  | 🚀 Edge Detector démarré
edge-detector  |    En attente de flux réseau...
edge-detector  | ✅ MQTT connecté à mqtt:1883
```

### 4. Envoyer des flux de test

Dans un deuxième terminal :

```bash
python scripts/test_mqtt_dataset.py
```

---

##  Topics MQTT

| Topic | Direction | Contenu |
|-------|-----------|---------|
| `iot/flows` | IoT → Edge | 16 features réseau en JSON |
| `iot/alerts` | Edge → Admin | Score, niveau, latence |
| `iot/actions` | Edge → Admin | Action recommandée |
| `iot/explanations` | Edge → Admin | Top 5 features SHAP |

### Exemple de message entrant (`iot/flows`)

```json
{
  "device_id": "camera01",
  "ts_ms": 1784415457000,
  "features": {
    "src_port": 52341,
    "dst_port": 80,
    "duration": 0.5,
    "src_bytes": 1200,
    "dst_bytes": 4500,
    "missed_bytes": 0,
    "src_pkts": 8,
    "src_ip_bytes": 1520,
    "dst_pkts": 6,
    "dst_ip_bytes": 4820,
    "dns_qclass": 0,
    "dns_qtype": 0,
    "dns_rcode": 0,
    "http_request_body_len": 0,
    "http_response_body_len": 0,
    "http_status_code": 200
  }
}
```

### Exemple d'alerte sortante (`iot/alerts`)

```json
{
  "ts_ms": 1784415457143,
  "device_id": "camera01",
  "prob_attack": 0.9987,
  "level": "HIGH",
  "action": "ATTAQUE — Isolation recommandée",
  "latency_ms": 12.8
}
```

---

## ⚙️ Logique de décision

| Score p | Niveau | Action |
|---------|--------|--------|
| p < 0.30 | 🟢 **LOW** | Trafic normal — aucune intervention |
| 0.30 ≤ p < 0.80 | 🟡 **MEDIUM** | Surveillance renforcée |
| p ≥ 0.80 | 🔴 **HIGH** | ATTAQUE — Isolation recommandée |

---

## 🔬 Les 16 features réseau

| # | Feature | Source | SHAP |
|---|---------|--------|------|
| 1 | `dst_ip_bytes` | conn.log | **1.352** ← Plus importante |
| 2 | `dst_port` | conn.log | **1.002** |
| 3 | `src_ip_bytes` | conn.log | 0.746 |
| 4 | `duration` | conn.log | 0.687 |
| 5 | `src_pkts` | conn.log | 0.664 |
| 6 | `dns_qclass` | dns.log | 0.630 |
| 7 | `dst_pkts` | conn.log | 0.492 |
| 8 | `dst_bytes` | conn.log | 0.418 |
| 9 | `dns_qtype` | dns.log | 0.270 |
| 10 | `src_bytes` | conn.log | 0.230 |
| 11 | `src_port` | conn.log | 0.127 |
| 12 | `missed_bytes` | conn.log | 0.037 |
| 13 | `dns_rcode` | dns.log | 0.018 |
| 14 | `http_resp_body_len` | http.log | 0.001 |
| 15 | `http_status_code` | http.log | 0.001 |
| 16 | `http_req_body_len` | http.log | 0.000 |

---

## 📈 Résultats détaillés par type d'attaque (Scénario IID)

| Type | N flux | Recall | F1 | Résultat |
|------|--------|--------|-----|---------|
| Backdoor | 3 742 | 99.81% | 0.9991 | ✅ Excellent |
| Ransomware | 2 748 | **100.00%** | **1.0000** | ✅ Parfait |
| Injection | 3 993 | 99.27% | 0.9964 | ✅ Excellent |
| Password | 3 969 | 96.65% | 0.9830 | ✅ Très bon |
| DoS | 2 934 | 97.58% | 0.9878 | ✅ Très bon |
| Scanning | 3 723 | 98.28% | 0.9913 | ✅ Très bon |
| XSS | 2 899 | 98.45% | 0.9922 | ✅ Très bon |
| DDoS | 3 997 | 73.15% | 0.8450 | ⚠️ Moyen |
| MitM | 206 | 68.45% | 0.8127 | ⚠️ Faible |

---

##  Docker

Le projet utilise 2 conteneurs Docker :

```yaml
services:
  mqtt:           # Eclipse Mosquitto 2 — Broker MQTT
  edge-detector:  # Pipeline IA Python 3.11
```

```bash
# Démarrer
docker-compose up --build

# Arrêter
docker-compose down

# Voir les logs
docker-compose logs -f edge-detector
```

---

##  Dataset

**TON-IoT** (Telemetry of Things) — UNSW Australia
- Moustafa, N. & Slay, J. (2021). IEEE Access, 9, 23862–23879.
- 211 043 flux réseau d'entraînement
- 9 types d'attaques : DDoS, DoS, Scanning, MitM,
  Ransomware, Backdoor, Injection, XSS, Password

---


