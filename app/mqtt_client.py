"""
app/mqtt_client.py — Client MQTT sécurisé

"""

import ssl
import paho.mqtt.client as mqtt


class MqttService:
 

    def __init__(
        self,
        host        : str,
        port        : int,
        client_id   : str,
        on_message_cb,
        username    : str  = None,
        password    : str  = None,
        tls         : bool = False,
        ca_cert     : str  = None,
        insecure    : bool = False,
    ):
       
        self.host = host
        self.port = port

        # Créer le client MQTT
        self.client = mqtt.Client(
            client_id=client_id,
            protocol=mqtt.MQTTv311
        )

        # Authentification
        if username and password:
            self.client.username_pw_set(username, password)

        # TLS (chiffrement)
        # POURQUOI : Sans TLS, les données IoT circulent
        #            en clair sur le réseau → risque
        if tls and ca_cert:
            self.client.tls_set(
                ca_certs=ca_cert,
                tls_version=ssl.PROTOCOL_TLS_CLIENT
            )
            if insecure:
                self.client.tls_insecure_set(True)

        # Callbacks (fonctions appelées automatiquement)
        self.client.on_connect    = self._on_connect
        self.client.on_message    = on_message_cb
        self.client.on_disconnect = self._on_disconnect

        self._subscriptions = []

    def _on_connect(self, client, userdata, flags, rc):
        
        if rc == 0:
            print(f"✅ MQTT connecté à {self.host}:{self.port}")
            # Réabonnement automatique après reconnexion
            for topic in self._subscriptions:
                client.subscribe(topic)
                print(f"   Abonné à : {topic}")
        else:
            codes = {
                1: "Version MQTT incorrecte",
                2: "Client ID invalide",
                3: "Serveur indisponible",
                4: "Identifiants incorrects",
                5: "Non autorisé"
            }
            print(f"❌ MQTT erreur {rc}: "
                  f"{codes.get(rc, 'Erreur inconnue')}")

    def _on_disconnect(self, client, userdata, rc):
        
        if rc != 0:
            print(f"⚠️ MQTT déconnecté (rc={rc})"
                  f" — reconnexion automatique...")

    def subscribe(self, topic: str):
        
        self._subscriptions.append(topic)

    def publish(self, topic: str, payload: str, qos: int = 0):
        
        self.client.publish(topic, payload, qos=qos)

    def connect(self):
       
        self.client.connect(
            self.host, self.port, keepalive=60
        )

    def loop_forever(self):
        
        self.client.loop_forever()

    def loop_start(self):
       
        self.client.loop_start()

    def loop_stop(self):
        
        self.client.loop_stop()