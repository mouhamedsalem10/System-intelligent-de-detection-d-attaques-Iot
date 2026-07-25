"""
app/inference.py — Pipeline d'inférence hybride


"""

import json
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading



class TFLiteEncoder:

    def __init__(self, model_path: str):
        """
        QUOI    : Charge un encodeur TFLite en mémoire
        PARAMÈTRE : model_path = chemin vers le .tflite
        """
        self.model_path = str(model_path)
        self._load()

    def _load(self):
        """Charge l'interpréteur TFLite"""
        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path
        )
        self.interpreter.allocate_tensors()
        self.input_details  = \
            self.interpreter.get_input_details()
        self.output_details = \
            self.interpreter.get_output_details()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        QUOI    : Exécute l'inférence TFLite
        PARAMÈTRE : x = tableau numpy
        RETOURNE  : embedding (vecteur numpy)
        COMMENT   : Reshape automatique selon la shape
                    attendue par le modèle
        """
        try:
            expected = self.input_details[0]['shape']
            x_r = x.reshape(expected).astype(np.float32)
            self.interpreter.set_tensor(
                self.input_details[0]['index'], x_r
            )
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            return out.flatten()

        except Exception:
            # Si erreur → recharger et réessayer
            self._load()
            expected = self.input_details[0]['shape']
            x_r = x.reshape(expected).astype(np.float32)
            self.interpreter.set_tensor(
                self.input_details[0]['index'], x_r
            )
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            return out.flatten()



class EdgeInference:

    def __init__(self, model_dir: str = "models"):
        """
        QUOI    : Charge tous les modèles au démarrage
        POURQUOI: On charge une fois au début pour
                  être rapide ensuite (< 1ms par flux)
        """
        self.model_dir = Path(model_dir)
        self._load_all()

    def _load_all(self):
        """Charge tous les composants du pipeline"""

        # 1. Ordre exact des 16 features
        # POURQUOI : L'ordre DOIT être identique entre
        #            entraînement et production
        #            Si on change l'ordre → mauvaises prédictions
        feat_path = self.model_dir / "feature_names_toniot.json"
        self.feature_names = json.loads(
            feat_path.read_text(encoding="utf-8")
        )

        # 2. Seuils de décision (meta.json)
        # POURQUOI : LOW=0.30, HIGH=0.80
        #            définis dans config.yaml aussi
        meta_path = self.model_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(
                meta_path.read_text(encoding="utf-8")
            )
            self.threshold_low  = \
                meta["thresholds"]["low"]   # 0.30
            self.threshold_high = \
                meta["thresholds"]["high"]  # 0.80
        else:
            self.threshold_low  = 0.30
            self.threshold_high = 0.80

        # 3. StandardScaler
        # POURQUOI : MÊME scaler que l'entraînement
        #            (mêmes moyennes et écarts-types)
        self.scaler = joblib.load(
            self.model_dir / "scaler_toniot.joblib"
        )

        # 4. Encodeurs TFLite
        # POURQUOI : TFLite = version légère pour Edge
        #            AE : 5.7 KB, CNN : 3.4 KB, LSTM : 12.5 KB
        self.enc_ae = TFLiteEncoder(
            str(self.model_dir / "encoder_ae_toniot.tflite")
        )
        self.enc_lstm = TFLiteEncoder(
            str(self.model_dir / "encoder_lstm_toniot.tflite")
        )
        self.enc_cnn = TFLiteEncoder(
            str(self.model_dir / "encoder_cnn_toniot.tflite")
        )

        # 5. XGBoost
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(
            str(self.model_dir / "xgb_toniot.json")
        )
        self._xgb_lock = threading.Lock()

        # 6. Pool de threads pour encodeurs en parallèle
        # POURQUOI : Les 3 encodeurs sont indépendants
        #            On les lance en parallèle → 3x plus rapide
        self.pool = ThreadPoolExecutor(max_workers=3)

        print(f"✅ EdgeInference chargé")
        print(f"   Features    : {len(self.feature_names)}")
        print(f"   Seuil LOW   : {self.threshold_low}")
        print(f"   Seuil HIGH  : {self.threshold_high}")

    def _sanitize(self, x: np.ndarray) -> np.ndarray:
       
        x = np.where(np.isnan(x), 0.0, x)
        x = np.where(np.isinf(x), 0.0, x)
        return x

    def predict_proba_from_feature_dict(
        self, feature_dict: dict
    ) -> float:
       

        # ÉTAPE 1 : Vecteur numpy dans l'ordre exact
        X = np.array(
            [[float(feature_dict.get(name, 0.0))
              for name in self.feature_names]],
            dtype=np.float32
        )

        # ÉTAPE 2 : Nettoyage
        X = self._sanitize(X)

        # ÉTAPE 3 : Normalisation
        X_scaled = self.scaler.transform(X).astype(np.float32)

        # ÉTAPE 4 : Préparer les entrées pour chaque encodeur
        X_ae   = X_scaled               # (1, 16)
        X_lstm = X_scaled.reshape(1, 1, -1)  # (1, 1, 16)
        X_cnn  = X_scaled.reshape(1, -1, 1)  # (1, 16, 1)

        # ÉTAPE 5 : Encodeurs en PARALLÈLE
        f_ae   = self.pool.submit(self.enc_ae.predict,   X_ae)
        f_lstm = self.pool.submit(self.enc_lstm.predict, X_lstm)
        f_cnn  = self.pool.submit(self.enc_cnn.predict,  X_cnn)

        z_ae   = f_ae.result()    # (16,)
        z_lstm = f_lstm.result()  # (16,)
        z_cnn  = f_cnn.result()   # (16,)

        # ÉTAPE 6 : Fusion → 64 dimensions
        # [X_scaled(16) | z_AE(16) | z_LSTM(16) | z_CNN(16)]
        X_fused = np.concatenate(
            [X_scaled.flatten(), z_ae, z_lstm, z_cnn]
        ).astype(np.float32).reshape(1, -1)

        # ÉTAPE 7 : XGBoost → probabilité
        with self._xgb_lock:
            try:
                proba = float(
                    self.xgb_model.inplace_predict(X_fused)[0]
                )
            except Exception:
                dmat  = xgb.DMatrix(X_fused)
                proba = float(
                    self.xgb_model.predict(dmat)[0]
                )

        return float(np.clip(proba, 0.0, 1.0))

    def decide(self, proba: float) -> dict:
        """
        QUOI    : Convertit la probabilité en décision
        POURQUOI: 3 niveaux pour adapter la réponse
                  au risque détecté
        RETOURNE : dict avec level, label, action
        """
        if proba < self.threshold_low:
            return {
                "level"  : "LOW",
                "label"  : "normal",
                "action" : "Aucune intervention nécessaire"
            }
        elif proba < self.threshold_high:
            return {
                "level"  : "MEDIUM",
                "label"  : "suspicious",
                "action" : "Surveillance renforcée"
            }
        else:
            return {
                "level"  : "HIGH",
                "label"  : "attack",
                "action" : "ATTAQUE — Isolation recommandée"
            }