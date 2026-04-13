import time
import joblib
import pandas as pd
import warnings
import os
import json
from datetime import datetime, timedelta

# Suppression des avertissements de connexion SQL/Pandas pour un log propre
warnings.filterwarnings("ignore", category=UserWarning)

# Import des composants
from db.connection import init_db
from db.repositories.detection_repo import DetectionRepository
from db.repositories.zone_repo import ZoneRepository
from db.repositories.camera_calibration_repo import CameraCalibrationRepository
from db.repositories.camera_repo import CameraRepository
from db.repositories.risk_event_repository import RiskEventRepository

from feature_extraction2.sliding_window import SlidingWindow
from feature_extraction2.pipeline import extract_features_pipeline
from feature_extraction2.activity_prediction import ActivityPredictor 

from system_automatic.analyzer import Analyzer as MetaAnalyzer
from system_automatic.risk_engine import RiskEngine
from system_automatic.decision_engine import DecisionEngine

class HSEAnalysisSystem:
    def __init__(self, model_path=None):
        init_db()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        # 1. Chemin du Modèle ML
        # Si aucun chemin n'est fourni, on construit le chemin absolu vers le dossier dataset
        if model_path is None:
            model_path = os.path.join(root_dir, "dataset machine Learning", "model_hse_v1.pkl")

        # 1. Chargement du Modèle ML
        print(f"Chargement du modèle ML : {model_path}...")
        try:
            model = joblib.load(model_path)
            self.predictor = ActivityPredictor(model)
        except Exception as e:
            print(f"Attention : Erreur chargement modèle : {e}")
            self.predictor = None

        # 2. Initialisation des Repositories
        self.detection_repo = DetectionRepository()
        self.camera_repo = CameraRepository()
        self.zone_repo = ZoneRepository()
        self.calib_repo = CameraCalibrationRepository()
        
        # 3. Méta-composants (Note: RiskEngine chargé en UTF-8)
        self.meta_analyzer = MetaAnalyzer()
        
        # --- CORRECTION ENCODAGE ICI ---
        # On s'assure que le RiskEngine ouvre le JSON en UTF-8
        rules_path = os.path.join(current_dir, 'hse_rules.json')
        print(f"Chargement des règles : {rules_path}")
        self.risk_engine = RiskEngine(rules_path)
        
        self.decision_engine = DecisionEngine(frequency_threshold=0.20)
        self.sliding_window = SlidingWindow(
            detection_repository=self.detection_repo, 
            window_duration=5.0, 
            step=1.0
        )
        
        # 4. Extracteurs
        from feature_extraction2.human_features import HumanPresenceFeatures
        from feature_extraction2.epi_features import EPIFeatures
        from feature_extraction2.machine_features import MachineVehicleFeatures
        from feature_extraction2.proximity_features import ProximityFeatures
        from feature_extraction2.temporal_dynamics_features import TemporalDynamicsFeatures
        from feature_extraction2.activity_stability_features import ActivityStabilityFeatures
        from feature_extraction2.behavior_analyzer import BehaviorAnalyzer

        self.human_extractor = HumanPresenceFeatures(1920, 1080, 2.0)
        self.epi_extractor = EPIFeatures(5.0, 3)
        self.machine_extractor = MachineVehicleFeatures(5.0)
        self.proximity_extractor = ProximityFeatures(5.0, 100, 150)
        self.temporal_extractor = TemporalDynamicsFeatures(5.0)
        self.stability_extractor = ActivityStabilityFeatures(5)
        self.behavior_extractor = BehaviorAnalyzer(30)
        
        self.current_camera_id = None
        self.zone_extractor = None

    def _update_camera_config(self, camera_id):
        if camera_id == self.current_camera_id and self.zone_extractor is not None:
            return True
            
        cam_info = self.camera_repo.get_camera_config(camera_id)
        if cam_info is not None and not (isinstance(cam_info, (pd.DataFrame, pd.Series)) and cam_info.empty):
            plan_id = int(cam_info['plan_id'])
            H = self.calib_repo.get_active_calibration(camera_id, plan_id)
            zones = self.zone_repo.get_active_zones_by_plan(plan_id=plan_id)
            
            if H is not None:
                from feature_extraction2.zone_features import ZoneFeatures
                self.zone_extractor = ZoneFeatures(zones=zones, homography=H, window_duration=5.0)
                self.current_camera_id = camera_id
                return True
        return False

    def _execute_pipeline(self, lookback_minutes, camera_id):
        if not self._update_camera_config(camera_id):
            return None

        # A. Récupération des données
        end_time = self.meta_analyzer.get_last_detection_timestamp()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        result = extract_features_pipeline(
            sliding_window=self.sliding_window,
            behavior_extractor=self.behavior_extractor,
            human_extractor=self.human_extractor,
            epi_extractor=self.epi_extractor,
            machine_extractor=self.machine_extractor,
            proximity_extractor=self.proximity_extractor,
            temporal_extractor=self.temporal_extractor,
            stability_extractor=self.stability_extractor,
            zone_extractor=self.zone_extractor,
            camera_ids=[camera_id],
            start_time=start_time,
            end_time=end_time,
            max_windows=100
        )

        df_all_list, df_ml_list = result if result else (None, None)
        if df_all_list is None or len(df_all_list) == 0:
            return None

        # B. IA : Prédiction
        df_ml = pd.DataFrame(df_ml_list)
        df_all = pd.DataFrame(df_all_list)

        if self.predictor:
            df_ml_pred = self.predictor.predict(df_ml)
            df_all["activity_pred"] = df_ml_pred["activity_pred"]
            df_all["activity_confidence"] = df_ml_pred["activity_confidence"]

        # C. Moteur de Risques
        df_analyzed = self.risk_engine.process_dataframe(df_all)

        # D. Décision
        df_final = self.decision_engine.aggregate_decisions(df_analyzed)

        # E. Affichage et Enregistrement
        if not df_final.empty:
            self.meta_analyzer.update_last_timestamp(df_analyzed)
            print(f" [!] {len(df_final)} Risque(s) détecté(s) sur Caméra {camera_id}")
            
            cols_to_show = [c for c in ['activity_pred', 'risk_level', 'risk_messages'] if c in df_final.columns]
            # Utilisation de tail(1) pour un affichage compact
            print(df_final[cols_to_show].tail(1).to_string(index=False))
            
            for _, row in df_final.iterrows():
                try:
                    RiskEventRepository.save_event_and_alert(row)
                except Exception as e:
                    print(f"  Erreur sauvegarde : {e}")
        
        return df_final

    def run(self, site_id, lookback_minutes=1, frequency_seconds=10):
        from db.repositories.camera_discovery_repository import CameraDiscoveryRepository
        print(f"--- Démarrage Système HSE Global (Site {site_id}) ---")
        
        while True:
            try:
                cameras = CameraDiscoveryRepository.get_site_cameras(site_id)
                
                if not cameras:
                    print(f" [ {datetime.now().strftime('%H:%M:%S')}] Aucune caméra active.")
                else:
                    for cam in cameras:
                        cam_id = cam['camera_id']
                        # Pipeline d'analyse
                        self._execute_pipeline(lookback_minutes, cam_id)
                
                # Log de fin de cycle
                print(f"--- [ {datetime.now().strftime('%H:%M:%S')}] Cycle Site {site_id} OK. Attente {frequency_seconds}s ---")
                time.sleep(frequency_seconds)
                
            except KeyboardInterrupt:
                print("\nArrêt du système...")
                break