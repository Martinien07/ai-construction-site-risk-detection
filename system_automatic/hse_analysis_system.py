import time
import pandas as pd
from datetime import datetime, timedelta

from db.connection import init_db
from db.repositories.detection_repo import DetectionRepository
from db.repositories.zone_repo import ZoneRepository
from db.repositories.camera_calibration_repo import CameraCalibrationRepository
from db.repositories.camera_repo import CameraRepository
from db.repositories.alert_repository import AlertRepository



from feature_extraction2.sliding_window import SlidingWindow
from feature_extraction2.human_features import HumanPresenceFeatures
from feature_extraction2.epi_features import EPIFeatures
from feature_extraction2.machine_features import MachineVehicleFeatures
from feature_extraction2.proximity_features import ProximityFeatures
from feature_extraction2.temporal_dynamics_features import TemporalDynamicsFeatures
from feature_extraction2.zone_features import ZoneFeatures
from feature_extraction2.activity_stability_features import ActivityStabilityFeatures
from feature_extraction2.behavior_analyzer import BehaviorAnalyzer
from feature_extraction2.pipeline import extract_features_pipeline

from system_automatic.analyzer import Analyzer as MetaAnalyzer
from system_automatic.risk_engine import RiskEngine

import sys
sys.path.insert(0, r"D:\ETUDEµ\CITE\ETAPE4\PROJET DE FIN DE SESSION\TOURMANT 1")


class HSEAnalysisSystem:
    def __init__(self):
        init_db()
        
        # Repositories
        self.detection_repo = DetectionRepository()
        self.camera_repo = CameraRepository()
        self.zone_repo = ZoneRepository()
        self.calib_repo = CameraCalibrationRepository()
        
        # Méta-composants
        self.meta_analyzer = MetaAnalyzer()
        self.risk_engine = RiskEngine('hse_rules.json')
        self.sliding_window = SlidingWindow(
            detection_repository=self.detection_repo, 
            window_duration=5.0, 
            step=1.0
        )
        
        # État dynamique de la caméra
        self.current_camera_id = None
        self.zone_extractor = None
        
        # Extracteurs statiques
        self.human_extractor = HumanPresenceFeatures(1920, 1080, 2.0)
        self.epi_extractor = EPIFeatures(5.0, 3)
        self.machine_extractor = MachineVehicleFeatures(5.0)
        self.proximity_extractor = ProximityFeatures(5.0, 100, 150)
        self.temporal_extractor = TemporalDynamicsFeatures(5.0)
        self.stability_extractor = ActivityStabilityFeatures(5)
        self.behavior_extractor = BehaviorAnalyzer(30)

    def _update_camera_config(self, camera_id):
            """Met à jour les zones et l'homographie si la caméra change."""
            if camera_id == self.current_camera_id and self.zone_extractor is not None:
                return True
                
            cam_info = self.camera_repo.get_camera_config(camera_id)
            
            # CORRECTION : On vérifie si cam_info n'est pas None ET n'est pas vide
            if cam_info is not None and not (isinstance(cam_info, (pd.DataFrame, pd.Series)) and cam_info.empty):
                # Selon ce que retourne ton repo, on extrait le plan_id
                # Si c'est un dictionnaire ou une Series :
                plan_id = int(cam_info['plan_id'])
                
                H = self.calib_repo.get_active_calibration(camera_id, plan_id)
                zones = self.zone_repo.get_active_zones_by_plan(plan_id=plan_id)
                
                if H is not None:
                    self.zone_extractor = ZoneFeatures(
                        zones=zones,
                        homography=H,
                        window_duration=5.0
                    )
                    self.current_camera_id = camera_id
                    print(f" Config mise à jour : Caméra {camera_id} (Plan {plan_id})")
                    return True
            return False

    def run(self, mode='oneshot', lookback_minutes=5, frequency_seconds=2, camera_id=1):
        if mode == 'oneshot':
            return self._execute_pipeline(lookback_minutes, camera_id)
        
        elif mode == 'cycle':
            print(f" Mode Cycle démarré (Fréquence: {frequency_seconds}s)")
            try:
                while True:
                    self._execute_pipeline(lookback_minutes, camera_id)
                    time.sleep(frequency_seconds)
            except KeyboardInterrupt:
                print("\n Arrêt du système.")

    def _execute_pipeline(self, lookback_minutes, camera_id):
            # 1. Mise à jour config caméra
            if not self._update_camera_config(camera_id):
                print(f" Impossible de charger la config pour la caméra {camera_id}")
                return None

            # 2. Gestion du temps
            end_time = self.meta_analyzer.get_last_detection_timestamp()
            start_time = end_time - timedelta(minutes=lookback_minutes)
            
            print(f"\n Fenêtre : {start_time.strftime('%H:%M:%S')} -> {end_time.strftime('%H:%M:%S')}")
            
            # 3. Appel du pipeline
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
            
            # On décompose le résultat 
            df_all_list, df_ml_list = result if result else (None, None)
            
            # CORRECTION : Vérification sécurisée de la liste
            if df_all_list is None or len(df_all_list) == 0:
                print(" Aucune donnée trouvée sur cette période.")
                return None

            # 4. Conversion et Mise à jour
            df_all = pd.DataFrame(df_all_list)
            
            # Vérification supplémentaire avant l'update
            if not df_all.empty:
                self.meta_analyzer.update_last_timestamp(df_all)
                print(f" {len(df_all)} fenêtres traitées avec succès.")
                return df_all
                
            return None