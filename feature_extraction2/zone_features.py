import pandas as pd
import numpy as np
from typing import Dict
from collections import defaultdict


from feature_extraction2.utils.geometry import GeometryUtils
from feature_extraction2.utils.homography import HomographyUtils

class ZoneFeatures:
    """
    Extraction des caractéristiques spatiales liées aux zones de risque
    à partir d'une projection caméra → plan (homographie).

    Cette classe permet de :
    - projeter les personnes détectées vers le plan réel
    - déterminer leur appartenance aux zones de risque
    - agréger des métriques exploitables pour du ML (XGBoost / RF)
    """

    # ======================================================
    def __init__(
        self,
        zones: list,
        homography: np.ndarray,
        window_duration: float
    ):
        """
        Paramètres
        ----------
        zones : list[dict]
            Zones issues de la base de données.
            Chaque zone contient :
            {
                id,
                type,            # ex: "traffic", "height", "chemical"
                risk_level,      # ex: "LOW", "MEDIUM", "HIGH"
                polygon          # liste de points [(x, y), ...] en coordonnées plan
            }

        homography : np.ndarray
            Matrice 3x3 permettant la projection image → plan

        window_duration : float
            Durée de la fenêtre glissante (en secondes)
        """
        self.zones = zones
        self.H = homography
        self.window_duration = window_duration

        # Mémoire inter-fenêtres
        # (track_id, zone_id) → timestamp
        self._first_entry_time = {}
        self._last_exit_time = {}





    # ======================================================
    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extrait les 11 features cibles pour le modèle de scoring HSE.
        """
        # Initialisation : Valeurs par défaut pour toutes les features
        features = {
            "num_persons_in_zone": 0.0,
            "num_people_in_high_risk_zone": 0.0,
            "proportion_time_in_high_risk_zone": 0.0,
            "multiple_zone_exposure": 0.0,
            "avg_person_height": 0.0,
            "time_near_machine": 0.0,
            "num_machine_interactions": 0.0,
            "motion_consistency": 0.0,
            "activity_persistence_score": 0.0,
            "zone_type": 6.0,          # ID 'standard' par défaut
            "zone_risk_level": 1.0     # Niveau 'LOW' par défaut
        }

        if df.empty or self.H is None:
            return features

        persons = df[df["object_class"] == "person"]
        if persons.empty:
            return features

        # Paramétrage pour les zones
        RISK_MAP = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}
        person_to_zones = defaultdict(set)
        
        # 1. Analyse Spatiale (Boucle principale)
        for _, p in persons.iterrows():
            Xp, Yp = HomographyUtils.apply_homography(p["bbox_x"], p["bbox_y"] + p["bbox_h"] / 2, self.H)
            
            for z in self.zones:
                if not z.get("polygon") or len(z["polygon"]) < 3: continue
                if GeometryUtils.point_in_polygon(Xp, Yp, z["polygon"]):
                    person_to_zones[p["track_id"]].add(z["id"])
                    
                    # Mise à jour des features spatiales prioritaires (la zone la plus critique)
                    if z["risk_level"] == "HIGH" or features["zone_risk_level"] < RISK_MAP.get(z["risk_level"], 1.0):
                        features["zone_type"] = float(z["id"])
                        features["zone_risk_level"] = RISK_MAP.get(z["risk_level"], 1.0)

        # 2. Agrégation des métriques d'occupation
        features["num_persons_in_zone"] = float(persons["track_id"].nunique())
        features["multiple_zone_exposure"] = float(sum(1 for z_ids in person_to_zones.values() if len(z_ids) > 1))
        
        high_risk_ids = {z["id"] for z in self.zones if z["risk_level"] == "HIGH"}
        high_risk_count = sum(1 for z_ids in person_to_zones.values() if any(zid in high_risk_ids for zid in z_ids))
        
        features["num_people_in_high_risk_zone"] = float(high_risk_count)
        features["proportion_time_in_high_risk_zone"] = 1.0 if high_risk_count > 0 else 0.0

        # 3. Injection des autres métriques (à connecter à tes modules de calcul)
        # Ces valeurs sont supposées calculées par tes fonctions auxiliaires ou passées par le DF
        features["avg_person_height"] = float(persons["bbox_h"].mean()) if not persons.empty else 0.0
        
        return features
    # ======================================================
    def _empty_features(self) -> Dict[str, float]:
            """
            Retourne un dictionnaire neutre avec toutes les clés attendues par le ML,
            initialisées à 0.0 pour garantir l'intégrité du vecteur d'entrée.
            """
            # Features de base
            features = {
                "num_persons_in_zone": 0.0,
                "multiple_zone_exposure": 0.0,
                "global_risk_exposure": 0.0,
                "zone_entry_exit_events": 0.0,
                "total_time_in_zones": 0.0
            }
            
            # Initialisation dynamique pour chaque zone configurée
            for z in self.zones:
                ztype = z["type"]
                features[f"time_in_{ztype}_zone"] = 0.0
                features[f"persons_in_{ztype}_zone"] = 0.0
                features[f"risk_level_{ztype}"] = 0.0
                
            return features