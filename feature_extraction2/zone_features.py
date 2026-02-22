import pandas as pd
import numpy as np
from typing import Dict
from collections import defaultdict

from utils.geometry import GeometryUtils
from utils.homography import HomographyUtils


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
        Calcule les features zones pour une fenêtre glissante.

        Paramètre
        ---------
        df : pd.DataFrame
            Détections sur la fenêtre courante
        """

        # -----------------------------
        # Cas limites
        # -----------------------------
        if df.empty or not self.zones or self.H is None:
            return self._empty_features()

        # On ne garde que les personnes
        persons = df[df["object_class"] == "person"]
        if persons.empty:
            return self._empty_features()

        # -----------------------------
        # Structures d'accumulation
        # -----------------------------

        # zone_id → set(track_id)
        persons_in_zone = defaultdict(set)

        # zone_id → temps cumulé passé dans la zone
        zone_time = defaultdict(float)

        # Comptage par type et niveau de risque (ML-ready)
        zone_type_count = defaultdict(int)
        zone_risk_count = defaultdict(int)

        # Détection d'événements entrée / sortie
        entry_exit_events = defaultdict(int)

        # Pour détecter exposition multi-zones par personne
        person_to_zones = defaultdict(set)

        # Zones à haut risque
        high_risk_zones = {
            z["id"] for z in self.zones if z["risk_level"] == "HIGH"
        }

        # Timestamp représentatif de la fenêtre
        timestamp = persons["timestamp"].iloc[0]

        # ==================================================
        # Boucle principale sur les personnes
        # ==================================================
        for _, p in persons.iterrows():
            pid = p["track_id"]

            # --------------------------------------------------
            # Point projeté = point au SOL (bas-centre bbox)
            # --------------------------------------------------
            x_img = p["bbox_x"]
            y_img = p["bbox_y"] + p["bbox_h"] / 2

            # Projection image → plan
            Xp, Yp = HomographyUtils.apply_homography(
                x_img, y_img, self.H
            )

            # Sécurité numérique
            if np.isnan(Xp) or np.isnan(Yp):
                continue

            # --------------------------------------------------
            # Test d'appartenance aux zones
            # --------------------------------------------------
            for z in self.zones:
                zid = z["id"]

                inside = GeometryUtils.point_in_polygon(
                    Xp, Yp, z["polygon"]
                )

                if not inside:
                    continue

                # -----------------------------
                # Présence personne / zone
                # -----------------------------
                persons_in_zone[zid].add(pid)
                person_to_zones[pid].add(zid)

                # -----------------------------
                # Temps passé (approximation sliding window)
                # -----------------------------
                zone_time[zid] += self.window_duration / len(persons)

                # -----------------------------
                # Comptages ML-ready
                # -----------------------------
                zone_type_count[z["type"]] += 1
                zone_risk_count[z["risk_level"]] += 1

                # -----------------------------
                # Entrée dans la zone
                # -----------------------------
                if (pid, zid) not in self._first_entry_time:
                    self._first_entry_time[(pid, zid)] = timestamp
                    entry_exit_events[zid] += 1

                # Mise à jour dernière présence
                self._last_exit_time[(pid, zid)] = timestamp

        # ==================================================
        # Agrégation finale
        # ==================================================

        # Nombre de personnes uniques dans au moins une zone
        num_persons_in_zone = len(person_to_zones)

        # Personnes exposées à plusieurs zones
        multiple_zone_exposure = sum(
            1 for zones in person_to_zones.values() if len(zones) > 1
        )

        # Temps total en zone
        total_time = sum(zone_time.values())

        # Temps en zones HIGH
        high_risk_time = sum(
            zone_time[zid] for zid in zone_time if zid in high_risk_zones
        )

        # Nombre de personnes dans zones HIGH
        num_people_in_high_risk_zone = sum(
            len(persons_in_zone[zid])
            for zid in persons_in_zone if zid in high_risk_zones
        )

        proportion_high_risk = (
            high_risk_time / total_time if total_time > 0 else 0.0
        )

        # ==================================================
        # Construction du dictionnaire final (ML-ready)
        # ==================================================
        features = {
            "num_persons_in_zone": num_persons_in_zone,
            "time_in_zone": total_time,
            "multiple_zone_exposure": multiple_zone_exposure,
            "num_people_in_high_risk_zone": num_people_in_high_risk_zone,
            "proportion_time_in_high_risk_zone": proportion_high_risk,
            "zone_entry_exit_events": sum(entry_exit_events.values()),
            "first_entry_time": min(self._first_entry_time.values())
            if self._first_entry_time else None,
            "last_exit_time": max(self._last_exit_time.values())
            if self._last_exit_time else None,
        }

        # Ajout des distributions aplaties (obligatoire pour ML)
        for k, v in zone_type_count.items():
            features[f"zone_type_{k}_count"] = v

        for k, v in zone_risk_count.items():
            features[f"zone_risk_{k}_count"] = v

        return features

    # ======================================================
    def _empty_features(self) -> Dict[str, float]:
        """
        Retourne un dictionnaire neutre lorsque la fenêtre
        ne contient aucune information exploitable
        """
        return {
            "num_persons_in_zone": 0,
            "time_in_zone": 0.0,
            "multiple_zone_exposure": 0,
            "num_people_in_high_risk_zone": 0,
            "proportion_time_in_high_risk_zone": 0.0,
            "zone_entry_exit_events": 0,
            "first_entry_time": None,
            "last_exit_time": None,
        }
