# feature_extraction2/proximity_features.py

import pandas as pd
import numpy as np
from typing import Dict
from itertools import combinations
from utils.geometry import GeometryUtils



class ProximityFeatures:
    """
    Extraction des caractéristiques de proximité spatiale
    entre personnes et machines sur une fenêtre glissante.
    """

    def __init__(
        self,
        window_duration: float,
        danger_distance_pp: float = 100.0,
        danger_distance_pm: float = 150.0,
    ):
        """
        window_duration : durée fenêtre (secondes)
        danger_distance_pp : seuil danger personne-personne (pixels)
        danger_distance_pm : seuil danger personne-machine (pixels)


        Pourquoi ce seuil est fondamental
        Détecter les situations de sur-densité (crowding)

        Quand plusieurs personnes sont trop proches :

        perte de mobilité

        risques de chutes

        impossibilité d’évacuation

        collisions humaines

        Typiquement :

        échafaudage

        passerelle

        tranchée

        zone confinée
        """
        self.window_duration = window_duration
        self.danger_distance_pp = danger_distance_pp
        self.danger_distance_pm = danger_distance_pm

    # ======================================================
    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return self._empty_features()

        persons = df[df["object_class"] == "person"]
        machines = df[df["object_class"].isin(["vehicle", "machinery"])]

        # --------------------------------------------------
        # PERSON ↔ PERSON
        # --------------------------------------------------
        pp_distances = self._pairwise_distances(persons)

        avg_pp = float(np.mean(pp_distances)) if pp_distances else 0.0
        std_pp = float(np.std(pp_distances)) if pp_distances else 0.0
        min_pp = float(np.min(pp_distances)) if pp_distances else 0.0

        dangerous_pp = sum(
            d < self.danger_distance_pp for d in pp_distances
        )

        # --------------------------------------------------
        # PERSON ↔ MACHINE
        # --------------------------------------------------
        pm_distances = self._cross_distances(persons, machines)

        avg_pm = float(np.mean(pm_distances)) if pm_distances else 0.0
        min_pm = float(np.min(pm_distances)) if pm_distances else 0.0

        dangerous_pm = sum(
            d < self.danger_distance_pm for d in pm_distances
        )

        time_near_machine = 0.0
        num_machine_interactions = 0

        for _, p in persons.iterrows():
            for _, m in machines.iterrows():
                d = GeometryUtils.euclidean_distance(
                    p["bbox_x"], p["bbox_y"],
                    m["bbox_x"], m["bbox_y"]
                )

                if d < self.danger_distance_pm:
                    num_machine_interactions += 1
                    time_near_machine += self.window_duration







        # --------------------------------------------------
        # OVERLAP PERSON ↔ MACHINE
        # --------------------------------------------------
        overlaps = self._bbox_overlaps(persons, machines)

        return {
            "avg_person_person_distance": avg_pp,
            "std_person_person_distance": std_pp,
            "avg_person_machine_distance": avg_pm,
            "min_person_person_distance": min_pp,
            "min_person_machine_distance": min_pm,
            "dangerous_proximity_frames": dangerous_pp + dangerous_pm,
            "bbox_overlap_person_machine": overlaps,
            "time_near_machine": time_near_machine,
            "num_machine_interactions": num_machine_interactions,
        }

    # ======================================================
    #  UTILS
    # ======================================================
    def _pairwise_distances(self, persons: pd.DataFrame):
        """
        Distances centre-centre entre personnes
        """
        coords = list(
            zip(persons["bbox_x"], persons["bbox_y"])
        )

        distances = []
        for (x1, y1), (x2, y2) in combinations(coords, 2):
            distances.append(np.hypot(x1 - x2, y1 - y2))

        return distances

    def _cross_distances(
        self,
        persons: pd.DataFrame,
        machines: pd.DataFrame,
    ):
        """
        Distances centre-centre personne ↔ machine
        """
        distances = []

        for _, p in persons.iterrows():
            for _, m in machines.iterrows():
                distances.append(
                    np.hypot(
                        p["bbox_x"] - m["bbox_x"],
                        p["bbox_y"] - m["bbox_y"],
                    )
                )

        return distances

    def _bbox_overlaps(
        self,
        persons: pd.DataFrame,
        machines: pd.DataFrame,
    ) -> int:
        """
        Nombre de collisions potentielles personne-machine
        """
        overlaps = 0

        for _, p in persons.iterrows():
            for _, m in machines.iterrows():
                if self._overlap(p, m):
                    overlaps += 1

        return overlaps

    def _overlap(self, a, b) -> bool:
        """
        Test d'intersection entre deux bounding boxes
        """
        ax1 = a["bbox_x"] - a["bbox_w"] / 2
        ay1 = a["bbox_y"] - a["bbox_h"] / 2
        ax2 = a["bbox_x"] + a["bbox_w"] / 2
        ay2 = a["bbox_y"] + a["bbox_h"] / 2

        bx1 = b["bbox_x"] - b["bbox_w"] / 2
        by1 = b["bbox_y"] - b["bbox_h"] / 2
        bx2 = b["bbox_x"] + b["bbox_w"] / 2
        by2 = b["bbox_y"] + b["bbox_h"] / 2

        return not (
            ax2 < bx1 or ax1 > bx2 or
            ay2 < by1 or ay1 > by2
        )

    def _empty_features(self) -> Dict[str, float]:
        return {
            "avg_person_person_distance": 0.0,
            "std_person_person_distance": 0.0,
            "avg_person_machine_distance": 0.0,
            "min_person_person_distance": 0.0,
            "min_person_machine_distance": 0.0,
            "dangerous_proximity_frames": 0,
            "bbox_overlap_person_machine": 0,
        }
