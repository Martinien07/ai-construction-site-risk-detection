# feature_extraction2/epi_features.py

import pandas as pd
from typing import Dict, Optional


# --------------------------------------------------
# Mapping explicite YOLO → métier
YOLO_TO_EPI = {
    "hardhat": ("helmet", True),
    "no_hardhat": ("helmet", False),
    "safety_vest": ("vest", True),
    "no_safety_vest": ("vest", False),
    "mask": ("mask", True),
    "no_mask": ("mask", False),
}


class EPIFeatures:
    """
    Extraction des caractéristiques liées au port des EPI
    (casque, gilet, masque) sur une fenêtre glissante.
    Association EPI ↔ personne par inclusion spatiale.
    """

    def __init__(self,
                 window_duration: float,
                 persistent_threshold: int = 3):
        self.window_duration = window_duration
        self.persistent_threshold = persistent_threshold
        self._consecutive_violation_count = 0

    # --------------------------------------------------
    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return self._empty_features()

        persons = df[df["object_class"] == "person"]
        if persons.empty:
            return self._empty_features()

        # Association EPI ↔ personnes
        person_epi = self._associate_epi_to_persons(df, persons)

        # États observés (exclut None)
        helmets = [p["helmet"] for p in person_epi.values() if p["helmet"] is not None]
        vests = [p["vest"] for p in person_epi.values() if p["vest"] is not None]
        masks = [p["mask"] for p in person_epi.values() if p["mask"] is not None]

        # Ratios de conformité
        helmet_ratio = sum(helmets) / len(helmets) if helmets else 0.0
        vest_ratio = sum(vests) / len(vests) if vests else 0.0
        mask_ratio = sum(masks) / len(masks) if masks else 0.0

        avg_epi = (helmet_ratio + vest_ratio + mask_ratio) / 3

        # Violations explicites uniquement
        no_helmet = sum(1 for p in person_epi.values() if p["helmet"] is False)
        no_vest = sum(1 for p in person_epi.values() if p["vest"] is False)
        no_mask = sum(1 for p in person_epi.values() if p["mask"] is False)

        has_violation = (no_helmet + no_vest + no_mask) > 0

        # Gestion de la persistance
        if has_violation:
            self._consecutive_violation_count += 1
        else:
            self._consecutive_violation_count = 0

        persistent_violation = (
            self._consecutive_violation_count >= self.persistent_threshold
        )

        violation_duration = (
            self._consecutive_violation_count * self.window_duration
        )

        return {
            "helmet_compliance_ratio": helmet_ratio,
            "vest_compliance_ratio": vest_ratio,
            "mask_compliance_ratio": mask_ratio,
            "avg_epi_compliance": avg_epi,
            "num_no_helmet_events": no_helmet,
            "num_no_vest_events": no_vest,
            "num_no_mask_events": no_mask,
            "epi_violation_duration": violation_duration,
            "persistent_epi_violation": int(persistent_violation),
        }

    # --------------------------------------------------
    def _associate_epi_to_persons(
        self,
        df: pd.DataFrame,
        persons: pd.DataFrame
    ) -> Dict[int, Dict[str, Optional[bool]]]:
        """
        Associe casque / gilet / masque à chaque personne
        par inclusion spatiale (centre de l'EPI dans bbox personne).
        """

        epis = df[df["object_class"].isin(YOLO_TO_EPI.keys())]

        result: Dict[int, Dict[str, Optional[bool]]] = {}

        for _, p in persons.iterrows():
            px, py, pw, ph = p["bbox_x"], p["bbox_y"], p["bbox_w"], p["bbox_h"]
            frame_time = p["timestamp"]

            result[p["track_id"]] = {
                "helmet": None,
                "vest": None,
                "mask": None,
            }

            for _, e in epis.iterrows():
                # même frame uniquement
                if e["timestamp"] != frame_time:
                    continue

                cx = e["bbox_x"] + e["bbox_w"] / 2
                cy = e["bbox_y"] + e["bbox_h"] / 2

                inside = (
                    px <= cx <= px + pw and
                    py <= cy <= py + ph
                )

                if not inside:
                    continue

                mapping = YOLO_TO_EPI.get(e["object_class"])
                if not mapping:
                    continue

                epi_name, epi_value = mapping
                result[p["track_id"]][epi_name] = epi_value

        return result

    # --------------------------------------------------
    def _empty_features(self) -> Dict[str, float]:
        return {
            "helmet_compliance_ratio": 0.0,
            "vest_compliance_ratio": 0.0,
            "mask_compliance_ratio": 0.0,
            "avg_epi_compliance": 0.0,
            "num_no_helmet_events": 0,
            "num_no_vest_events": 0,
            "num_no_mask_events": 0,
            "epi_violation_duration": 0.0,
            "persistent_epi_violation": 0,
        }
