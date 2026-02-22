import numpy as np
from collections import deque
from typing import Dict


class ActivityStabilityFeatures:
    """
    Extraction de features de stabilité temporelle de l'activité.

    Objectifs ML :
    - réduire le bruit inter-fenêtres
    - détecter les activités stables (travail, inspection)
    - différencier passage / circulation / activité réelle

    Ces features utilisent une mémoire glissante
    (stateful, compatible temps réel).
    """

    def __init__(
        self,
        history_size: int = 5,
    ):
        """
        Paramètres :
        - history_size : nombre de fenêtres précédentes utilisées
        """

        self.history_size = history_size

        # Buffers glissants
        self._speed_history = deque(maxlen=history_size)
        self._person_count_history = deque(maxlen=history_size)
        self._motion_score_history = deque(maxlen=history_size)

    # ======================================================
    def extract(self, current_features: Dict) -> Dict[str, float]:
        """
        Calcule les features de stabilité à partir
        des features déjà extraites pour la fenêtre courante.

        current_features : dict du pipeline (features instantanées)
        """

        avg_speed = float(current_features.get("avg_person_speed", 0.0))
        num_persons = float(current_features.get("avg_num_persons", 0.0))
        erratic_motion = float(current_features.get("erratic_motion_score", 0.0))

        # --------------------------------------------------
        # Mise à jour de l'historique AVANT calcul
        # --------------------------------------------------
        self._update_history(avg_speed, num_persons, erratic_motion)

        # Pas assez d'historique → valeurs neutres mais cohérentes
        if len(self._speed_history) < 2:
            return {
                "activity_persistence_score": 0.5,
                "motion_consistency": 0.5,
                "motion_stability_score": 0.5,
            }

        # --------------------------------------------------
        # 1 Persistance de l'activité
        # --------------------------------------------------
        person_std = np.std(self._person_count_history)
        activity_persistence_score = 1 / (1 + person_std)

        # --------------------------------------------------
        # 2 Consistance du mouvement
        # --------------------------------------------------
        speed_std = np.std(self._speed_history)
        speed_mean = np.mean(self._speed_history) + 1e-6

        motion_consistency = 1 - (speed_std / speed_mean)
        motion_consistency = float(np.clip(motion_consistency, 0.0, 1.0))

        # --------------------------------------------------
        # 3 Stabilité comportementale (anti-bruit)
        # --------------------------------------------------
        erratic_variation = np.std(self._motion_score_history)
        motion_stability_score = 1 / (1 + erratic_variation)

        return {
            "activity_persistence_score": float(activity_persistence_score),
            "motion_consistency": motion_consistency,
            "motion_stability_score": float(motion_stability_score),
        }

    # ======================================================
    def _update_history(
        self,
        avg_speed: float,
        num_persons: float,
        erratic_motion: float,
    ):
        """Met à jour les buffers historiques"""
        self._speed_history.append(avg_speed)
        self._person_count_history.append(num_persons)
        self._motion_score_history.append(erratic_motion)
