import numpy as np
from collections import deque
from typing import Dict


class ActivityTransitionFeatures:
    """
    Extraction de features liées aux transitions d'activité.

    Objectifs ML :
    - détecter les changements d'activité (inspection → travail, etc.)
    - capturer la dynamique de déplacement entre zones
    - différencier activité continue vs fragmentée

    Ces features exploitent un historique court de fenêtres.
    """

    def __init__(
        self,
        history_size: int = 5,
        person_change_threshold: float = 0.5,
    ):
        """
        Paramètres :
        - history_size : nombre de fenêtres utilisées pour l'historique
        - person_change_threshold : variation relative du nombre
          de personnes considérée comme un changement significatif
        """

        self.history_size = history_size
        self.person_change_threshold = person_change_threshold

        # Mémoire glissante
        self._person_count_history = deque(maxlen=history_size)
        self._zone_count_history = deque(maxlen=history_size)
        self._high_risk_history = deque(maxlen=history_size)

    # ======================================================
    def extract(self, current_features: Dict) -> Dict[str, float]:
        """
        Calcule les features de transition d'activité.

        current_features : dictionnaire de features déjà extraites
        pour la fenêtre courante (une ligne du pipeline).
        """

        num_persons = current_features.get("avg_num_persons", 0.0)
        num_zones = current_features.get("num_persons_in_zone", 0)
        num_high_risk = current_features.get("num_people_in_high_risk_zone", 0)

        # --------------------------------------------------
        # Warm-up : pas assez d'historique
        # --------------------------------------------------
        if len(self._person_count_history) < 2:
            self._update_history(num_persons, num_zones, num_high_risk)
            return self._empty_features()

        # --------------------------------------------------
        # 1 Taux de changement d'activité
        # --------------------------------------------------
        prev_persons = np.mean(self._person_count_history)
        delta_persons = abs(num_persons - prev_persons)

        activity_change_rate = (
            delta_persons / (prev_persons + 1e-6)
            if prev_persons > 0
            else 0.0
        )

        # Limitation pour stabilité ML
        activity_change_rate = float(np.clip(activity_change_rate, 0.0, 1.0))

        # --------------------------------------------------
        # 2 Transitions entre zones
        # --------------------------------------------------
        zone_variation = np.std(self._zone_count_history)
        zone_transition_count = int(round(zone_variation))

        # --------------------------------------------------
        # 3 Instabilité du niveau de risque
        # --------------------------------------------------
        risk_variation = np.std(self._high_risk_history)
        risk_transition_score = risk_variation / (np.mean(self._high_risk_history) + 1e-6)

        risk_transition_score = float(np.clip(risk_transition_score, 0.0, 1.0))

        # --------------------------------------------------
        # Mise à jour historique
        # --------------------------------------------------
        self._update_history(num_persons, num_zones, num_high_risk)

        return {
            "activity_change_rate": activity_change_rate,
            "zone_transition_count": zone_transition_count,
            "risk_transition_score": risk_transition_score,
        }

    # ======================================================
    def _update_history(
        self,
        num_persons: float,
        num_zones: int,
        num_high_risk: int,
    ):
        """Met à jour les buffers internes"""
        self._person_count_history.append(num_persons)
        self._zone_count_history.append(num_zones)
        self._high_risk_history.append(num_high_risk)

    # ======================================================
    def _empty_features(self) -> Dict[str, float]:
        """
        Valeurs neutres pour les premières fenêtres.
        Pas de NaN (robuste XGBoost).
        """
        return {
            "activity_change_rate": 0.0,
            "zone_transition_count": 0,
            "risk_transition_score": 0.0,
        }
