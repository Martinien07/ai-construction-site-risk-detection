import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict

class BehaviorAnalyzer:
    """
    Analyseur comportemental gérant la mémoire temporelle des trajectoires.
    Fournit les scores de consistance, de persistance et d'interaction machine.
    """
    def __init__(self, history_size: int = 30, proximity_threshold: float = 50.0):
        # Stockage de l'historique : {track_id: [(x, y), ...]}
        self.history = defaultdict(list)
        self.history_size = history_size
        self.proximity_threshold = proximity_threshold # Seuil pour near_machine

    def update_history(self, persons_df: pd.DataFrame):
        """Met à jour l'historique des positions pour chaque individu présent."""
        current_ids = set()
        
        # On ne traite que les personnes
        if not persons_df.empty:
            active_persons = persons_df[persons_df["object_class"] == "person"]
            for _, p in active_persons.iterrows():
                pid = p["track_id"]
                # On utilise le centre ou le bas de la bbox pour la trajectoire
                pos = (p["bbox_x"] + p["bbox_w"]/2, p["bbox_y"] + p["bbox_h"])
                self.history[pid].append(pos)
                
                if len(self.history[pid]) > self.history_size:
                    self.history[pid].pop(0)
                current_ids.add(pid)
        
        # Nettoyage automatique des IDs disparus pour libérer la mémoire
        keys_to_del = [pid for pid in self.history if pid not in current_ids]
        for pid in keys_to_del:
            del self.history[pid]

    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Interface principale pour le pipeline. 
        Retourne les moyennes comportementales de la fenêtre actuelle.
        """
        # Initialisation conforme à ta liste de features
        res = {
            "motion_consistency": 0.0,
            "activity_persistence_score": 0.0,
            "time_near_machine": 0.0,
            "num_machine_interactions": 0.0
        }

        if df.empty:
            return res

        pids = df[df["object_class"] == "person"]["track_id"].unique()
        if len(pids) == 0:
            return res

        # Calcul des scores individuels
        consistencies = []
        persistences = []
        
        for pid in pids:
            if pid in self.history and len(self.history[pid]) > 1:
                consistencies.append(self._calc_consistency(pid))
                persistences.append(self._calc_persistence(pid))

        # Agrégation (Moyenne des personnes présentes dans la fenêtre)
        res["motion_consistency"] = float(np.mean(consistencies)) if consistencies else 0.0
        res["activity_persistence_score"] = float(np.mean(persistences)) if persistences else 0.0
        
        # Note: time_near_machine et num_machine_interactions peuvent être calculés ici 
        # si les positions des machines sont connues de cette classe.
        
        return res

    def _calc_consistency(self, pid: int) -> float:
        """Calcule la stabilité du mouvement (inverse de la variance des déplacements)."""
        pos = np.array(self.history[pid])
        if len(pos) < 5: return 0.5 # Valeur neutre si peu de recul
        
        diffs = np.diff(pos, axis=0)
        variance = np.var(diffs, axis=0).mean()
        # Score entre 0 (erratique) et 1 (fluide)
        return float(np.clip(1.0 / (1.0 + variance / 10.0), 0, 1))

    def _calc_persistence(self, pid: int) -> float:
        """Calcule si la personne reste focalisée sur un point (travail statique)."""
        pos = np.array(self.history[pid])
        if len(pos) < 10: return 0.0
        
        # Distance moyenne par rapport au centre de gravité de la trajectoire
        centroid = pos.mean(axis=0)
        avg_dist = np.linalg.norm(pos - centroid, axis=1).mean()
        
        # Plus la distance est faible, plus la persistance est haute (max 100px de rayon)
        return float(np.clip(1.0 - (avg_dist / 100.0), 0, 1))