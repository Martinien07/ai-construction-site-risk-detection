import pandas as pd
import numpy as np
from typing import Dict
from math import atan2, degrees
from feature_extraction2.geometry import GeometryUtils


class TemporalDynamicsFeatures:
    """
    Extraction des caractéristiques dynamiques des personnes
    sur une fenêtre glissante.

    Ces features permettent de détecter :
    - les vitesses anormales
    - les arrêts brusques (sudden stop)
    - les mouvements erratiques
    - les interactions à risque avec des machines
    """

    def __init__(self,
                 window_duration: float,
                 sudden_stop_threshold: float = 0.5,  # % chute vitesse
                 high_speed_threshold: float = 50.0,  # pixels/sec
                 near_machine_distance: float = 150.0):
        self.window_duration = window_duration
        self.sudden_stop_threshold = sudden_stop_threshold
        self.high_speed_threshold = high_speed_threshold
        self.near_machine_distance = near_machine_distance

    """
    Paramètres :
    - window_duration : durée d’une fenêtre glissante (en secondes)
    - sudden_stop_threshold : chute relative de vitesse pour compter un arrêt brutal
    - high_speed_threshold : seuil de vitesse considérée comme rapide
    - near_machine_distance : distance minimale pour considérer qu'une personne est proche d'une machine
    """

    # ======================================================
    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule toutes les features dynamiques pour une fenêtre glissante
        à partir des détections stockées dans df.
        """
        if df.empty:
            return self._empty_features()  # Aucun objet détecté → retour features à 0

        # Séparation des personnes et des machines
        persons = df[df["object_class"] == "person"]
        machines = df[df["object_class"].isin(["vehicle", "machinery"])]

        # Initialisation des listes et compteurs
        speeds = []              # vitesses calculées par track
        directions = []          # angles de mouvement
        lifetimes = []           # durée d'observation d'une personne dans la fenêtre
        sudden_stops = 0         # nombre d'arrêts brusques
        high_speed_near_machine = 0  # nombre d'événements "vitesse élevée proche machine"

        # --------------------------------------------------
        # Parcours par individu (track_id)
        for track_id, track_df in persons.groupby("track_id"):
            # Tri des détections par timestamp pour calculer les vitesses correctement
            track_df = track_df.sort_values("timestamp")

            # Calcul du centre de la bounding box pour chaque frame
            coords = [
                (row["bbox_x"] + row["bbox_w"]/2,
                 row["bbox_y"] + row["bbox_h"]/2)
                for _, row in track_df.iterrows()
            ]

            timestamps = track_df["timestamp"].values
            track_speeds = []       # vitesses successives pour ce track
            track_directions = []   # directions successives pour ce track

            # --------------------------------------------------
            # Calcul des vitesses et directions
            for i in range(1, len(coords)):
                dt = (timestamps[i] - timestamps[i-1])/ np.timedelta64(1, "s") # delta temps
                if dt <= 0:
                    continue  # éviter division par 0 ou timestamps identiques

                # vitesse euclidienne pixels/sec
                s = GeometryUtils.euclidean_distance(coords[i], coords[i-1]) / dt
                track_speeds.append(s)

                # direction en degrés (0° = x+, 90° = y+)
                angle = degrees(atan2(coords[i][1] - coords[i-1][1],
                                      coords[i][0] - coords[i-1][0]))
                track_directions.append(angle)

            # --------------------------------------------------
            # Agrégation globale
            if track_speeds:
                speeds.extend(track_speeds)
                directions.extend(track_directions)
                lifetimes.append(len(track_speeds))

                # --------------------------------------------------
                # Détection des arrêts brusques (sudden stop)
                for j in range(1, len(track_speeds)):
                    # Si la vitesse chute de plus de sudden_stop_threshold (ex. 50%)
                    if track_speeds[j] < track_speeds[j-1] * (1 - self.sudden_stop_threshold):
                        sudden_stops += 1

                # --------------------------------------------------
                # Détection de vitesse élevée proche machine
                if not machines.empty:
                    for s, pos in zip(track_speeds, coords[1:]):
                        if s > self.high_speed_threshold:
                            # Pour chaque machine, calculer la distance
                            for _, m in machines.iterrows():
                                machine_center = (m["bbox_x"] + m["bbox_w"]/2,
                                                  m["bbox_y"] + m["bbox_h"]/2)
                                dist = GeometryUtils.euclidean_distance(pos, machine_center)
                                # Si proche de la machine, compte l'événement
                                if dist < self.near_machine_distance:
                                    high_speed_near_machine += 1
                                    break  # éviter double comptage pour la même frame

        # --------------------------------------------------
        # Calcul des features statistiques
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        std_speed = float(np.std(speeds)) if speeds else 0.0
        direction_variance = float(np.var(directions)) if directions else 0.0
        track_lifetime_mean = float(np.mean(lifetimes)) if lifetimes else 0.0

        # Combinaison vitesse + direction pour détecter mouvement erratique
        erratic_motion_score = std_speed * direction_variance

        # --------------------------------------------------
        # Retour du dictionnaire de features
        return {
            "avg_person_speed": avg_speed,
            "std_person_speed": std_speed,
            "direction_variance": direction_variance,
            "track_lifetime_mean": track_lifetime_mean,
            "sudden_stop_events": sudden_stops,
            "erratic_motion_score": erratic_motion_score,
            "high_speed_near_machine": high_speed_near_machine,
        }

    # ======================================================
    def _empty_features(self) -> Dict[str, float]:
        """
        Retourne des features à 0 si aucune personne n'est détectée.
        """
        return {
            "avg_person_speed": 0.0,
            "std_person_speed": 0.0,
            "direction_variance": 0.0,
            "track_lifetime_mean": 0.0,
            "sudden_stop_events": 0,
            "erratic_motion_score": 0.0,
            "high_speed_near_machine": 0,
        }
