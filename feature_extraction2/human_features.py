# feature_extraction/human_features.py

import pandas as pd
from typing import Dict


class HumanPresenceFeatures:
    """
    Extraction des caractéristiques liées à la présence humaine
    dans une fenêtre temporelle.
    """

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 stationary_threshold: float = 2.0):
        """
        Paramètres :
        image_width : largeur de l'image (pixels)
        image_height : hauteur de l'image (pixels)
        stationary_threshold : seuil de déplacement (pixels)
                               sous lequel une personne est considérée immobile
        """
        self.image_width = image_width
        self.image_height = image_height
        self.image_area = image_width * image_height
        self.stationary_threshold = stationary_threshold

    def extract(self, window_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extrait toutes les caractéristiques humaines
        d'une fenêtre glissante.
        """

        # Filtrer uniquement les personnes
        df = window_df[window_df["object_class"] == "person"]

        if df.empty:
            # Aucune personne détectée
            return {
                "avg_num_persons": 0.0,
                "max_num_persons": 0.0,
                "unique_person_tracks": 0,
                "person_density": 0.0,
                "person_presence_ratio": 0.0,
                "num_person_entries": 0,
                "stationary_person_ratio": 0.0,
                "crowding_score": 0.0,
                "avg_person_height":0.0
            }

        features = {}

        # =====================================================
        #  Moyenne de personnes par frame
        # =====================================================
        persons_per_frame = df.groupby("timestamp")["track_id"].nunique()
        features["avg_num_persons"] = persons_per_frame.mean()

        # =====================================================
        # 2 Pic de personnes
        # =====================================================
        features["max_num_persons"] = persons_per_frame.max()

        # =====================================================
        # 3 Nombre de personnes distinctes
        # =====================================================
        features["unique_person_tracks"] = df["track_id"].nunique()

        # =====================================================
        #  Densité humaine
        # personnes / surface image
        # =====================================================
        features["person_density"] = (
            features["avg_num_persons"] / self.image_area
            if self.image_area > 0 else 0.0
        )

        # =====================================================
        #  Ratio de présence humaine
        # % de frames avec au moins une personne
        # =====================================================
        total_frames = window_df["timestamp"].nunique()
        frames_with_person = persons_per_frame.count()

        features["person_presence_ratio"] = (
            frames_with_person / total_frames
            if total_frames > 0 else 0.0
        )

        # =====================================================
        #  Entrées / sorties rapides
        # (apparition ou disparition d'un track)
        # =====================================================
        track_counts = df.groupby("track_id")["timestamp"].count()
        features["num_person_entries"] = (track_counts == 1).sum()



        # ----------------------------------
        # Taille (proxy hauteur)
        # ----------------------------------
        avg_person_height = (
            df["bbox_h"].mean()
            if not df.empty and "bbox_h" in df.columns
            else 0.0
        )

        features["avg_person_height"] = float(avg_person_height)


        # =====================================================
        #  Ratio de personnes immobiles
        # =====================================================
        stationary_tracks = 0
        total_tracks = df["track_id"].nunique()

        for track_id, track_df in df.groupby("track_id"):
            if len(track_df) < 2:
                continue

            dx = track_df["bbox_x"].max() - track_df["bbox_x"].min()
            dy = track_df["bbox_y"].max() - track_df["bbox_y"].min()
            displacement = (dx ** 2 + dy ** 2) ** 0.5

            if displacement < self.stationary_threshold:
                stationary_tracks += 1

        features["stationary_person_ratio"] = (
            stationary_tracks / total_tracks
            if total_tracks > 0 else 0.0
        )

        # =====================================================
        #  Score de sur-densité (crowding)
        # Normalisé par un seuil empirique
        # =====================================================
        CROWDING_THRESHOLD = 5.0  # à ajuster selon le site
        features["crowding_score"] = min(
            features["avg_num_persons"] / CROWDING_THRESHOLD,
            1.0
        )

        return features
