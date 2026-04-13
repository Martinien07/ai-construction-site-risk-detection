import pandas as pd
from db.connection import db_read
from datetime import datetime, timedelta

class DetectionRepository:
    """
    Accès aux détections YOLO stockées en base
    """

    @staticmethod
    def get_detections(camera_ids, start_time, end_time):
        """
        Récupère les détections pour un ensemble de caméras
        sur une période donnée
        """

        placeholders = ",".join(["%s"] * len(camera_ids))

        query = f"""
        SELECT
            id,
            camera_id,
            timestamp,
            object_class,
            track_id,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            confidence
        FROM detections
        WHERE camera_id IN ({placeholders})
          AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        """

        params = list(camera_ids) + [start_time, end_time]

        conn = None
        try:
            conn = db_read()
            df = pd.read_sql(query, conn, params=params)
            return df

        finally:
            # TRÈS IMPORTANT : libérer la connexion vraiment important pour assurer la libération du pool
            if conn is not None:
                conn.close()


                
    @staticmethod
    def get_last_detection_time(camera_ids):

        placeholders = ",".join(["%s"] * len(camera_ids))

        query = f"""
        SELECT MAX(timestamp) as last_ts
        FROM detections
        WHERE camera_id IN ({placeholders})
        """

        conn = None

        try:
            conn = db_read()
            df = pd.read_sql(query, conn, params=camera_ids)

            return df["last_ts"].iloc[0]

        finally:
            if conn is not None:
                conn.close()

    # Dans db/repositories/detection_repo.py

    @staticmethod
    def get_global_last_detection_time():
        """Récupère le timestamp le plus récent sur l'ensemble des caméras."""
        query = "SELECT MAX(timestamp) as last_ts FROM detections"
        
        conn = None
        try:
            conn = db_read()
            df = pd.read_sql(query, conn)
            # Gestion du cas où la table serait vide
            if df.empty or df["last_ts"].isnull().all():
                return datetime.now()
            return df["last_ts"].iloc[0]
        finally:
            if conn is not None:
                conn.close()