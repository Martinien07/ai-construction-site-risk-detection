import json
import numpy as np
from db.connection import get_read_connection

class CameraCalibrationRepository:
    """
    Accès optimisé aux calibrations (homographies)
    """

class CameraCalibrationRepository:
    def __init__(self):
        # Le cache vit à l'intérieur de l'instance
        self._cache = {}

    def get_active_calibration(self, camera_id: int, plan_id: int):
        key = (camera_id, plan_id)
        
        # 1. Vérification du cache
        if key in self._cache:
            return self._cache[key]
        
        # 2. Si pas en cache, on appelle la base de données
        H = self._fetch_from_db(camera_id, plan_id)
        
        # 3. On stocke dans le cache
        if H is not None:
            self._cache[key] = H
        return H
    def _fetch_from_db(self, camera_id, plan_id):
            conn = None
            cursor = None
            row = None
            
            try:
                conn = get_read_connection()
                # On crée le curseur manuellement
                cursor = conn.cursor(dictionary=True)
                
                query = """
                    SELECT homography
                    FROM camera_calibrations
                    WHERE camera_id = %s
                    AND plan_id = %s
                    AND is_active = 1
                    LIMIT 1
                """
                cursor.execute(query, (int(camera_id), int(plan_id)))
                row = cursor.fetchone()
                
            finally:
                # On ferme manuellement le curseur ET la connexion
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            if not row or not row["homography"]:
                return None

            try:
                return np.array(json.loads(row["homography"]))
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Erreur de parsing pour Cam {camera_id}: {e}")
                return None