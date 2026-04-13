import pandas as pd
from db.connection import db_read

class CameraRepository:
    """
    Accès optimisé aux caméras installées sur les plans
    """

    @staticmethod
    def get_cameras_by_site(site_id):
        """
        Récupère toutes les caméras d’un site avec gestion sécurisée du pool
        """
        query = """
        SELECT 
            c.id, c.plan_id, c.name, c.stream_url, 
            c.x_plan, c.y_plan, c.orientation, c.fov 
        FROM cameras c
        JOIN plans p ON p.id = c.plan_id
        WHERE p.site_id = %s
        """
        conn = None
        try:
            conn = db_read()
            return pd.read_sql(query, conn, params=[site_id])
        finally:
            if conn:
                conn.close()

    @staticmethod
    def get_camera_config(camera_id):
        """
        Récupère la configuration complète d'une caméra spécifique (dont son plan_id)
        """
        query = "SELECT * FROM cameras WHERE id = %s"
        conn = None
        try:
            conn = db_read()
            df = pd.read_sql(query, conn, params=[camera_id])
            return df.iloc[0] if not df.empty else None
        finally:
            if conn:
                conn.close()