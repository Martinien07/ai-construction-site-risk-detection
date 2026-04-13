from db.connection import db_read

class CameraDiscoveryRepository:
    @staticmethod
    def get_site_cameras(site_id):
        query = """
            SELECT 
                c.id as camera_id, 
                c.name as camera_name,
                c.stream_url,
                c.is_webcam,            -- AJOUTÉ ICI
                p.level as plan_level
            FROM cameras c
            JOIN plans p ON c.plan_id = p.id
            WHERE p.site_id = %s
        """
        conn = db_read()
        try:
            # dictionary=True transforme chaque ligne en dict Python
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, (site_id,))
            return cursor.fetchall()
        finally:
            conn.close()