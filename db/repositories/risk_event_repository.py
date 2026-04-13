import pandas as pd
from db.connection import db_write

class RiskEventRepository:
    @staticmethod
    def save_event_and_alert(row):
        query = """
            INSERT INTO risk_events 
            (window_start, window_end, camera_id, activity_pred, activity_confidence, 
             risk_score, risk_level, risk_messages, zone_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Formatage sécurisé des dates
        def format_date(dt):
            if pd.isna(dt): return None
            try:
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return str(dt)

        params = (
            format_date(row.get('window_start')),
            format_date(row.get('window_end')),
            row.get('camera_id'),
            str(row.get('activity_pred')),
            float(row.get('activity_confidence', 0)),
            float(row.get('risk_score', 0.0)),
            row.get('risk_level'),
            row.get('risk_messages'),
            row.get('zone_id')
        )
        
        conn = db_write()
        cursor = None
        try:
            cursor = conn.cursor() # On crée le curseur normalement
            cursor.execute(query, params)
            conn.commit()
            print(f" [DB SUCCESS] Risque {row.get('risk_level')} enregistré pour Cam {row.get('camera_id')}")
        except Exception as e:
            print(f" [DB ERROR] Échec lors de l'insertion : {e}")
            if conn: conn.rollback()
        finally:
            if cursor: cursor.close() # On ferme manuellement
            if conn: conn.close()