from db.connection import db_write

class AlertRepository:
    @staticmethod
    def save_alert(alert_data):
        """
        alert_data est un dict contenant les clés correspondantes aux colonnes
        """
        query = """
        INSERT INTO alerts (
            risk_event_id, alert_level, status, 
            window_start_detection, window_end_detection, 
            camera_id, activity_pred, activity_confidence, 
            risk_score, recommendations
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            alert_data.get('risk_event_id'),
            alert_data.get('alert_level'),
            'new', # status par défaut
            alert_data.get('start_time'),
            alert_data.get('end_time'),
            alert_data.get('camera_id'),
            alert_data.get('activity_pred'),
            alert_data.get('activity_confidence'),
            alert_data.get('risk_score'),
            alert_data.get('recommendations')
        )
        
        conn = db_write()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
            conn.commit()
        finally:
            conn.close()