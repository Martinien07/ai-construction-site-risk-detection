from datetime import datetime, timedelta
import pandas as pd
from db.repositories.detection_repo import DetectionRepository

class Analyzer:
    def __init__(self):
        # On garde une trace de là où on s'est arrêté
        self.last_processed_timestamp = None

    def get_last_detection_timestamp(self):
            """Récupère la borne temporelle en objet datetime natif."""
            if self.last_processed_timestamp:
                return self.last_processed_timestamp
                
            # Appel du repository
            last_in_db = DetectionRepository.get_global_last_detection_time()
            
            # Si c'est un Timestamp pandas, on le convertit en datetime natif
            if isinstance(last_in_db, pd.Timestamp):
                return last_in_db.to_pydatetime()
                
            return last_in_db if last_in_db else datetime.now()
                

    def fetch_recent_data(self, start_time, end_time, camera_ids=None):
        """Délègue l'extraction en utilisant le repository."""
        # On peut laisser camera_ids optionnel pour une flexibilité totale
        return DetectionRepository.get_detections(camera_ids, start_time, end_time)

    def update_last_timestamp(self, df):
        """Actualise le curseur de temps après un traitement réussi."""
        if not df.empty and 'timestamp' in df.columns:
            self.last_processed_timestamp = pd.to_datetime(df['timestamp']).max()