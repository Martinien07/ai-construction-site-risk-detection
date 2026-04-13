# features/sliding_window.py

from typing import Iterator, List
from datetime import datetime, timedelta
import pandas as pd


class SlidingWindow:
    """
    Fenêtre glissante temporelle optimisée
    (1 seule requête DB, fenêtres en mémoire)
    """

    def __init__(self,
                 detection_repository,
                 window_duration: float,
                 step: float):
        """
        Paramètres :
        detection_repository : instance de DetectionRepository
        window_duration : durée de la fenêtre (en secondes)
        step : pas de glissement (en secondes)
        """
        self.detection_repository = detection_repository
        self.window_duration = window_duration
        self.step = step

    def generate(self, camera_ids: List[int], start_time: datetime, end_time: datetime) -> Iterator[pd.DataFrame]:
            
            # 1. Chargement global
            df_all = self.detection_repository.get_detections(
                camera_ids=camera_ids, start_time=start_time, end_time=end_time
            )

            if df_all.empty:
                return

            df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

            # 2. Boucle par caméra
            for cam_id in camera_ids:
                df_cam = df_all[df_all["camera_id"] == cam_id]
                
                if df_cam.empty:
                    continue

                current_start = start_time
                window_delta = timedelta(seconds=self.window_duration)
                step_delta = timedelta(seconds=self.step)

                # 3. Fenêtrage spécifique à la caméra
                while current_start + window_delta <= end_time:
                    current_end = current_start + window_delta
                    
                    mask = (df_cam["timestamp"] >= current_start) & (df_cam["timestamp"] < current_end)
                    window_df = df_cam.loc[mask]
                    
                    if not window_df.empty:
                        yield window_df, current_start, current_end
                    
                    current_start += step_delta
