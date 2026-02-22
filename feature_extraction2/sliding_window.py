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

    def generate(self,
                 camera_ids: List[int],
                 start_time: datetime,
                 end_time: datetime) -> Iterator[pd.DataFrame]:
        """
        Génère les fenêtres glissantes en mémoire
        """

        # Chargement UNIQUE depuis la base
        df_all = self.detection_repository.get_detections(
            camera_ids=camera_ids,
            start_time=start_time,
            end_time=end_time
        )

        if df_all.empty:
            return

        # Conversion en datetime (sécurité)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

        current_start = start_time
        window_delta = timedelta(seconds=self.window_duration)
        step_delta = timedelta(seconds=self.step)

        # Fenêtres glissantes en RAM
        while current_start + window_delta <= end_time:
            current_end = current_start + window_delta

            mask = (
                (df_all["timestamp"] >= current_start) &
                (df_all["timestamp"] < current_end)
            )

            yield df_all.loc[mask]

            current_start += step_delta
