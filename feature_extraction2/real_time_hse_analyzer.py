import time
import pandas as pd
from datetime import datetime, timedelta

from feature_extraction2.pipeline import extract_features_pipeline
from db.repositories.detection_repo import DetectionRepository


class RealTimeHSEAnalyzer:

    def __init__(
        self,
        sliding_window,
        human_extractor,
        epi_extractor,
        machine_extractor,
        proximity_extractor,
        temporal_extractor,
        zone_extractor,
        stability_extractor,
        camera_ids=[1],
        analysis_duration_sec=300,
        fetch_interval_sec=1,
        max_windows=None
    ):

        """
        Classe principale d'analyse HSE temps réel
        """

        self.sliding_window = sliding_window

        self.human_extractor = human_extractor
        self.epi_extractor = epi_extractor
        self.machine_extractor = machine_extractor
        self.proximity_extractor = proximity_extractor
        self.temporal_extractor = temporal_extractor
        self.zone_extractor = zone_extractor
        self.stability_extractor = stability_extractor

        self.camera_ids = camera_ids

        self.analysis_duration = timedelta(seconds=analysis_duration_sec)

        self.fetch_interval = fetch_interval_sec

        self.max_windows = max_windows

        self.detection_repo = DetectionRepository()

    # ======================================================
    # récupérer timestamp dernière détection
    # ======================================================

    def get_last_detection_time(self):

        last_ts = self.detection_repo.get_last_detection_time(self.camera_ids)

        print("\nDEBUG LAST DETECTION")
        print("Timestamp récupéré :", last_ts)

        if last_ts is None:
            print("Aucune détection trouvée")
            return None

        return last_ts

    # ======================================================
    # EXTRACTION FEATURES VIA PIPELINE
    # ======================================================

    def extract_features(self, start_time, end_time):

        print("\n EXTRACTION FEATURES")
        print("Start :", start_time)
        print("End   :", end_time)

        df_all, df_ml = extract_features_pipeline(

            sliding_window=self.sliding_window,

            human_extractor=self.human_extractor,
            epi_extractor=self.epi_extractor,
            machine_extractor=self.machine_extractor,
            proximity_extractor=self.proximity_extractor,
            temporal_extractor=self.temporal_extractor,
            zone_extractor=self.zone_extractor,
            stability_extractor=self.stability_extractor,

            camera_ids=self.camera_ids,
            start_time=start_time,
            end_time=end_time,

            max_windows=self.max_windows
        )

        df_all = pd.DataFrame(df_all)
        df_ml = pd.DataFrame(df_ml)

        print("\n FEATURES EXTRAITES")
        print("Nombre fenêtres :", len(df_all))

        print("\n DF_ALL")
        print(df_all.head())

        print("\n DF_ML")
        print(df_ml.head())

        return df_all, df_ml

    # ======================================================
    # ANALYSE UNE FOIS
    # ======================================================

    def run_once(self):
            print("\n============================")
            print("ANALYSE HSE")
            print("============================")

            last_time = self.get_last_detection_time()

            if last_time is None:
                print("Aucune détection trouvée")
                return None, None

            # Calcul du start_time
            start_time = last_time - self.analysis_duration

            # --- CORRECTION ICI ---
            # On force la conversion en datetime natif Python 
            # pour éviter l'erreur de conversion MySQL
            if hasattr(start_time, 'to_pydatetime'):
                start_time = start_time.to_pydatetime()
            if hasattr(last_time, 'to_pydatetime'):
                last_time = last_time.to_pydatetime()
            # ----------------------

            df_all, df_ml = self.extract_features(start_time, last_time)

            return df_all, df_ml

    # ======================================================
    # BOUCLE TEMPS REEL
    # ======================================================

    def run_realtime(self):

        print("\n DEMARRAGE SURVEILLANCE TEMPS REEL")

        while True:

            start = time.time()

            self.run_once()

            elapsed = time.time() - start

            sleep_time = max(0, self.fetch_interval - elapsed)

            time.sleep(sleep_time)