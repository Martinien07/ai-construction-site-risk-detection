import pandas as pd
from typing import Dict


class MachineVehicleFeatures:
    """
    Extraction des caractéristiques liées aux machines et véhicules
    sur une fenêtre glissante temporelle.
    """

    def __init__(self, window_duration: float):
        """
        window_duration : durée d'une fenêtre (en secondes)
        """
        self.window_duration = window_duration

    def extract(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extrait les features machines / véhicules à partir d'une fenêtre.

        df : DataFrame contenant les détections de la fenêtre
        """

        if df.empty:
            return self._empty_features()

        # Séparation par type
        persons = df[df["object_class"] == "person"]
        vehicles = df[df["object_class"] == "vehicle"]
        machines = df[df["object_class"] == "machinery"]

        # Frames uniques
        frames = df["timestamp"].unique()
        num_frames = len(frames)

        if num_frames == 0:
            return self._empty_features()

        # ============================
        # Comptage par frame
        # ============================
        machines_per_frame = []
        machine_presence_frames = 0
        co_presence_frames = 0
        dominance_frames = 0
        machine_only_frames = 0

        for ts in frames:
            persons_f = persons[persons["timestamp"] == ts]["track_id"].nunique()
            machines_f = (
                vehicles[vehicles["timestamp"] == ts]["track_id"].nunique()
                + machines[machines["timestamp"] == ts]["track_id"].nunique()
            )

            machines_per_frame.append(machines_f)

            if machines_f > 0:
                machine_presence_frames += 1

            if machines_f > 0 and persons_f > 0:
                co_presence_frames += 1

            if machines_f > persons_f:
                dominance_frames += 1

            if machines_f > 0 and persons_f == 0:
                machine_only_frames += 1

        # ============================
        # Calculs globaux
        # ============================
        num_machines_avg = sum(machines_per_frame) / num_frames
        machine_presence_ratio = machine_presence_frames / num_frames
        co_presence_ratio = co_presence_frames / num_frames
        dominance_ratio = dominance_frames / num_frames

        machine_only_duration = (
            machine_only_frames / num_frames
        ) * self.window_duration

        # Machines distinctes
        unique_machine_tracks = pd.concat(
            [vehicles["track_id"], machines["track_id"]]
        ).nunique()

        # Ratio type machine
        num_vehicles = vehicles["track_id"].nunique()
        num_machinery = machines["track_id"].nunique()

        if (num_vehicles + num_machinery) > 0:
            machine_type_ratio = num_vehicles / (num_vehicles + num_machinery)
        else:
            machine_type_ratio = 0.0

        return {
            "num_machines_avg": num_machines_avg,
            "machine_presence_ratio": machine_presence_ratio,
            "unique_machine_tracks": unique_machine_tracks,
            "machine_type_ratio": machine_type_ratio,
            "co_presence_person_machine": co_presence_ratio,
            "machine_dominance_ratio": dominance_ratio,
            "machine_only_duration": machine_only_duration,
        }

    def _empty_features(self) -> Dict[str, float]:
        """
        Valeurs par défaut si aucune détection
        """
        return {
            "num_machines_avg": 0.0,
            "machine_presence_ratio": 0.0,
            "unique_machine_tracks": 0,
            "machine_type_ratio": 0.0,
            "co_presence_person_machine": 0.0,
            "machine_dominance_ratio": 0.0,
            "machine_only_duration": 0.0,
        }
