from datetime import timedelta


def compute_time_interval(last_detection_time, window_duration_sec):

    """
    Détermine l'intervalle d'analyse

    end_time   = dernière détection
    start_time = end_time - durée fenêtre
    """

    end_time = last_detection_time

    start_time = end_time - timedelta(seconds=window_duration_sec)

    return start_time, end_time