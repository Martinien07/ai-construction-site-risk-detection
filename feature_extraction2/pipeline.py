# ==========================================================
# feature_extraction2/pipeline.py
# ==========================================================

import pandas as pd
from typing import Tuple, List
from datetime import datetime

from feature_extraction2.ml_feature_config import ML_FEATURE_COLUMNS



def extract_features_pipeline(
    sliding_window,
    human_extractor,
    epi_extractor,
    machine_extractor,
    proximity_extractor,
    temporal_extractor,
    zone_extractor,
    stability_extractor=None,
    transition_extractor=None,
    camera_ids: List[int] = None,
    start_time: datetime = None,
    end_time: datetime = None,
    max_windows: int | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline principal d'extraction des features.

    Retourne :
    ---------
    df_all_features :
        Toutes les features extraites (audit / debug / analyse)

    df_ml_features :
        Sous-ensemble strict des features utilisées
        par les modèles ML (XGBoost, RF, etc.)
    """

    all_rows = []
    previous_rows = []  # mémoire inter-fenêtres

    # ======================================================
    # BOUCLE FENÊTRE GLISSANTE
    # ======================================================
    for i, window_df in enumerate(
        sliding_window.generate(
            camera_ids=camera_ids,
            start_time=start_time,
            end_time=end_time
        )
    ):
        if max_windows is not None and i >= max_windows:
            break

        if window_df.empty:
            continue

        # --------------------------------------------------
        # MÉTADONNÉES DE FENÊTRE
        # --------------------------------------------------
        row = {
            "window_index": i,
            "window_start": window_df["timestamp"].min(),
            "window_end": window_df["timestamp"].max(),
            "camera_id": window_df["camera_id"].iloc[0],
            "num_detections": len(window_df),
        }

        # --------------------------------------------------
        # FEATURES INSTANTANÉES / FENÊTRE
        # --------------------------------------------------
        row.update(human_extractor.extract(window_df))
        row.update(epi_extractor.extract(window_df))
        row.update(machine_extractor.extract(window_df))
        row.update(proximity_extractor.extract(window_df))
        row.update(temporal_extractor.extract(window_df))
        row.update(zone_extractor.extract(window_df))


        # --------------------------------------------------
        # FEATURES DE STABILITÉ (INTER-FENÊTRES)
        # --------------------------------------------------
        if stability_extractor is not None:
            stability_features = stability_extractor.extract(row)
            row.update(stability_features)

        # --------------------------------------------------
        # FEATURES DE TRANSITION (INTER-FENÊTRES)
        # --------------------------------------------------
        if transition_extractor is not None:
            transition_features = transition_extractor.extract(
                current_row=row,
                history=previous_rows
            )
            row.update(transition_features)

        # --------------------------------------------------
        # MÉMOIRE TEMPORELLE
        # --------------------------------------------------
        previous_rows.append(row)
        all_rows.append(row)

    # ======================================================
    # DATAFRAME COMPLET
    # ======================================================
    df_all_features = pd.DataFrame(all_rows)

    if df_all_features.empty:
        return df_all_features, df_all_features

    # ======================================================
    # DATAFRAME ML (SÉLECTION STRICTE)
    # ======================================================
    df_ml_features = (
        df_all_features
        .reindex(columns=ML_FEATURE_COLUMNS)
        .fillna(0)
    )

    return df_all_features, df_ml_features
