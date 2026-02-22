# feature_extraction2/ml_feature_config.py

ML_FEATURE_COLUMNS = [

    # -----------------------------
    # Présence humaine
    # -----------------------------
    "avg_num_persons",
    "max_num_persons",
    "unique_person_tracks",
    "person_density",
    "person_presence_ratio",

    # -----------------------------
    # EPI
    # -----------------------------
    "avg_epi_compliance",
    "helmet_compliance_ratio",
    "vest_compliance_ratio",
    "num_no_helmet_events",
    "num_no_vest_events",

    # -----------------------------
    # Machines
    # -----------------------------
    "num_machines_avg",
    "machine_presence_ratio",
    "unique_machine_tracks",
    "co_presence_person_machine",

    # -----------------------------
    # Proximité
    # -----------------------------
    "dangerous_proximity_frames",
    "min_person_machine_distance",

    # -----------------------------
    # Dynamique temporelle
    # -----------------------------
    "avg_person_speed",
    "std_person_speed",
    "sudden_stop_events",
    "erratic_motion_score",
    "direction_variance",

    # -----------------------------
    # Zones (même si souvent à 0)
    # -----------------------------
    "num_persons_in_zone",
    "num_people_in_high_risk_zone",
    "proportion_time_in_high_risk_zone",
    "multiple_zone_exposure",

    "avg_person_height",
    "time_near_machine",
    "num_machine_interactions",
    "motion_consistency",
    "activity_persistence_score",
]
