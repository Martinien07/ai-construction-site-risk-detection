import json
from config.confidence_loader import load_default_conf
from db.connection import db_read


def _load_confidence_from_db(site_id: int, camera_id: int):
    """
    Récupère la configuration de seuil depuis la base de données.
    Priorité : camera > site
    """
    db = db_read()
    cursor = db.cursor(dictionary=True)

    # 1. Configuration caméra
    cursor.execute(
        "SELECT confidence_config FROM cameras WHERE id = %s",
        (camera_id,)
    )
    row = cursor.fetchone()
    if row and row["confidence_config"]:
        try:
            return json.loads(row["confidence_config"])
        except Exception:
            pass

    # 2. Configuration site
    cursor.execute(
        "SELECT confidence_config FROM sites WHERE id = %s",
        (site_id,)
    )
    row = cursor.fetchone()
    if row and row["confidence_config"]:
        try:
            return json.loads(row["confidence_config"])
        except Exception:
            pass

    cursor.close()
    db.close()
    return None


def get_confidence_threshold(
    class_name: str,
    site_id: int,
    camera_id: int
) -> float:
    """
    Retourne le seuil de confiance pour une classe donnée.
    Ordre :
    1. DB caméra
    2. DB site
    3. YAML par défaut
    """

    # --- DB ---
    db_conf = _load_confidence_from_db(site_id, camera_id)
    if isinstance(db_conf, dict) and class_name in db_conf:
        return float(db_conf[class_name])

    # --- YAML fallback ---
    default_conf = load_default_conf()
    return float(default_conf["default"].get(class_name, 0.4))
