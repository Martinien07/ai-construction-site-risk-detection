import yaml
from pathlib import Path


CONF_FILE = Path(__file__).parent / "confidence_config.yaml"


def load_default_conf() -> dict:
    """
    Charge la configuration de seuils par défaut depuis confidence_config.yaml.

    Retourne toujours un dictionnaire valide.
    En cas d'erreur, retourne une config minimale sûre.
    """

    if not CONF_FILE.exists():
        return {
            "default": {}
        }

    try:
        with open(CONF_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("Invalid YAML structure")

        if "default" not in data or not isinstance(data["default"], dict):
            raise ValueError("Missing 'default' section")

        # Validation des valeurs
        clean_conf = {}
        for cls, value in data["default"].items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                clean_conf[cls] = float(value)

        return {
            "default": clean_conf
        }

    except Exception:
        # Fallback sécurisé
        return {
            "default": {}
        }
