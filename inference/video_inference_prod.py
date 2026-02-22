import cv2
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from inference.tracking import Tracker
from inference.filters import get_confidence_threshold
from inference.persistence import save_detections_batch





# ============================================================
# CONFIGURATION PRODUCTION
# ============================================================

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_yolo11x.pt"
)

TARGET_FPS = 5
DB_BATCH_SIZE = 20




# pour normaliser la sortie du modèle avant l'enrégistrement dans la base de données 

def normalize_class_name(class_name: str) -> str:
    """
    Normalise le nom de classe pour garantir la consistance :
    - Tout en minuscules
    - Remplace les espaces/soulignés/tirets par des soulignés
    - Supprime les caractères spéciaux
    """
    # Convertir en minuscules
    normalized = class_name.lower()
    
    # Remplacer espaces et tirets par soulignés
    normalized = normalized.replace(" ", "_").replace("-", "_")
    
    # Supprimer les caractères spéciaux (garder seulement alphanumérique et _)
    normalized = ''.join(c for c in normalized if c.isalnum() or c == '_')
    
    # Éviter les soulignés multiples
    normalized = '_'.join(filter(None, normalized.split('_')))
    
    return normalized


# ============================================================
# INFÉRENCE VIDÉO – PRODUCTION STRICTE
# ============================================================

def run_video_production(
    video_path: str,
    site_id: int,
    camera_id: int,
    model_path: str | Path | None = None
):
    """
    Inférence vidéo PRODUCTION :
    - échantillonnage stable
    - détection YOLO
    - seuil dynamique (DB → YAML)
    - tracking centroid
    - sauvegarde batch DB
    - AUCUN debug / affichage
    """

    # ----------------------------
    # Modèle
    # ----------------------------
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    model = YOLO(str(model_path))

    # ----------------------------
    # Tracker
    # ----------------------------
    tracker = Tracker()

    # ----------------------------
    # Vidéo
    # ----------------------------
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = max(int(fps / TARGET_FPS), 1)
    frame_idx = 0

    db_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                raw_class_name = r.names[int(box.cls)]
                class_name = normalize_class_name(raw_class_name)


                confidence = float(box.conf)

                threshold = get_confidence_threshold(
                    class_name=class_name,
                    site_id=site_id,
                    camera_id=camera_id
                )

                if confidence < threshold:
                    continue

                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = bbox

                track_id = tracker.track(bbox, class_name,frame_idx)

                db_buffer.append({
                    "site_id": site_id,
                    "camera_id": camera_id,
                    "timestamp": datetime.now(),
                    "object_class": class_name,
                    "confidence": confidence,
                    "bbox_x": x1,
                    "bbox_y": y1,
                    "bbox_w": x2 - x1,
                    "bbox_h": y2 - y1,
                    "track_id": track_id
                })

        # ----------------------------
        # Batch DB
        # ----------------------------
        if len(db_buffer) >= DB_BATCH_SIZE:
            save_detections_batch(db_buffer)
            db_buffer.clear()

        frame_idx += 1

    # Flush final
    if db_buffer:
        save_detections_batch(db_buffer)

    cap.release()
