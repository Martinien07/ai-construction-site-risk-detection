import cv2
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from inference.tracking import Tracker
from inference.filters import get_confidence_threshold
from inference.persistence import save_detections_batch


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_yolo11x.pt"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEBUG_MODE = True

# DEBUG ALLÉGÉ
DEBUG_DRAW_EVERY = 1
DEBUG_PRINT_EVERY = 1
DB_BATCH_SIZE = 1


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
# INFÉRENCE VIDÉO
# ============================================================

def run_video(
    video_path: str,
    site_id: int,
    camera_id: int,
    model_path: str | Path | None = None
):
    """
    Inférence vidéo – DEBUG ALLÉGÉ :
    - ~6 fps
    - YOLO detection
    - seuil dynamique
    - tracking centroid
    - batch DB optimisé
    - debug visuel & logs limités
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
    frame_step = max(int(fps / 5), 1)
    frame_idx = 0

    print(f"[INFO] FPS={fps} | frame_step={frame_step}")

    db_buffer = []
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        processed_frames += 1
        draw_debug = DEBUG_MODE and (processed_frames % DEBUG_DRAW_EVERY == 0)
        print_debug = DEBUG_MODE and (processed_frames % DEBUG_PRINT_EVERY == 0)

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
                threshold=0.1

                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = bbox

                # ----------------------------
                # Filtrage seuil
                # ----------------------------
                if confidence < threshold:
                    if print_debug:
                        print(
                            f"[REJECT] cam={camera_id} "
                            f"{class_name} {confidence:.2f} < {threshold}"
                        )
                    continue

                # ----------------------------
                # Détection acceptée
                # ----------------------------
                track_id = tracker.track(bbox, class_name,frame_idx)

                if print_debug:
                    print(
                        f"[ACCEPT] cam={camera_id} "
                        f"{class_name} #{track_id} "
                        f"{confidence:.2f}"
                    )

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
                # Annotation DEBUG (rare)
                # ----------------------------
                if draw_debug:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"{class_name} #{track_id}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

        # ----------------------------
        # Batch DB
        # ----------------------------
        if len(db_buffer) >= DB_BATCH_SIZE:
            save_detections_batch(db_buffer)
            db_buffer.clear()

        # ----------------------------
        # Sauvegarde frame DEBUG
        # ----------------------------
        if draw_debug:
            cv2.putText(
                frame,
                "DEBUG MODE",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            out_path = OUTPUT_DIR / f"frame_{frame_idx}.jpg"
            cv2.imwrite(str(out_path), frame)

        frame_idx += 1

    # Flush DB final
    if db_buffer:
        save_detections_batch(db_buffer)

    cap.release()
    print("[INFO] Fin de l'inférence vidéo")
