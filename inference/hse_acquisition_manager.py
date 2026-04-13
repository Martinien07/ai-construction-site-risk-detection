import cv2
import threading
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from inference.persistence import save_detections_batch

class HSEAcquisitionManager:
    def __init__(self, site_id, model_path="../models/yolo26x.pt"):
        self.site_id = site_id
        self.model = YOLO(model_path)
        self.caps = {}      
        self.recorders = {} 
        self.recorder_start_times = {}
        # ÉLÉMENT CLÉ : On crée un dictionnaire pour partager les images
        self.shared_frames = {} 
        self.is_running = False
        self._discover_and_connect()

    def _get_source_string(self, cam):
        cam_name = cam.get('camera_name', 'Caméra Inconnue')
        if cam.get('is_webcam', 0) == 1:
            print(f" [OK] Source Webcam pour : {cam_name}")
            return 0 
        return cam.get('stream_url')

    def _setup_recorder(self, cam_id, cap):
        if cam_id in self.recorders:
            self.recorders[cam_id].release()
        out_dir = Path(f"data/recordings/cam_{cam_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = str(out_dir / f"{timestamp}.avi")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.recorders[cam_id] = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        self.recorder_start_times[cam_id] = time.time()
        print(f" Vidéo démarrée : {filename} ({w}x{h})")

    def _discover_and_connect(self):
        from db.repositories.camera_discovery_repository import CameraDiscoveryRepository
        raw_cameras = CameraDiscoveryRepository.get_site_cameras(self.site_id)
        for cam in raw_cameras:
            source = self._get_source_string(cam)
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                self.caps[cam['camera_id']] = cap
                self._setup_recorder(cam['camera_id'], cap)

    def start_dual_flux_pipeline(self):
        self.is_running = True
        self.inference_thread = threading.Thread(target=self._run_batch_inference_loop)
        self.record_thread = threading.Thread(target=self._run_recording_and_view_loop)
        
        # On lance l'IA d'abord (qui va lire les images)
        self.inference_thread.start()
        # On lance l'enregistrement ensuite
        self.record_thread.start()

    def _run_batch_inference_loop(self):
        """THREAD PRODUCTEUR : C'est le SEUL qui fait cap.read()"""
        while self.is_running:
            batch_frames = []
            batch_ids = []
            for cam_id, cap in list(self.caps.items()):
                ret, frame = cap.read()
                if ret:
                    # On stocke l'image pour que l'autre thread y accède
                    self.shared_frames[cam_id] = frame
                    batch_frames.append(frame)
                    batch_ids.append(cam_id)

            if batch_frames:
                results = self.model(batch_frames, verbose=False)
                for i, r in enumerate(results):
                    self._dispatch_results(r, batch_ids[i])
            time.sleep(0.01)

    def _run_recording_and_view_loop(self):
        """THREAD CONSOMMATEUR : Utilise les images de shared_frames"""
        while self.is_running:
            for cam_id in list(self.caps.keys()):
                # On récupère l'image sans bloquer le flux matériel
                frame = self.shared_frames.get(cam_id)
                if frame is not None:
                    # 1. Rotation 15min
                    if time.time() - self.recorder_start_times[cam_id] > 900:
                        self._setup_recorder(cam_id, self.caps[cam_id])
                    # 2. Enregistrement
                    self.recorders[cam_id].write(frame)
                    # 3. Affichage
                    if cam_id == 11 or cam_id == 1:
                        cv2.imshow(f"Monitor Caméra {cam_id}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break
            time.sleep(0.03) # Limite à ~30 FPS pour soulager le CPU

    def _dispatch_results(self, result, cam_id):
        """Transforme les résultats YOLO en format DB et les sauvegarde."""
        db_buffer = []
        current_ts = datetime.now()
        
        for box in result.boxes:
            # Récupération des données YOLO
            class_id = int(box.cls)
            class_name = result.names[class_id] # Nom de l'objet (person, helmet, etc.)
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist() # [x1, y1, x2, y2]

            # On prépare le dictionnaire pour ta fonction save_detections_batch
            db_buffer.append({
                "camera_id": cam_id,
                "timestamp": current_ts,
                "object_class": class_name,
                "confidence": conf,
                "bbox_x": bbox[0],
                "bbox_y": bbox[1],
                "bbox_w": bbox[2] - bbox[0],
                "bbox_h": bbox[3] - bbox[1],
                "track_id": 0 # À remplacer par ton tracker si tu l'utilises
            })

        # Sauvegarde effective en base de données
        if db_buffer:
            try:
                save_detections_batch(db_buffer)
            except Exception as e:
                print(f" Erreur DB sur Cam {cam_id}: {e}")
        
        """Traitement des résultats par caméra."""
        # Pour le moment, simple log. Ici tu feras appel à ton DecisionEngine
        nb_objets = len(result.boxes)
        if nb_objets > 0:
            print(f" IA Cam {cam_id}: {nb_objets} objets détectés")


    def stop(self):
        self.is_running = False
        cv2.destroyAllWindows()

        