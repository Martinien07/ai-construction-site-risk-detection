import numpy as np
from collections import defaultdict

class Tracker:
    def __init__(self, max_distance=50, max_age=10):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_age = max_age

    def _centroid(self, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def track(self, box_xyxy, class_name, frame_idx):
        centroid = self._centroid(box_xyxy)

        best_id = None
        best_dist = float("inf")

        for track_id, track in list(self.tracks.items()):
            if frame_idx - track["last_seen"] > self.max_age:
                del self.tracks[track_id]
                continue

            if track["class"] != class_name:
                continue

            dist = np.linalg.norm(centroid - track["centroid"])
            if dist < best_dist and dist < self.max_distance:
                best_dist = dist
                best_id = track_id

        if best_id is not None:
            self.tracks[best_id]["centroid"] = centroid
            self.tracks[best_id]["last_seen"] = frame_idx
            return best_id

        track_id = self.next_id
        self.tracks[track_id] = {
            "centroid": centroid,
            "class": class_name,
            "last_seen": frame_idx
        }
        self.next_id += 1
        return track_id
