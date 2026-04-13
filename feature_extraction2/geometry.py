# features/geometry.py

import math
from typing import Tuple, Dict


class GeometryUtils:
    """
    Classe utilitaire regroupant toutes les fonctions géométriques
    utilisées pour manipuler des bounding boxes et des points 2D.
    """

    @staticmethod
    def bbox_center(bbox: Dict) -> Tuple[float, float]:
        """
        Calcule le centre (x, y) d'une bounding box.

        Paramètres :
        bbox : dictionnaire contenant
               - bbox_x : coordonnée x du coin supérieur gauche
               - bbox_y : coordonnée y du coin supérieur gauche
               - bbox_w : largeur
               - bbox_h : hauteur

        Retour :
        (cx, cy) : coordonnées du centre de la bounding box
        """
        cx = bbox["bbox_x"] + bbox["bbox_w"] / 2
        cy = bbox["bbox_y"] + bbox["bbox_h"] / 2
        return cx, cy

    @staticmethod
    def bbox_area(bbox: Dict) -> float:
        """
        Calcule la surface d'une bounding box.

        Retour :
        Surface en pixels
        """
        return bbox["bbox_w"] * bbox["bbox_h"]

    @staticmethod
    def iou_bbox(bbox1: Dict, bbox2: Dict) -> float:
        """
        Calcule l'Intersection over Union (IoU) entre deux bounding boxes.
        L'IoU est utilisé pour mesurer le chevauchement entre deux objets.

        Retour :
        Valeur comprise entre 0 et 1
        """
        x1 = max(bbox1["bbox_x"], bbox2["bbox_x"])
        y1 = max(bbox1["bbox_y"], bbox2["bbox_y"])
        x2 = min(bbox1["bbox_x"] + bbox1["bbox_w"],
                 bbox2["bbox_x"] + bbox2["bbox_w"])
        y2 = min(bbox1["bbox_y"] + bbox1["bbox_h"],
                 bbox2["bbox_y"] + bbox2["bbox_h"])

        # Dimensions de l'intersection
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        intersection = inter_w * inter_h

        # Surface totale
        union = (
            GeometryUtils.bbox_area(bbox1)
            + GeometryUtils.bbox_area(bbox2)
            - intersection
        )

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float],
                           p2: Tuple[float, float]) -> float:
        """
        Calcule la distance euclidienne entre deux points 2D.

        Paramètres :
        p1, p2 : tuples (x, y)

        Retour :
        Distance en pixels
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 +
                         (p1[1] - p2[1]) ** 2)

    @staticmethod
    def speed(p1: Tuple[float, float],
              p2: Tuple[float, float],
              delta_t: float) -> float:
        """
        Calcule la vitesse entre deux positions successives.

        Paramètres :
        p1, p2 : positions successives
        delta_t : intervalle de temps en secondes

        Retour :
        Vitesse (pixels / seconde)
        """
        if delta_t <= 0:
            return 0.0

        return GeometryUtils.euclidean_distance(p1, p2) / delta_t
