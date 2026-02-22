# features/homography.py

import numpy as np
from typing import Tuple


class HomographyUtils:
    """
    Outils liés aux transformations homographiques
    image → plan
    """

    @staticmethod
    def apply_homography(
        x: float,
        y: float,
        H: np.ndarray
    ) -> Tuple[float, float]:
        """
        Applique une homographie à un point image.

        Paramètres :
        x, y : coordonnées image (pixels)
        H : matrice 3x3 d'homographie

        Retour :
        (X, Y) : coordonnées dans le plan
        """
        point_img = np.array([x, y, 1.0])
        point_plan = H @ point_img

        if point_plan[2] == 0:
            return 0.0, 0.0

        X = point_plan[0] / point_plan[2]
        Y = point_plan[1] / point_plan[2]

        return float(X), float(Y)
