# features/zones.py

from typing import Tuple, List


class ZoneUtils:
    """
    Classe utilitaire pour le raisonnement spatial :
    vérification de l'appartenance d'un point à une zone définie par un polygone.
    """

    @staticmethod
    def point_in_polygon(point: Tuple[float, float],
                         polygon: List[Tuple[float, float]]) -> bool:
        """
        Vérifie si un point appartient à un polygone (zone).

        Méthode :
        Algorithme du rayon (Ray Casting).

        Paramètres :
        point : (x, y) coordonnées sur le plan
        polygon : liste de points [(x1, y1), (x2, y2), ...]

        Retour :
        True si le point est dans la zone, False sinon
        """
        x, y = point
        inside = False
        n = len(polygon)

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            # Test de croisement du rayon horizontal
            if ((y1 > y) != (y2 > y)):
                x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
                if x < x_intersect:
                    inside = not inside

        return inside
