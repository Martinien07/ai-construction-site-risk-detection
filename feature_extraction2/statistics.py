# features/statistics.py

import numpy as np
from typing import List


class StatsUtils:
    """
    Classe utilitaire pour le calcul de statistiques simples
    avec gestion des cas vides (robustesse).
    """

    @staticmethod
    def mean(values: List[float]) -> float:
        """
        Calcule la moyenne d'une liste de valeurs.

        Retour :
        0 si la liste est vide
        """
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def std(values: List[float]) -> float:
        """
        Calcule l'écart-type d'une liste de valeurs.

        Retour :
        0 si la liste est vide
        """
        return float(np.std(values)) if values else 0.0

    @staticmethod
    def minimum(values: List[float]) -> float:
        """
        Calcule la valeur minimale d'une liste.

        Retour :
        0 si la liste est vide
        """
        return float(min(values)) if values else 0.0

    @staticmethod
    def maximum(values: List[float]) -> float:
        """
        Calcule la valeur maximale d'une liste.

        Retour :
        0 si la liste est vide
        """
        return float(max(values)) if values else 0.0

    @staticmethod
    def ratio(numerator: float, denominator: float) -> float:
        """
        Calcule un ratio de manière sécurisée.

        Retour :
        0 si le dénominateur est nul
        """
        return numerator / denominator if denominator > 0 else 0.0
