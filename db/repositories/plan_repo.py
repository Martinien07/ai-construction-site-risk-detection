import pandas as pd
from db.connection import db_read


class PlanRepository:
    """
    Accès aux plans du site (RDC, R+1, sous-sol, etc.)
    """

    @staticmethod
    def get_plans_by_site(site_id):
        """
        Récupère tous les plans d’un site
        """

        query = """
        SELECT
            id,
            site_id,
            level,
            image_path,
            scale_factor
        FROM plans
        WHERE site_id = %s
        """

        conn = db_read()
        return pd.read_sql(query, conn, params=[site_id])
