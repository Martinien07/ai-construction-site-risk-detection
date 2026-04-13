# db/repositories/zone_repo.py

import json
from typing import List, Dict
from db.connection import db_read


class ZoneRepository:
    """
    Accès aux zones de risque définies sur les plans.

    Une zone est définie dans le référentiel du plan
    (coordonnées plan, polygone).
    """

    # ======================================================
    def get_active_zones_by_plan(self, plan_id: int) -> List[Dict]:
        """
        Récupère toutes les zones actives associées à un plan.

        Paramètres :
        plan_id : identifiant du plan

        Retour :
        Liste de dictionnaires contenant :
        - id
        - plan_id
        - name
        - type
        - polygon (liste de points [[x,y], ...])
        - risk_level
        """
        query = """
        SELECT
            id,
            plan_id,
            name,
            type,
            polygon,
            risk_level
        FROM zones
        WHERE plan_id = %s
          AND is_active = 1
        """

        conn = db_read()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (plan_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        zones = []

        for row in rows:
            try:
                polygon = json.loads(row["polygon"])
            except Exception:
                polygon = []

            zones.append({
                "id": row["id"],
                "plan_id": row["plan_id"],
                "name": row["name"],
                "type": row["type"],
                "polygon": polygon,
                "risk_level": row["risk_level"],
            })

        return zones

    # ======================================================
    def get_active_zones_by_site(self, site_id: int) -> List[Dict]:
        """
        Récupère toutes les zones actives d’un site
        (tous plans confondus).

        Utile pour analyses globales ou dashboards.
        """
        query = """
        SELECT
            z.id,
            z.plan_id,
            z.name,
            z.type,
            z.polygon,
            z.risk_level
        FROM zones z
        JOIN plans p ON p.id = z.plan_id
        WHERE p.site_id = %s
          AND z.is_active = 1
        """

        conn = db_read()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (site_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        zones = []

        for row in rows:
            try:
                polygon = json.loads(row["polygon"])
            except Exception:
                polygon = []

            zones.append({
                "id": row["id"],
                "plan_id": row["plan_id"],
                "name": row["name"],
                "type": row["type"],
                "polygon": polygon,
                "risk_level": row["risk_level"],
            })

        return zones
