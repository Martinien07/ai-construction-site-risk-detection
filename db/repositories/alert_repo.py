# db/repositories/alert_repo.py

import pandas as pd
from db.connection import db_read, db_write


class AlertRepository:
    """
    Repository responsable de l'accès aux alertes HSE en base de données.
    """

    # ======================================================
    # INSERTION
    # ======================================================
    @staticmethod
    def create_alert(
        risk_event_id: int,
        camera_id: int,
        plan_id: int,
        level: str,
        message: str,
        status: str = "OPEN"
    ) -> int:
        """
        Crée une nouvelle alerte HSE en base de données.

        Retourne :
        - id de l'alerte créée
        """

        query = """
        INSERT INTO alerts (
            risk_event_id,
            camera_id,
            plan_id,
            level,
            message,
            status,
            created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """

        conn = None
        cursor = None

        try:
            conn = db_write()
            cursor = conn.cursor()

            cursor.execute(
                query,
                (
                    risk_event_id,
                    camera_id,
                    plan_id,
                    level,
                    message,
                    status
                )
            )

            conn.commit()
            return cursor.lastrowid

        finally:
            # Fermeture propre des ressources
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # ======================================================
    # UPDATE
    # ======================================================
    @staticmethod
    def update_status(alert_id: int, new_status: str) -> None:
        """
        Met à jour le statut d'une alerte existante.
        """

        query = """
        UPDATE alerts
        SET status = %s
        WHERE id = %s
        """

        conn = None
        cursor = None

        try:
            conn = db_write()
            cursor = conn.cursor()
            cursor.execute(query, (new_status, alert_id))
            conn.commit()

        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    # ======================================================
    # LECTURE
    # ======================================================
    @staticmethod
    def get_alerts_by_site(
        site_id: int,
        start_time=None,
        end_time=None
    ) -> pd.DataFrame:
        """
        Récupère les alertes d’un site, optionnellement filtrées par période.
        """

        query = """
        SELECT
            a.id,
            a.level,
            a.status,
            a.message,
            a.created_at,
            c.name AS camera_name,
            p.level AS plan_level
        FROM alerts a
        JOIN cameras c ON c.id = a.camera_id
        JOIN plans p ON p.id = a.plan_id
        WHERE p.site_id = %s
        """

        params = [site_id]

        if start_time is not None and end_time is not None:
            query += " AND a.created_at BETWEEN %s AND %s"
            params.extend([start_time, end_time])

        query += " ORDER BY a.created_at DESC"

        conn = None

        try:
            conn = db_read()
            return pd.read_sql(query, conn, params=params)

        finally:
            if conn:
                conn.close()
    
