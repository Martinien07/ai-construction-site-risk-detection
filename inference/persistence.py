from db.connection import db_write,db_read
from db.retry import retry_db
from mysql.connector import Error



@retry_db()
def save_detections_batch(detections):
    db = db_write()
    cursor = db.cursor()

    values = [
        (
            d["camera_id"],
            d["timestamp"],
            d["object_class"],
            d["confidence"],
            d["bbox_x"],
            d["bbox_y"],
            d["bbox_w"],
            d["bbox_h"],
            d["track_id"]
        )
        for d in detections
    ]

    cursor.executemany("""
        INSERT INTO detections (
            camera_id, timestamp, object_class, confidence,
            bbox_x, bbox_y, bbox_w, bbox_h, track_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, values)

    db.commit()
    cursor.close()
    db.close()





def get_detections_for_frame(camera_id, start_ts, end_ts):
    """
    Récupère les détections pour une caméra
    entre deux timestamps (frame time window).
    """

    query = """
        SELECT
            object_class,
            confidence,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            track_id
        FROM detections
        WHERE camera_id = %s
          AND timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp ASC
    """

    conn = None
    cursor = None

    try:
        conn = db_read()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(query, (camera_id, start_ts, end_ts))
        rows = cursor.fetchall()

        return rows

    except Error as e:
        # En production : logger plutôt que print
        print(f"[DB ERROR] get_detections_for_frame: {e}")
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
