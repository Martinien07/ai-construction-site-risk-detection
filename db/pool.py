from mysql.connector import pooling
from db.env import load_db_env

_POOL_READ = None
_POOL_WRITE = None


def init_pools(read_size=5, write_size=5):
    """
    Initialise les pools de connexions MySQL (lecture / écriture).
    Cette fonction doit être appelée UNE SEULE FOIS au démarrage.
    """
    global _POOL_READ, _POOL_WRITE

    conf = load_db_env()

    _POOL_READ = pooling.MySQLConnectionPool(
        pool_name="read_pool",
        pool_size=read_size,
        **conf
    )

    _POOL_WRITE = pooling.MySQLConnectionPool(
        pool_name="write_pool",
        pool_size=write_size,
        **conf
    )


def get_read_connection():
    """
    Retourne une connexion du pool de lecture.
    """
    if _POOL_READ is None:
        raise RuntimeError(
            "Pool READ non initialisé. Appelez init_pools() avant toute requête."
        )
    return _POOL_READ.get_connection()


def get_write_connection():
    """
    Retourne une connexion du pool d'écriture.
    """
    if _POOL_WRITE is None:
        raise RuntimeError(
            "Pool WRITE non initialisé. Appelez init_pools() avant toute requête."
        )
    return _POOL_WRITE.get_connection()
