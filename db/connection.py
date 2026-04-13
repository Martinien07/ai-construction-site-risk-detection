from db.pool import get_read_connection, get_write_connection, init_pools


def init_db():
    """
    Initialise l'accès à la base de données.
    À appeler au démarrage de l'application ou du notebook.
    """
    init_pools()


def db_read():
    return get_read_connection()


def db_write():
    return get_write_connection()
