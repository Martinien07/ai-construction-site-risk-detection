from dotenv import load_dotenv
import os

load_dotenv()  # charge le fichier .env

def load_db_env():
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ.get("DB_PORT", 3306)),
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "database": os.environ["DB_NAME"],
        "auth_plugin": "mysql_native_password"
    }
