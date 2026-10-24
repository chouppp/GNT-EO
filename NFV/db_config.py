"""Shared MySQL settings; importing this module never connects to a database."""
import configparser
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "database.ini"


def load_database_config(config_path=None):
    """Read on each call; resolve the default independently of the working directory."""
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    parser = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8-sig") as stream:
        parser.read_file(stream)
    if "mysql" not in parser:
        raise ValueError(f"Missing [mysql] section in {path}")
    section = parser["mysql"]
    for key in ("host", "user", "database", "port", "charset"):
        if not section.get(key, "").strip():
            raise ValueError(f"Missing mysql.{key} in {path}")
    try:
        port = section.getint("port")
    except ValueError as exc:
        raise ValueError(f"mysql.port must be an integer in {path}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"mysql.port must be between 1 and 65535 in {path}")
    password_env = section.get("password_env", "").strip()
    if password_env:
        if password_env not in os.environ:
            raise ValueError(f"Set the environment variable {password_env} configured in {path}")
        password = os.environ[password_env]
    elif "password" in section:
        password = section["password"]
    else:
        raise ValueError(f"Missing mysql.password in {path}")
    return dict(host=section["host"].strip(), port=port,
                user=section["user"].strip(), password=password,
                database=section["database"].strip(), charset=section["charset"].strip())


def connect_database(*, autocommit=False):
    """Keep transaction behavior explicit; connection identity comes only from INI."""
    import pymysql
    return pymysql.connect(**load_database_config(), autocommit=autocommit)
