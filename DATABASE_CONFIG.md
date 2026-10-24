# Shared Database Configuration

All current algorithm and data-import entrypoints use `NFV/db_config.py` to read `database.ini` from the project root.

## Create the local configuration

If `database.ini` already exists, edit it without overwriting your local settings. Otherwise, copy `database.example.ini` when available, or create a file using the template in `README.md`.

## Switch databases

1. Edit the `[mysql]` section of `database.ini`.
2. Change `database` to the desired database name. To connect to another server, also update `host`, `port`, `user`, and the password settings.
3. Save the file and restart the relevant scripts. Existing connections do not switch automatically.

The configuration location is resolved from the code location, independently of the working directory. The target database must already contain the required schema and experiment data. Switching configuration does not create or copy databases.

## Password settings

The example configuration uses `password_env = EO_DB_PASSWORD`. Set this variable in the same PowerShell window before launching an experiment:

```powershell
$env:EO_DB_PASSWORD = "YOUR_MYSQL_PASSWORD"
```

Replace the placeholder with the actual server password. A new terminal requires the variable to be set again. When `password_env` is populated, the named variable is required and takes precedence over `password`.

Alternatively, leave `password_env` empty and set `password` in your local configuration. Percent signs in passwords are read literally. Do not publish a configuration containing a real password. Changing this file does not change the MySQL account's server-side password.

## Scope

A3C-GCN training and mapping use the same shared file. Legacy YAML fields `host`, `port`, `user`, `name`, `password`, and `password_env` no longer control the connection. The training YAML field `database.max_paths_per_pair` remains an algorithm parameter. Mapping output records the effective connection settings without the password.

The shared configuration covers EO, PSO, GA, NORD, D-ViNE, CEVNE, A3C-GCN, network data generation/import, and traffic-test entrypoints. Historical backup scripts are excluded from the maintained runtime entrypoints and were removed from the project during anonymization.

## Use in a new script

Run from the project root, or make the project root available on `PYTHONPATH`:

```python
from NFV.db_config import connect_database

conn = connect_database()
try:
    # Perform database operations using conn.
    pass
finally:
    conn.close()
```

The default is `autocommit=False`. A3C-GCN real-time mapping explicitly uses `autocommit=True`.
