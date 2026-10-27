# EO / NFV Experiments

This project implements EO, A3C-GCN, CEVNE, D-ViNE, GA, and NORD for virtual network embedding.
All commands below use **Windows PowerShell**. The project root is the directory containing this README, `requirements.txt`, and `NFV/`.

## 1. Install dependencies

Open PowerShell in the project root and create an isolated Python environment:

```powershell
python -m venv .venv-eo
.\.venv-eo\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Alternatively, activate an existing Python or Conda environment with the required dependencies. The requirements include numerical computing, plotting, NetworkX, the MySQL client, Pyomo, TensorFlow, PyTorch, PyTorch Geometric, and the test runner. Versions are not pinned, and compatibility across all version combinations has not been verified.

If activation scripts are disabled, use the environment's interpreter directly:

```powershell
.\.venv-eo\Scripts\python.exe -m pip install -r requirements.txt
```

### External GLPK installation

GLPK is installed separately from the Python requirements:

- [GNU GLPK project](https://www.gnu.org/software/glpk/)
- [WinGLPK binaries for Windows](https://sourceforge.net/projects/winglpk/files/winglpk/)
- [GNU GLPK source downloads](https://ftp.gnu.org/gnu/glpk/)

WinGLPK is a separate Windows distribution with older releases; its version is not necessarily the latest GNU source release. Extract the archive and use `w64` for the supplied 64-bit binaries. Keep the accompanying DLLs and add the directory containing `glpsol.exe` to PATH. Reopen the terminal, reactivate your Python environment, and check:

```powershell
glpsol --version
```

CEVNE and D-ViNE invoke GLPK through Pyomo. Check solver availability in the Python environment:

```powershell
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('glpk').available())"
```

The result should be `True`. Install and start the MySQL server separately.

## 2. Configure the database

All current algorithm entrypoints read the existing `database.ini` in the project root. Edit this file directly to configure your MySQL connection; no example-file copy or password environment variable is needed. The filename used by the code is `database.ini`, not `data.ini`.

Update the existing `[mysql]` section with your own connection settings:

```ini
[mysql]
host = localhost
port = 3306
user = root
password = YOUR_MYSQL_PASSWORD
database = YOUR_DATABASE_NAME
charset = utf8mb4
password_env =
```

**You must update both `password` and `database` in the existing `database.ini`:**

- `password`: replace `YOUR_MYSQL_PASSWORD` with your actual MySQL account password.
- `database`: replace `YOUR_DATABASE_NAME` with the actual name of the MySQL database that contains your experiment tables (for Option B, the database into which you import the supplied SQL files).
- `password_env`: leave it blank (`password_env =`) so the application reads `password` directly from this file.

Changing only the password is not sufficient: the database name must also match your local setup.

- Change `host`, `port`, and `user` as needed for your MySQL server.
- Restart scripts after changing the configuration; existing connections do not switch automatically.
- Editing this file does not change the server-side MySQL password.
- Keep any configuration containing a real password private.

Prepare the database schema and experiment data before running. The main tables are `node_single_table`, `link_single_table`, and `link_table`. Creating an empty database or changing its name in the configuration does not create tables or copy data.

From the project root, verify the connection and selected database without modifying table data:

```powershell
python -c "from NFV.db_config import connect_database; c=connect_database(); q=c.cursor(); q.execute('SELECT DATABASE(), CURRENT_USER()'); print(q.fetchone()); q.close(); c.close()"
```

GNT-EO, CEVNE, D-ViNE, GA, and NORD update database resources and traffic during real-time experiments. Use matching initial database states for algorithm comparisons and avoid concurrent experiments against the same resource tables.

## 3. Before launching

### 3.1 Generate a topology or use the supplied SQL data

**Recommended: Option B, using the supplied pre-generated database.** Use Option A only when you need a newly generated topology. Configure the existing `database.ini` as described in Section 2 before either option.

**Option A: Generate a new topology with Waxman.** `NFV/waxman.py` generates a connected directed Waxman topology. First create the target MySQL database and the required empty tables (`node_single_table`, `link_single_table`, and `link_table`); the import scripts do not create the schema. You can use the table definitions from the SQL dumps in Option B. In `NFV/write_to_link_dataset.py`, uncomment the following two lines in the main block so that nodes are imported along with single-hop links:

```python
nodes = load_nodes_from_file("p_net.yaml")
insert_nodes_into_db(nodes)
```

Then run the scripts in this order from the `NFV` directory: **`waxman.py` -> `write_to_link_dataset.py` -> `linkdata_generator.py`**. Start the following block in the project root:

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\')
try {
    & $Python .\waxman.py
    if ($LASTEXITCODE -ne 0) { throw "Topology generation failed." }
    & $Python .\write_to_link_dataset.py
    if ($LASTEXITCODE -ne 0) { throw "Node/link import failed." }
    & $Python .\linkdata_generator.py
    if ($LASTEXITCODE -ne 0) { throw "Candidate-path import failed." }
} finally {
    Pop-Location
}
```

1. `waxman.py` writes `NFV/p_net_directed.pkl` and `NFV/p_net.yaml`. Its main block controls `num_nodes`, `wm_alpha`, and `wm_beta`.
2. `write_to_link_dataset.py` reads `p_net.yaml` and inserts single-hop links into `link_single_table`; with the two lines above enabled, it also inserts nodes into `node_single_table`.
3. `linkdata_generator.py` reads `p_net_directed.pkl` and the single-hop bandwidth data, generates candidate paths with a six-hop cutoff, and inserts them into `link_table`.

The import scripts insert records without clearing existing data. Use empty tables for a new topology to avoid duplicate or mixed topology data. Generating topology files alone does not populate the database.

**Option B (recommended): Use the pre-generated database files.** From the project root, extract the supplied archive:

```powershell
Expand-Archive -LiteralPath .\sql_example.zip -DestinationPath .\sql_example
```

If `sql_example/` is already extracted, skip this step. Create or select your own experiment database in your MySQL client, then import all three files below in the listed order:

| File | Contents |
| --- | --- |
| `sql_example/node_single_table.sql` | Substrate nodes and CPU resources. |
| `sql_example/link_single_table.sql` | Directed single-hop links and their resources. |
| `sql_example/link_table.sql` | Pre-generated candidate paths and path resources. |

Select your target database in your MySQL client before importing these files. The dumps contain table-recreation statements, so import into the intended experiment database. Set `database` in `database.ini` to that database's name and configure your own password as described in Section 2. For Option B, skip all three generation/import scripts in Option A and keep the matching supplied topology file; do not combine the pre-generated SQL data with an unrelated, newly randomized Waxman topology.

### 3.2 Required inputs and launch environment

Required inputs include:

- `NFV/p_net_directed.pkl`: substrate topology.
- `NFV/vnet_dataset/`: virtual networks referenced by the events.
- Each algorithm's local `vnr_events.yaml`: used separately by EO, CEVNE, D-ViNE, GA, and NORD.
- A3C-GCN's configured `event_file`, defaulting to `NFV/Random_event_generator/vnr_events.yaml`.
- A3C-GCN's configured `checkpoint`, defaulting to `NFV/a3c_gcn/checkpoints/a3c_gcn_actor_best.pt`.

Event files in different directories may differ. Align event sequences, virtual-network inputs, and initial database states for comparisons.

**Activate an environment with the dependencies and start in the project root.** Each launch block below includes these initialization commands, so copy the entire block:

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
```

`$Python` selects the interpreter from the current environment. Without activation, replace its assignment with `$Python = Join-Path $ProjectRoot '.venv-eo\Scripts\python.exe'` to use the project environment directly.

`PYTHONPATH` makes the `NFV` package available to scripts launched from subdirectories. `Push-Location` selects the correct directory for relative event and output paths; `finally` restores the previous directory. These commands do not set the database password: configure it as described above.

## 4. EO

Full NFV event experiment: `NFV/test/nfv_realtime.py`. The root-level `EO.py` is a separate optimization example.

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\test')
try {
    & $Python .\nfv_realtime.py
} finally {
    Pop-Location
}
```

Events: `NFV/test/vnr_events.yaml`.
Outputs: a new numbered directory under `NFV/result_storage/EO/`.

## 5. A3C-GCN

### 5.1 Train a model

Entrypoint: `NFV/a3c_gcn/train.py`. Edit `train_config.example.yaml` to select training and validation events, episode counts, and the output directory. Database connections still use the root-level `database.ini`.

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location $ProjectRoot
try {
    & $Python -m NFV.a3c_gcn.train --config NFV/a3c_gcn/train_config.example.yaml
} finally {
    Pop-Location
}
```

Training reads a database snapshot and allocates/releases resources in memory without updating resource tables. The default model directory is `NFV/a3c_gcn/checkpoints/`. Skip training when suitable Actor weights are already available.

### 5.2 Run mapping experiments

Entrypoint: `NFV/a3c_gcn/a3c_gcn_realtime.py`. Edit `checkpoint`, `event_file`, `vnet_dir`, and `output_dir` in `mapping_config.example.yaml`. Relative configuration paths are resolved from the project root.

The default mode replays events against an in-memory snapshot without database writes:

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location $ProjectRoot
try {
    & $Python -m NFV.a3c_gcn.a3c_gcn_realtime --config NFV/a3c_gcn/mapping_config.example.yaml
} finally {
    Pop-Location
}
```

To allocate and release resources in the database:

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location $ProjectRoot
try {
    & $Python -m NFV.a3c_gcn.a3c_gcn_realtime --config NFV/a3c_gcn/mapping_config.example.yaml --write-db
} finally {
    Pop-Location
}
```

Write mode requires InnoDB resource tables. The program checks for full initial capacity and does not reset the database automatically.

This entrypoint does not simulate traffic or network temperature; its temperature output is not directly comparable with the other real-time experiments. Default results go into a new numbered directory under `NFV/result_storage/A3C_GCN/`.

## 6. CEVNE

Entrypoint: `NFV/CEVNE/CEVNE_realtime.py`. Requires external GLPK.

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\CEVNE')
try {
    & $Python .\CEVNE_realtime.py
} finally {
    Pop-Location
}
```

Events: `NFV/CEVNE/vnr_events.yaml`. Relative output files are written in the algorithm directory.

## 7. D-ViNE

Entrypoint: `NFV/Dvine/D-vine_realtime.py`. Requires external GLPK. Preserve the hyphen in the script filename.

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\Dvine')
try {
    & $Python .\D-vine_realtime.py
} finally {
    Pop-Location
}
```

Events: `NFV/Dvine/vnr_events.yaml`. The script's `DVINE_RESULT_ROOT` selects the result root; each run creates a new numbered subdirectory.

## 8. GA

Entrypoint: `NFV/GA_base/GA.py`.

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\GA_base')
try {
    & $Python .\GA.py
} finally {
    Pop-Location
}
```

Events: `NFV/GA_base/vnr_events.yaml`. Current outputs go to `NFV/GA_base/GA/`; preserve any previous results you need before repeating an experiment.

## 9. NORD

Full event-mapping entrypoint: `NFV/NORD/NORD_copy.py`. `NORD_realtime.py` currently reads networks and computes node attributes rather than running the complete event deployment experiment.

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\NORD')
try {
    & $Python .\NORD_copy.py
} finally {
    Pop-Location
}
```

Events: `NFV/NORD/vnr_events.yaml`. Outputs in that directory include `nord_vne_mapping.yaml`, `NORD_flow_history.yaml`, and `NORD_resource_usage.yaml`.

## 10. Troubleshooting

| Error | What to check |
| --- | --- |
| Invalid expression after `& $Python` | Copy the entire launch block, including the `$Python` assignment. |
| Cannot open `nfv_realtime.py` | Start in the project root and include the `Push-Location` command. |
| `No module named NFV` | Confirm that the project root is included in `PYTHONPATH`. |
| `No module named EO2_test` | Use the documented EO launch block to run the script in `NFV/test`. |
| Missing `vnr_events.yaml` | Use the correct algorithm working directory; for A3C-GCN, check the YAML configuration. |
| Missing `glpsol` | Install GLPK externally and add its executable directory to PATH. |
| Missing `EO_DB_PASSWORD` | In the existing `database.ini`, leave `password_env =` blank and set `password` to your actual MySQL password. |
| MySQL `Access denied` | Check the configured account, password source, and server-side credentials. |
| Missing database or tables | Prepare the required schema and experiment data before selecting the database. |
| Missing A3C-GCN weights | Check `checkpoint` or train an Actor model first. |

Commands reflect the inspected entrypoints and path handling. Full experiments have not been executed as part of documentation validation.


## 11. Initialization experiments: GA, PSO, and EO2

`NFV/test/pso_realtime.py` is the launcher for initialization experiments. Despite its filename, it supports all three models: **GA**, **PSO**, and **EO2**.

Open `NFV/test/pso_realtime.py` and change `OPTIMIZER_NAME` near the top of the file to one of the following exact values:

```python
OPTIMIZER_NAME = "GA"   # Select GA.
# OPTIMIZER_NAME = "PSO"  # Select PSO instead.
# OPTIMIZER_NAME = "EO2"  # Select EO2 instead.
```

Keep exactly one assignment active. Save the file, then run the same launcher directly; no separate launch script is required when switching models:

```powershell
$ProjectRoot = (Get-Location).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

Push-Location (Join-Path $ProjectRoot 'NFV\test')
try {
    & $Python .\pso_realtime.py
} finally {
    Pop-Location
}
```

Configure your database name and password before running. The launcher reads `NFV/test/vnr_events.yaml` and uses the same substrate topology and virtual-network dataset as the EO experiment. Use the same inputs and initial resource state when comparing the three models.
