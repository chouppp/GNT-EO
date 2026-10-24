"""Resolve configured relative paths from the project root, not the process cwd."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def resolve_config_paths(config):
    result = dict(config)
    for key in ("checkpoint", "event_file", "vnet_dir", "output_dir"):
        if result.get(key) is not None:
            result[key] = project_path(result[key])
    for key in ("event_files", "validation_event_files"):
        if result.get(key) is not None:
            result[key] = [project_path(value) for value in result[key]]
    return result
