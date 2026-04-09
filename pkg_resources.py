from pathlib import Path
import importlib


def resource_filename(package_name, resource_name):
    pkg = importlib.import_module(package_name)
    pkg_dir = Path(pkg.__file__).resolve().parent
    return str(pkg_dir / resource_name)