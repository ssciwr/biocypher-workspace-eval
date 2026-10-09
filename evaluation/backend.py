"""Import the agentic-workspace backend from a biocypher-components-registry checkout.

The registry is not an installable package, so the evaluation imports its
workspace modules from a checkout: a clone of [pins] registry_repo at
registry_commit (config.toml), or a local checkout given by EVAL_REGISTRY_REPO.
The checkout's git commit is recorded with every run.
"""

import importlib
import subprocess
import sys
from pathlib import Path

from evaluation import config as cfg


class BackendNotFoundError(ImportError):
    """The registry checkout is missing or has no workspace backend."""


def load(config: dict | None = None):
    """Return (client_loop, service) modules of the registry backend."""
    try:
        repo = cfg.registry_repo(config or cfg.load())
    except subprocess.CalledProcessError as error:  # offline and no clone yet
        raise BackendNotFoundError(f"cannot clone the registry: {error}") from error
    if not (repo / "src" / "core" / "workspace" / "service.py").exists():
        raise BackendNotFoundError(
            f"no biocypher-components-registry checkout at {repo}; "
            "set [pins] registry_repo/registry_commit or EVAL_REGISTRY_REPO"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    client_loop = importlib.import_module("src.core.workspace.client_loop")
    service = importlib.import_module("src.core.workspace.service")
    return client_loop, service


def repo_path(config: dict | None = None) -> Path:
    return cfg.registry_repo(config or cfg.load())
