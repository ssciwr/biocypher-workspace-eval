"""Import the agentic-workspace backend from a biocypher-components-registry checkout.

The registry is not an installable package, so the evaluation imports its
workspace modules from a checkout: [pins] registry_repo in config.toml
(relative to this repository), or EVAL_REGISTRY_REPO. The checkout's git
commit is recorded with every run; check out the backend version you want
to evaluate there.
"""

import importlib
import sys
from pathlib import Path

from evaluation import config as cfg


class BackendNotFoundError(ImportError):
    """The registry checkout is missing or has no workspace backend."""


def load(config: dict | None = None):
    """Return (client_loop, service) modules of the registry backend."""
    repo = cfg.registry_repo(config or cfg.load())
    if not (repo / "src" / "core" / "workspace" / "service.py").exists():
        raise BackendNotFoundError(
            f"no biocypher-components-registry checkout at {repo}; "
            "set [pins] registry_repo or EVAL_REGISTRY_REPO"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    client_loop = importlib.import_module("src.core.workspace.client_loop")
    service = importlib.import_module("src.core.workspace.service")
    return client_loop, service


def repo_path(config: dict | None = None) -> Path:
    return cfg.registry_repo(config or cfg.load())
