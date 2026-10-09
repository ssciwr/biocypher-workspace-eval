"""Load the evaluation configuration and resolve paths."""

import hashlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
# EVAL_CONFIG selects another configuration file (e.g. a separate round).
CONFIG_FILE = Path(os.getenv("EVAL_CONFIG", EVAL_DIR / "config.toml"))
DATA_DIR = EVAL_DIR / "data"
PROMPTS_DIR = EVAL_DIR / "prompts"
DATASET_FILE = "synthetic_protein_interactions.tsv"
LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6")


@dataclass(frozen=True)
class Level:
    name: str
    dataset: str
    seed_scaffold: bool
    max_calls: int
    max_minutes: float
    final_marker: str

    @property
    def prompt_file(self) -> Path:
        return PROMPTS_DIR / f"{self.name}.md"

    @property
    def dataset_file(self) -> Path | None:
        if self.dataset == "none":
            return None
        return DATA_DIR / self.dataset / DATASET_FILE


@dataclass(frozen=True)
class Model:
    name: str
    base_url: str
    api_key_env: str


def load(path: Path = CONFIG_FILE) -> dict:
    raw = path.read_bytes()
    config = tomllib.loads(raw.decode())
    config["sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def levels(config: dict) -> dict[str, Level]:
    return {name: Level(name=name, **spec) for name, spec in config["levels"].items()}


def models(config: dict) -> list[Model]:
    return [Model(**spec) for spec in config["models"]]


def _repo(config: dict, key: str, env: str) -> Path:
    override = os.getenv(env)
    path = Path(override) if override else REPO_ROOT / config["pins"][key]
    return path.resolve()


def reference_repo(config: dict) -> Path:
    return _repo(config, "reference_repo", "EVAL_REFERENCE_REPO")


def registry_repo(config: dict) -> Path:
    return _repo(config, "registry_repo", "EVAL_REGISTRY_REPO")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
