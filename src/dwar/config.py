"""Typed loaders for scoring and league-format YAML configs.

League settings are parameters, not constants (CLAUDE.md rule 3): everything
format-specific comes from `config/` through these models.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return Path(os.environ.get("DWAR_CONFIG_DIR", REPO_ROOT / "config"))


def data_dir() -> Path:
    return Path(os.environ.get("DWAR_DATA_DIR", REPO_ROOT / "data"))


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoringConfig(_Frozen):
    """Points per unit of each counting stat. Core fields are required so a
    typo in a YAML file fails instead of silently scoring as zero."""

    name: str
    pass_yd: float
    pass_td: float
    pass_int: float
    pass_2pt: float
    rush_yd: float
    rush_td: float
    rush_2pt: float
    rec: float
    rec_yd: float
    rec_td: float
    rec_2pt: float
    fumble_lost: float
    st_td: float
    # Optional extras; zero unless a format opts in.
    pass_fd: float = 0.0
    rush_fd: float = 0.0
    rec_fd: float = 0.0
    # Extra points per reception by position, e.g. TE premium = {"TE": 0.5}.
    rec_bonus_by_position: dict[str, float] = Field(default_factory=dict)


class LeagueFormat(_Frozen):
    name: str
    scoring: str
    teams: int = Field(gt=1)
    lineup: dict[str, int]
    discount: float = Field(default=0.85, gt=0, le=1)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_scoring(name: str) -> ScoringConfig:
    return ScoringConfig(**_read_yaml(config_dir() / "scoring" / f"{name}.yaml"))


def load_format(name: str) -> LeagueFormat:
    return LeagueFormat(**_read_yaml(config_dir() / "formats" / f"{name}.yaml"))
