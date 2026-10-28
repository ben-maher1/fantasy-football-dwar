"""Typed loaders for scoring and league-format YAML configs.

League settings are parameters, not constants (CLAUDE.md rule 3): everything
format-specific comes from `config/` through these models.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

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


class DraftDates(_Frozen):
    """Final day of each NFL Draft, with a conservative fallback for
    unlisted years (see config/draft_dates.yaml)."""

    default_month_day: str = Field(pattern=r"^\d{2}-\d{2}$")
    end_dates: dict[int, date] = Field(default_factory=dict)

    def end_of(self, year: int) -> date:
        if year in self.end_dates:
            return self.end_dates[year]
        return date.fromisoformat(f"{year}-{self.default_month_day}")


class SamplerSettings(_Frozen):
    chains: int = Field(ge=2)
    tune: int = Field(gt=0)
    draws: int = Field(gt=0)


class DiagnosticThresholds(_Frozen):
    max_divergences: int = Field(ge=0)
    max_rhat: float = Field(gt=1)
    min_ess_bulk: float = Field(gt=0)
    min_ess_tail: float = Field(gt=0)


class ModelConfig(_Frozen):
    seed: int
    first_season: int
    positions: tuple[str, ...]
    likelihood: Literal["sqrt_hurdle", "sqrt_hurdle_log", "studentt"]
    # Per-position overrides of `likelihood`.
    likelihood_by_position: dict[str, Literal["sqrt_hurdle", "sqrt_hurdle_log", "studentt"]] = {}
    # Weeks below this many plays (attempts + sacks + carries) are left out
    # of a position's talent fit; they still count as games played.
    min_plays: dict[str, int] = {}

    def likelihood_for(self, position: str) -> str:
        return self.likelihood_by_position.get(position, self.likelihood)

    def likelihoods(self) -> dict[str, str]:
        return {p: self.likelihood_for(p) for p in self.positions}
    sampler: SamplerSettings
    diagnostics: DiagnosticThresholds


class ValueConfig(_Frozen):
    draws: int = Field(gt=0)                 # posterior draws carried into the value layer
    horizon_seasons: int = Field(ge=0)       # full seasons valued after the current one
    replacement_seasons: int = Field(gt=0)   # recent complete seasons that set replacement level
    pool_min_games: int = Field(ge=1)        # games for a player-season to count in that pool


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_scoring(name: str) -> ScoringConfig:
    return ScoringConfig(**_read_yaml(config_dir() / "scoring" / f"{name}.yaml"))


def load_format(name: str) -> LeagueFormat:
    return LeagueFormat(**_read_yaml(config_dir() / "formats" / f"{name}.yaml"))


def load_draft_dates() -> DraftDates:
    return DraftDates(**_read_yaml(config_dir() / "draft_dates.yaml"))


def load_model_config() -> ModelConfig:
    return ModelConfig(**_read_yaml(config_dir() / "model.yaml"))


def load_value_config() -> ValueConfig:
    return ValueConfig(**_read_yaml(config_dir() / "value.yaml"))
