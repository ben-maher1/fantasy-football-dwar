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
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class RosterSettings(_Frozen):
    """Active roster spots (starters + bench) and how a typical team splits
    them by position. Taxi and IR spots hold rookies and injured players and
    are left out. With these set, value is measured against the waiver wire
    and counts only the weeks a player would be in a lineup."""

    bench: int = Field(ge=0)
    by_position: dict[str, int]


class LeagueFormat(_Frozen):
    name: str
    scoring: str
    teams: int = Field(gt=1)
    lineup: dict[str, int]
    discount: float = Field(default=0.85, gt=0, le=1)
    roster: RosterSettings | None = None

    @model_validator(mode="after")
    def _roster_fits(self) -> LeagueFormat:
        if self.roster is not None:
            active = sum(self.lineup.values()) + self.roster.bench
            if sum(self.roster.by_position.values()) != active:
                raise ValueError(f"roster.by_position sums to {sum(self.roster.by_position.values())}, "
                                 f"but starters + bench = {active}")
        return self


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
    # Situation factors per position (features/situation.py): level factors
    # shift a season's scoring, range factors widen its yearly step. Empty
    # means none (the step 5 model).
    situation_level: dict[str, tuple[str, ...]] = {}
    situation_range: dict[str, tuple[str, ...]] = {}
    # Per-position age curve (model/talent.py): quadratic (default) or hinge.
    age_curve: dict[str, Literal["quadratic", "hinge", "cubic", "plateau", "talent"]] = {}
    # Usage and early-career terms (model/talent.py), each a list of positions:
    # weekly opportunity points + per-player efficiency, a first-round step in
    # the rookie prior, and career-year growth scaled by draft capital.
    opportunity: tuple[str, ...] = ()
    opportunity_years: int = 0  # >0: opportunity points only in career years 1..n
    rookie_round1: tuple[str, ...] = ()
    growth: tuple[str, ...] = ()
    game_steps: tuple[str, ...] = ()  # talent also steps after every game (model/talent.py)
    game_mode: str = "walk"           # "walk": lasting per-game steps; "form": decaying in-season form
    qb_drift: tuple[str, ...] = ()    # offseason talent drift with the QB rating (model/talent.py)
    form_rho: float | None = None      # hold the form decay rate fixed ("form" mode); None = estimate it
    season_rho: tuple[str, ...] = ()  # talent pulled toward average between seasons (rho_season; random walk otherwise)
    season_rho_mean: Literal["free", "vet"] = "free"  # what season_rho pulls toward
    tau_y3: tuple[str, ...] = ()      # first-round picks' larger offseason step into career year 3 (lam_y3)
    tau_early: tuple[str, ...] = ()   # offseason talent step larger early in a career (model/talent.py lam_tau)
    injury_exits: tuple[str, ...] = ()  # leave injury-exit games out of the fit (features/exits.py)
    young_steps: tuple[str, ...] = ()  # lasting per-game talent steps for young players ("form" mode)
    first_season_by_position: dict[str, int] = {}  # a shorter history window where a position uses one
    draws_by_position: dict[str, int] = {}  # sampler draws where a position needs more than `sampler.draws`

    def likelihood_for(self, position: str) -> str:
        return self.likelihood_by_position.get(position, self.likelihood)

    def likelihoods(self) -> dict[str, str]:
        return {p: self.likelihood_for(p) for p in self.positions}
    sampler: SamplerSettings
    diagnostics: DiagnosticThresholds


def uses_situation(cfg: ModelConfig) -> bool:
    return any(cfg.situation_level.values()) or any(cfg.situation_range.values())


class ValueConfig(_Frozen):
    draws: int = Field(gt=0)                 # posterior draws carried into the value layer
    horizon_seasons: int = Field(ge=0)       # full seasons valued after the current one
    qb_fade: tuple[str, ...] = ()            # positions whose QB-drift rating fades toward average in later seasons
    replacement_seasons: int = Field(gt=0)   # recent complete seasons that set replacement level
    pool_min_games: int = Field(ge=1)        # games for a player-season to count in that pool
    calibrate_spread: tuple[str, ...] = ()   # positions whose backtest learns a spread factor
    spread_factor: dict[str, float] = {}     # weekly runs: factor per position (value/spread.py)
    short_by_role: tuple[str, ...] = ()      # positions whose short-game chance depends on preseason role
    injury_history: bool = False             # injury history in the games model + current-status cut
    games_hinge_quantile: float | None = None  # games model: talent stops adding games above this quantile
    games_lagged_talent: bool = False        # games model trained on last season's talent


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def load_scoring(name: str) -> ScoringConfig:
    return ScoringConfig(**_read_yaml(config_dir() / "scoring" / f"{name}.yaml"))


def load_format(name: str) -> LeagueFormat:
    return LeagueFormat(**_read_yaml(config_dir() / "formats" / f"{name}.yaml"))


def load_out_for_season(season: int, news_as_of: date) -> frozenset[str]:
    """Player ids known (by `news_as_of`) to miss the rest of `season`
    (config/out_for_season.yaml)."""
    path = REPO_ROOT / "config" / "out_for_season.yaml"
    if not path.exists():
        return frozenset()
    rows = (yaml.safe_load(path.read_text()) or {}).get("players") or []
    return frozenset(
        str(r["player_id"]) for r in rows
        if int(r["season"]) == season and date.fromisoformat(str(r["known_on"])) <= news_as_of
    )


def load_draft_dates() -> DraftDates:
    return DraftDates(**_read_yaml(config_dir() / "draft_dates.yaml"))


def load_model_config() -> ModelConfig:
    return ModelConfig(**_read_yaml(config_dir() / "model.yaml"))


def load_value_config() -> ValueConfig:
    return ValueConfig(**_read_yaml(config_dir() / "value.yaml"))
