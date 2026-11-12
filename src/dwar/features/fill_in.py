"""Fill-in games: an RB's games in which his team's lead back did not play
(ben, 2026-11-11).

A backup who gets the lead back's work for a week scores far more than
usual (2002-25: 7.4 vs 4.5 half-PPR PPG, 8.7 vs 4.7 carries), but those
points say about half as much about his next season as his other games do
(0.42 vs 0.83 next-season PPG per point). The RB model gets a term for these
games (`fill_in` in config/model.yaml), so the bump is not read as talent.

The lead back for a team game is the RB with the most carries in the team's
earlier games that season, if he had at least LEAD_SHARE of its RB carries
over at least MIN_TEAM_GAMES games. Only earlier games are used, so the flag
is known before kickoff and needs no lookahead. A lead back who has left the
team (traded or released) keeps flagging his old team's games until another
back passes his carries; in-season departures of lead backs are rare.
"""

from __future__ import annotations

import polars as pl

LEAD_SHARE = 0.40
MIN_TEAM_GAMES = 2


def lead_backs(weeks: pl.DataFrame) -> pl.DataFrame:
    """Per team game (team, season, week): the lead back as of kickoff
    (`lead`) and whether he played it (`lead_played`); only games that have
    a lead back."""
    tw = (
        weeks.select("team", "season", "week").unique().sort("team", "season", "week")
        .with_columns(k=pl.int_range(pl.len()).over("team", "season"))
    )
    rb = (
        weeks.filter(pl.col("position") == "RB")
        .group_by("player_id", "team", "season", "week").agg(c=pl.col("carries").fill_null(0).sum())
    )
    backs = rb.select("player_id", "team", "season").unique()
    grid = (
        tw.join(backs, on=["team", "season"])
        .join(rb, on=["player_id", "team", "season", "week"], how="left")
        .with_columns(pl.col("c").fill_null(0))
        .sort("team", "season", "player_id", "k")
        .with_columns(before=pl.col("c").cum_sum().shift(1, fill_value=0).over("team", "season", "player_id"))
    )
    top = (
        grid.sort("before", "player_id", descending=[True, False])
        .group_by("team", "season", "week", maintain_order=True)
        .agg(lead=pl.col("player_id").first(), lead_c=pl.col("before").first(),
             total=pl.col("before").sum(), k=pl.col("k").first())
        .filter((pl.col("k") >= MIN_TEAM_GAMES) & (pl.col("total") > 0)
                & (pl.col("lead_c") >= LEAD_SHARE * pl.col("total")))
    )
    played = weeks.select("team", "season", "week", lead="player_id").unique().with_columns(lead_played=pl.lit(True))
    return (
        top.join(played, on=["team", "season", "week", "lead"], how="left")
        .with_columns(pl.col("lead_played").fill_null(False))
        .select("team", "season", "week", "lead", "lead_played")
    )


def flag_fill_ins(weeks: pl.DataFrame) -> pl.DataFrame:
    """`weeks` with a boolean `fill_in`: an RB's game whose team's lead back
    (`lead_backs`) did not play it. Needs player_id, team, season, week,
    position and carries; no flags without carries."""
    if "carries" not in weeks.columns or "position" not in weeks.columns:
        return weeks.with_columns(fill_in=pl.lit(False))
    lead = lead_backs(weeks).filter(~pl.col("lead_played")).select("team", "season", "week", "lead")
    return (
        weeks.with_row_index("_f")
        .join(lead, on=["team", "season", "week"], how="left")
        .with_columns(fill_in=(pl.col("position") == "RB") & pl.col("lead").is_not_null()
                      & (pl.col("lead") != pl.col("player_id")))
        .sort("_f").drop("_f", "lead")
    )
