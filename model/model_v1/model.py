from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
PLAYER_CSV = DATA_DIR / "player_data.csv"
STINT_CSV = DATA_DIR / "stint_data.csv"


@dataclass(frozen=True)
class LineupStat:
    team: str
    lineup: tuple[str, str, str, str]
    rating_total: float
    minutes: float
    goals_for: float
    goals_against: float

    @property
    def net_goals(self) -> float:
        return self.goals_for - self.goals_against

    @property
    def net_per_60(self) -> float:
        if self.minutes <= 0:
            return 0.0
        return self.net_goals * 60.0 / self.minutes


def _load_players(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["player"], df["rating"]))


def _lineup_rating(players: tuple[str, str, str, str], ratings: dict[str, float]) -> float:
    return sum(ratings.get(p, 0.0) for p in players)


def _canonical_lineup(players: list[str]) -> tuple[str, str, str, str]:
    return tuple(sorted(players))


def build_lineup_stats(
    stint_df: pd.DataFrame,
    ratings: dict[str, float],
    max_rating: float = 8.0,
) -> list[LineupStat]:
    stats = {}

    for row in stint_df.itertuples(index=False):
        minutes = float(row.minutes)

        home_players = [row.home1, row.home2, row.home3, row.home4]
        away_players = [row.away1, row.away2, row.away3, row.away4]

        for team, players, gf, ga in [
            (row.h_team, home_players, row.h_goals, row.a_goals),
            (row.a_team, away_players, row.a_goals, row.h_goals),
        ]:
            lineup = _canonical_lineup(players)
            rating_total = _lineup_rating(lineup, ratings)
            if rating_total > max_rating:
                continue

            key = (team, lineup)
            if key not in stats:
                stats[key] = {
                    "minutes": 0.0,
                    "goals_for": 0.0,
                    "goals_against": 0.0,
                    "rating_total": rating_total,
                }
            stats[key]["minutes"] += minutes
            stats[key]["goals_for"] += float(gf)
            stats[key]["goals_against"] += float(ga)

    lineup_stats = []
    for (team, lineup), agg in stats.items():
        lineup_stats.append(
            LineupStat(
                team=team,
                lineup=lineup,
                rating_total=agg["rating_total"],
                minutes=agg["minutes"],
                goals_for=agg["goals_for"],
                goals_against=agg["goals_against"],
            )
        )
    return lineup_stats


def top_lineups(
    lineup_stats: list[LineupStat],
    min_minutes: float = 0.0,
    top_n: int = 10,
) -> dict[str, list[LineupStat]]:
    by_team: dict[str, list[LineupStat]] = {}
    for stat in lineup_stats:
        if stat.minutes < min_minutes:
            continue
        by_team.setdefault(stat.team, []).append(stat)

    for team, stats in by_team.items():
        stats.sort(key=lambda s: (s.net_per_60, s.net_goals, s.minutes), reverse=True)
        by_team[team] = stats[:top_n]
    return by_team


def main() -> None:
    ratings = _load_players(PLAYER_CSV)
    stints = pd.read_csv(STINT_CSV)
    lineup_stats = build_lineup_stats(stints, ratings, max_rating=8.0)

    leaders = top_lineups(lineup_stats, min_minutes=1.0, top_n=5)
    for team, stats in leaders.items():
        print(f"Team: {team}")
        for s in stats:
            players = ", ".join(s.lineup)
            print(
                f"  lineup=[{players}] rating={s.rating_total:.1f} "
                f"mins={s.minutes:.2f} net={s.net_goals:.1f} net/60={s.net_per_60:.2f}"
            )


if __name__ == "__main__":
    main()
