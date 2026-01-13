# Lineup Model Overview

This module builds and ranks four-player lineups for wheelchair rugby stints.
It uses the player ratings and stint results to find the most effective lineups
while enforcing the roster rating cap of 8.

## Inputs

- `data/player_data.csv`: columns `player`, `rating`
- `data/stint_data.csv`: columns `game_id`, `h_team`, `a_team`, `minutes`,
  `h_goals`, `a_goals`, `home1..home4`, `away1..away4`

## How It Works

1. Load player ratings into a lookup map.
2. Iterate over each stint and build two lineups:
   - Home lineup from `home1..home4`
   - Away lineup from `away1..away4`
3. Canonicalize each lineup by sorting the player names so the same four
   players are treated as a single unit regardless of order.
4. Compute the lineup rating total and drop any lineup with rating > 8.
5. Aggregate lineup totals by team and lineup:
   - minutes played
   - goals for
   - goals against
6. Compute net goals and net goals per 60 minutes to compare effectiveness.

## Ranking

Lineups are ranked within each team by:

1. Net goals per 60 minutes (primary)
2. Net goals (secondary)
3. Minutes played (tie-breaker)

The `top_lineups` function lets you set:

- `min_minutes`: filter out small samples
- `top_n`: number of lineups returned per team

## Running

```bash
python3 SportsModule/model.py
```

The script prints the top 5 lineups per team with rating, minutes, net goals,
and net goals per 60.
