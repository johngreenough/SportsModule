#!/usr/bin/env python
# coding: utf-8

# # Interactive Lineup Dashboard
# 
# Streamlit app to tweak lineups and see updated metrics for:
# - Normalized GD per stint (from lineup optimization inputs)
# - GD per minute vs opponent teams (from linear model inputs)
# 
# Run via `streamlit run streamlit_dashboard.ipynb` or export to `.py` with Jupyter's Download as Python.
# 

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from itertools import combinations
from typing import List, Tuple

DATA_DIR = "data/"
PLAYER_CSV = DATA_DIR + "player_data.csv"
STINT_CSV = DATA_DIR + "stint_data.csv"

st.set_page_config(page_title="Lineup Dashboard", layout="wide")


# In[2]:


@st.cache_data(show_spinner=False)
def load_data():
    player_df = pd.read_csv(PLAYER_CSV)
    stint_df = pd.read_csv(STINT_CSV)

    # Add team column to player data if missing
    if "team" not in player_df.columns:
        player_df = player_df.copy()
        player_df["team"] = player_df["player"].apply(lambda p: "_".join(p.split("_")[:-1]))

    # Build stint_gd dict from columns stint{i}_gd
    stint_gd = {}
    max_stint = 0
    for _, row in player_df.iterrows():
        player = row["player"]
        stint_gd[player] = {}
        for i in range(1, 17):
            col = f"stint{i}_gd"
            if col in row and pd.notna(row[col]):
                stint_gd[player][i] = float(row[col])
                max_stint = max(max_stint, i)

    # WOWY
    wowy_dict = dict(zip(player_df["player"], player_df.get("wowy", pd.Series([0]*len(player_df)))))

    return player_df, stint_df, stint_gd, wowy_dict, max_stint


@st.cache_data(show_spinner=False)
def build_player_opponent_stats(stint_df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the linearmodel aggregation to get plus/minus per minute vs each opponent."""
    records = []
    for _, row in stint_df.iterrows():
        game_id = row["game_id"]
        h_team = row["h_team"]
        a_team = row["a_team"]
        minutes = row["minutes"]
        h_goals = row["h_goals"]
        a_goals = row["a_goals"]

        for i in range(1, 5):
            player = row[f"home{i}"]
            records.append({
                "player": player,
                "team": h_team,
                "opponent": a_team,
                "minutes": minutes,
                "goal_margin": h_goals - a_goals,
                "plus_minus_per_min": (h_goals - a_goals) / minutes if minutes > 0 else 0,
            })
        for i in range(1, 5):
            player = row[f"away{i}"]
            records.append({
                "player": player,
                "team": a_team,
                "opponent": h_team,
                "minutes": minutes,
                "goal_margin": a_goals - h_goals,
                "plus_minus_per_min": (a_goals - h_goals) / minutes if minutes > 0 else 0,
            })

    df = pd.DataFrame(records)
    grouped = df.groupby(["player", "team", "opponent"], as_index=False).agg(
        minutes=("minutes", "sum"),
        goal_margin=("goal_margin", "sum"),
        n_games=("goal_margin", "count"),
        plus_minus_per_min=("plus_minus_per_min", "mean"),
    )
    grouped["total_plus_minus_per_min"] = grouped["goal_margin"] / grouped["minutes"].replace(0, np.nan)
    grouped["total_plus_minus_per_min"] = grouped["total_plus_minus_per_min"].fillna(0)
    return grouped


player_df, stint_df, stint_gd, wowy_dict, MAX_STINT = load_data()
player_opponent_stats = build_player_opponent_stats(stint_df)
TEAMS = sorted(player_df["team"].unique())


# In[3]:


def available_players(team: str) -> List[str]:
    return sorted(player_df[player_df["team"] == team]["player"].tolist())


def lineup_score(lineup: List[str], stint_num: int, wowy_factor: float) -> float:
    score = 0.0
    for p in lineup:
        gd = stint_gd.get(p, {}).get(stint_num, 0.0)
        wowy = wowy_dict.get(p, 0.0)
        score += gd + wowy * wowy_factor
    return score


def optimize_lineup(team: str, stint_num: int, rating_limit: float, wowy_factor: float) -> Tuple[List[str], float]:
    team_players = player_df[player_df["team"] == team]
    ratings = dict(zip(team_players["player"], team_players["rating"]))
    candidates = [p for p in team_players["player"] if stint_num in stint_gd.get(p, {})]
    best_lineup, best_score = None, float("-inf")

    for combo in combinations(candidates, 4):
        rating_sum = sum(ratings.get(p, 0) for p in combo)
        if rating_sum <= rating_limit:
            s = lineup_score(combo, stint_num, wowy_factor)
            if s > best_score:
                best_score = s
                best_lineup = list(combo)

    return (best_lineup or []), (best_score if best_lineup else float("nan"))


def gd_per_minute_vs_opponent(lineup: List[str], opponent: str) -> float:
    subset = player_opponent_stats[player_opponent_stats["opponent"] == opponent]
    vals = []
    for p in lineup:
        v = subset[subset["player"] == p]["total_plus_minus_per_min"]
        if not v.empty:
            vals.append(float(v.iloc[0]))
    return float(np.sum(vals)) if vals else float("nan")


# In[4]:


st.title("Lineup Tuning Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    team = st.selectbox("Team", TEAMS, index=TEAMS.index("Canada") if "Canada" in TEAMS else 0)
    stint_num = st.slider("Stint #", 1, MAX_STINT, 1)
with col2:
    opponent = st.selectbox("Opponent", [t for t in TEAMS if t != team], index=0)
    rating_limit = st.slider("Rating cap", 4.0, 12.0, 8.0, 0.5)
with col3:
    wowy_factor = st.slider("WOWY factor", 0.0, 2.0, 1.0, 0.1)

team_players = available_players(team)
default_lineup, default_score = optimize_lineup(team, stint_num, rating_limit, wowy_factor)

# ML-style (linearmodel) recommendation: top 4 by GD/min vs opponent
ml_reco = (
    player_opponent_stats[
        (player_opponent_stats["team"] == team) & (player_opponent_stats["opponent"] == opponent)
    ]
    .sort_values("total_plus_minus_per_min", ascending=False)
    .head(4)
)
ml_lineup = ml_reco["player"].tolist()

if "selected_lineup" not in st.session_state:
    st.session_state["selected_lineup"] = default_lineup if default_lineup else (ml_lineup if ml_lineup else team_players[:4])

st.markdown("### Select lineup")
selected = st.multiselect(
    "Choose 4 players",
    options=team_players,
    default=st.session_state["selected_lineup"],
    key="lineup_select",
)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("Use optimization suggestion", use_container_width=True) and default_lineup:
        st.session_state["selected_lineup"] = default_lineup
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
with btn_col2:
    if st.button("Use ML (GD/min) suggestion", use_container_width=True) and ml_lineup:
        st.session_state["selected_lineup"] = ml_lineup
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

selected = st.session_state["selected_lineup"]

if len(selected) != 4:
    st.warning("Please pick exactly 4 players")
    st.stop()

# Scores
selected_score = lineup_score(selected, stint_num, wowy_factor)
selected_gd_vs_opp = gd_per_minute_vs_opponent(selected, opponent)

metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("Selected lineup normalized GD (sum)", f"{selected_score:.2f}")
metrics_col2.metric("Recommended lineup normalized GD", f"{default_score:.2f}" if not np.isnan(default_score) else "n/a")
metrics_col3.metric("GD/min vs opponent", f"{selected_gd_vs_opp:.3f}" if not np.isnan(selected_gd_vs_opp) else "n/a")

st.markdown("### Suggested lineups")
sg1, sg2 = st.columns(2)
with sg1:
    st.markdown("**Optimization model (stint GD + WOWY)**")
    st.write(", ".join(default_lineup) if default_lineup else "No valid lineup")
with sg2:
    st.markdown("**ML model (GD/min vs opponent)**")
    st.write(", ".join(ml_lineup) if ml_lineup else "No valid lineup")

st.markdown("### Player details for stint")
player_rows = []
for p in selected:
    row = player_df[player_df["player"] == p].iloc[0]
    player_rows.append({
        "player": p,
        "rating": row["rating"],
        f"stint{stint_num}_gd": stint_gd.get(p, {}).get(stint_num, 0.0),
        "wowy": wowy_dict.get(p, 0.0),
    })
st.dataframe(pd.DataFrame(player_rows))

st.markdown("### Visuals")
viz_col1, viz_col2 = st.columns(2)

# Normalized GD components for selected lineup
norm_gd_df = pd.DataFrame([
    {
        "player": p,
        "stint_gd": stint_gd.get(p, {}).get(stint_num, 0.0),
        "wowy_contrib": wowy_dict.get(p, 0.0) * wowy_factor,
        "total": stint_gd.get(p, {}).get(stint_num, 0.0) + wowy_dict.get(p, 0.0) * wowy_factor,
    }
    for p in selected
])
with viz_col1:
    st.subheader("Selected lineup normalized GD parts")
    st.bar_chart(norm_gd_df.set_index("player")[["stint_gd", "wowy_contrib", "total"]])

# GD/min vs opponent for selected lineup
opp_subset = player_opponent_stats[
    (player_opponent_stats["opponent"] == opponent) & (player_opponent_stats["player"].isin(selected))
][["player", "total_plus_minus_per_min"]]
with viz_col2:
    st.subheader("GD/min vs opponent (selected players)")
    if opp_subset.empty:
        st.info("No matchup data for these players vs this opponent.")
    else:
        st.bar_chart(opp_subset.set_index("player"))

st.markdown("### GD/min vs opponent (linearmodel-style)")
opp_table = player_opponent_stats[
    (player_opponent_stats["opponent"] == opponent) & (player_opponent_stats["player"].isin(selected))
][["player", "minutes", "goal_margin", "total_plus_minus_per_min"]]
st.dataframe(opp_table)

st.caption("Scores mirror the lineup_optimization normalized GD logic (stint_gd + WOWY*factor) and the linearmodel GD/min aggregation per opponent.")

