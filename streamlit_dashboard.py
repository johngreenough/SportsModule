#!/usr/bin/env python
"""
Team Canada Wheelchair Rugby - Lineup Dashboard

A coaching dashboard for Team Canada's wheelchair rugby team that provides:
- Lineup suggestions from optimization and ML models
- Player effectiveness heatmaps vs each opponent
- Rating constraints with female player bonus (+0.5 per female)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import Ridge

# Configuration
DATA_DIR = "data/"
PLAYER_CSV = DATA_DIR + "player_data.csv"
STINT_CSV = DATA_DIR + "stint_data.csv"
TEAM = "Canada"
BASE_RATING_LIMIT = 8.0
FEMALE_BONUS = 0.5

st.set_page_config(
    page_title="Team Canada Wheelchair Rugby",
    page_icon="🏆",
    layout="wide"
)


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """Load player and stint data, extract Canada-specific information."""
    player_df = pd.read_csv(PLAYER_CSV)
    stint_df = pd.read_csv(STINT_CSV)

    # Add team column if missing
    if "team" not in player_df.columns:
        player_df["team"] = player_df["player"].apply(lambda p: "_".join(p.split("_")[:-1]))

    # Filter to Canada players
    canada_players = player_df[player_df["team"] == TEAM].copy()

    # Build stint_gd dict from columns stint{i}_gd
    stint_gd = {}
    max_stint = 0
    for _, row in player_df.iterrows():
        player = row["player"]
        stint_gd[player] = {}
        for i in range(1, 17):
            col = f"stint{i}_gd"
            if col in row.index and pd.notna(row[col]):
                stint_gd[player][i] = float(row[col])
                max_stint = max(max_stint, i)

    # WOWY values
    wowy_dict = {}
    if "wowy" in player_df.columns:
        wowy_dict = dict(zip(player_df["player"], player_df["wowy"].fillna(0)))
    else:
        wowy_dict = {p: 0 for p in player_df["player"]}

    # Get all teams for opponent selection
    all_teams = sorted(player_df["team"].unique())
    opponents = [t for t in all_teams if t != TEAM]

    return player_df, canada_players, stint_df, stint_gd, wowy_dict, max_stint, opponents


@st.cache_data(show_spinner=False)
def build_player_opponent_stats(_stint_df: pd.DataFrame) -> pd.DataFrame:
    """Build player-opponent statistics from stint data."""
    records = []
    for _, row in _stint_df.iterrows():
        h_team = row["h_team"]
        a_team = row["a_team"]
        minutes = row["minutes"]
        h_goals = row["h_goals"]
        a_goals = row["a_goals"]

        # Home players
        for i in range(1, 5):
            player = row[f"home{i}"]
            records.append({
                "player": player,
                "team": h_team,
                "opponent": a_team,
                "minutes": minutes,
                "goals_for": h_goals,
                "goals_against": a_goals,
                "goal_margin": h_goals - a_goals,
            })

        # Away players
        for i in range(1, 5):
            player = row[f"away{i}"]
            records.append({
                "player": player,
                "team": a_team,
                "opponent": h_team,
                "minutes": minutes,
                "goals_for": a_goals,
                "goals_against": h_goals,
                "goal_margin": a_goals - h_goals,
            })

    df = pd.DataFrame(records)
    grouped = df.groupby(["player", "team", "opponent"], as_index=False).agg(
        minutes=("minutes", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        goal_margin=("goal_margin", "sum"),
        n_stints=("goal_margin", "count"),
    )
    grouped["plus_minus_per_min"] = grouped["goal_margin"] / grouped["minutes"].replace(0, np.nan)
    grouped["plus_minus_per_min"] = grouped["plus_minus_per_min"].fillna(0)
    return grouped


@st.cache_data(show_spinner=False)
def compute_player_coefficients(_stint_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute player-opponent coefficients using Ridge regression.
    Replicates the linearmodel.ipynb approach.
    """
    # Create interaction features
    records = []
    for _, row in _stint_df.iterrows():
        minutes = row["minutes"]
        if minutes <= 0:
            continue

        # Home team perspective
        home_record = {
            "target": (row["h_goals"] - row["a_goals"]) / minutes,
            "minutes": minutes,
        }
        for i in range(1, 5):
            player = row[f"home{i}"]
            opponent = row["a_team"]
            home_record[f"{player}_vs_{opponent}"] = 1
        records.append(home_record)

        # Away team perspective
        away_record = {
            "target": (row["a_goals"] - row["h_goals"]) / minutes,
            "minutes": minutes,
        }
        for i in range(1, 5):
            player = row[f"away{i}"]
            opponent = row["h_team"]
            away_record[f"{player}_vs_{opponent}"] = 1
        records.append(away_record)

    interaction_df = pd.DataFrame(records).fillna(0)

    # Prepare features
    feature_cols = [c for c in interaction_df.columns if c not in ["target", "minutes"]]
    X = interaction_df[feature_cols]
    y = interaction_df["target"]
    weights = interaction_df["minutes"]

    # Train Ridge regression
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X, y, sample_weight=weights)

    # Extract coefficients
    coef_records = []
    for feat, coef in zip(feature_cols, model.coef_):
        if "_vs_" in feat:
            parts = feat.split("_vs_")
            player = parts[0]
            opponent = parts[1]
            coef_records.append({
                "player": player,
                "opponent": opponent,
                "coefficient": coef
            })

    return pd.DataFrame(coef_records)


# =============================================================================
# Lineup Functions
# =============================================================================

def get_rating_limit(selected_players: List[str], female_players: List[str]) -> float:
    """Calculate rating limit based on female players in selection."""
    females_in_selection = len([p for p in selected_players if p in female_players])
    return BASE_RATING_LIMIT + (FEMALE_BONUS * females_in_selection)


def optimize_lineup(
    canada_players_df: pd.DataFrame,
    stint_num: int,
    stint_gd: Dict,
    wowy_dict: Dict,
    female_players: List[str],
    wowy_factor: float = 1.0
) -> Tuple[Optional[List[str]], float, float]:
    """
    Optimize lineup for a stint with female bonus rating adjustment.
    Returns (lineup, score, rating_sum) or (None, nan, nan) if no valid lineup.
    """
    players = canada_players_df["player"].tolist()
    ratings = dict(zip(canada_players_df["player"], canada_players_df["rating"]))

    # Get players with GD for this stint
    valid_players = [p for p in players if stint_num in stint_gd.get(p, {})]
    if len(valid_players) < 4:
        return None, float("nan"), float("nan")

    best_score = float("-inf")
    best_lineup = None
    best_rating_sum = None

    for combo in combinations(valid_players, 4):
        # Calculate rating limit for this combo (depends on females in combo)
        females_in_combo = len([p for p in combo if p in female_players])
        adjusted_limit = BASE_RATING_LIMIT + (FEMALE_BONUS * females_in_combo)

        rating_sum = sum(ratings[p] for p in combo)
        if rating_sum <= adjusted_limit:
            score = sum(
                stint_gd[p][stint_num] + wowy_dict.get(p, 0) * wowy_factor
                for p in combo
            )
            if score > best_score:
                best_score = score
                best_lineup = list(combo)
                best_rating_sum = rating_sum

    return best_lineup, best_score, best_rating_sum


def get_ml_recommendation(
    player_opponent_stats: pd.DataFrame,
    opponent: str,
    female_players: List[str],
    canada_players_df: pd.DataFrame
) -> Tuple[List[str], float, float]:
    """
    Get ML-based lineup recommendation (top 4 by GD/min vs opponent).
    Respects rating constraint with female bonus.
    """
    # Filter to Canada vs this opponent
    canada_vs_opp = player_opponent_stats[
        (player_opponent_stats["team"] == TEAM) &
        (player_opponent_stats["opponent"] == opponent)
    ].copy()

    if canada_vs_opp.empty:
        return [], float("nan"), float("nan")

    # Sort by plus_minus_per_min
    canada_vs_opp = canada_vs_opp.sort_values("plus_minus_per_min", ascending=False)

    # Get ratings
    ratings = dict(zip(canada_players_df["player"], canada_players_df["rating"]))

    # Greedy selection respecting rating constraint
    selected = []
    total_rating = 0.0
    total_gd = 0.0

    for _, row in canada_vs_opp.iterrows():
        if len(selected) >= 4:
            break

        player = row["player"]
        player_rating = ratings.get(player, 0)

        # Check if adding this player would exceed limit
        potential_selection = selected + [player]
        females_in_selection = len([p for p in potential_selection if p in female_players])
        limit = BASE_RATING_LIMIT + (FEMALE_BONUS * females_in_selection)

        if total_rating + player_rating <= limit:
            selected.append(player)
            total_rating += player_rating
            total_gd += row["plus_minus_per_min"]

    return selected, total_gd, total_rating


# =============================================================================
# Visualization Functions
# =============================================================================

def create_effectiveness_heatmap(player_coefficients: pd.DataFrame) -> go.Figure:
    """Create an interactive heatmap of Canada player effectiveness vs opponents."""
    # Filter to Canada players
    canada_coefs = player_coefficients[
        player_coefficients["player"].str.startswith("Canada")
    ].copy()

    if canada_coefs.empty:
        return None

    # Pivot for heatmap
    pivot = canada_coefs.pivot(
        index="player",
        columns="opponent",
        values="coefficient"
    )

    # Sort players by average effectiveness
    pivot["avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("avg", ascending=False)
    pivot = pivot.drop(columns=["avg"])

    # Create heatmap
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn",
        zmin=-0.5,
        zmax=0.5,
        labels={"color": "GD/min contribution"},
        title="Player Effectiveness vs Opponents (GD/min)"
    )

    fig.update_layout(
        xaxis_title="Opponent",
        yaxis_title="Player",
        height=500
    )

    return fig


def create_lineup_comparison_chart(
    optim_lineup: List[str],
    ml_lineup: List[str],
    ratings: Dict[str, float]
) -> go.Figure:
    """Create a bar chart comparing the two recommended lineups."""
    data = []

    for player in optim_lineup:
        data.append({
            "Player": player,
            "Rating": ratings.get(player, 0),
            "Source": "Optimization"
        })

    for player in ml_lineup:
        data.append({
            "Player": player,
            "Rating": ratings.get(player, 0),
            "Source": "ML Model"
        })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Player",
        y="Rating",
        color="Source",
        barmode="group",
        title="Lineup Comparison - Player Ratings"
    )

    return fig


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    # Load data
    player_df, canada_players, stint_df, stint_gd, wowy_dict, max_stint, opponents = load_data()
    player_opponent_stats = build_player_opponent_stats(stint_df)
    player_coefficients = compute_player_coefficients(stint_df)

    # Get ratings dict
    ratings = dict(zip(canada_players["player"], canada_players["rating"]))
    canada_player_list = sorted(canada_players["player"].tolist())

    # ==========================================================================
    # Header
    # ==========================================================================
    st.title("🏆 Team Canada Wheelchair Rugby")
    st.markdown("### Lineup Optimization Dashboard")

    # ==========================================================================
    # Sidebar
    # ==========================================================================
    st.sidebar.header("Configuration")

    # Opponent selection
    opponent = st.sidebar.selectbox(
        "Select Opponent",
        opponents,
        index=0
    )

    # Stint selection
    stint_num = st.sidebar.slider(
        "Stint Number",
        min_value=1,
        max_value=max_stint,
        value=1
    )

    # WOWY factor
    wowy_factor = st.sidebar.slider(
        "WOWY Factor",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )

    st.sidebar.markdown("---")

    # Female player selection
    st.sidebar.subheader("Female Players")
    st.sidebar.caption("Check players who are female (+0.5 rating bonus each)")

    female_players = []
    for player in canada_player_list:
        player_short = player.replace("Canada_", "")
        rating = ratings.get(player, 0)
        if st.sidebar.checkbox(f"{player_short} (Rating: {rating})", key=f"female_{player}"):
            female_players.append(player)

    # Show current rating limit
    base_limit = BASE_RATING_LIMIT
    female_bonus_total = FEMALE_BONUS * len(female_players)
    current_limit = base_limit + female_bonus_total

    st.sidebar.markdown("---")
    st.sidebar.metric(
        "Rating Limit",
        f"{current_limit:.1f}",
        delta=f"+{female_bonus_total:.1f} female bonus" if female_bonus_total > 0 else None
    )

    # ==========================================================================
    # Main Content
    # ==========================================================================

    # Get recommendations
    optim_lineup, optim_score, optim_rating = optimize_lineup(
        canada_players, stint_num, stint_gd, wowy_dict, female_players, wowy_factor
    )

    ml_lineup, ml_gd, ml_rating = get_ml_recommendation(
        player_opponent_stats, opponent, female_players, canada_players
    )

    # Recommended Lineups Section
    st.markdown("## Recommended Lineups")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Optimization Model")
        st.caption("Based on stint GD + WOWY")
        if optim_lineup:
            for i, player in enumerate(optim_lineup, 1):
                player_short = player.replace("Canada_", "")
                is_female = "👩" if player in female_players else ""
                st.write(f"{i}. **{player_short}** {is_female} (Rating: {ratings.get(player, 0)})")
            st.metric("Total Rating", f"{optim_rating:.1f} / {get_rating_limit(optim_lineup, female_players):.1f}")
            st.metric("Optimization Score", f"{optim_score:.2f}")
        else:
            st.warning("No valid lineup found for this stint")

    with col2:
        st.markdown("### 🤖 ML Model")
        st.caption(f"Based on GD/min vs {opponent}")
        if ml_lineup:
            for i, player in enumerate(ml_lineup, 1):
                player_short = player.replace("Canada_", "")
                is_female = "👩" if player in female_players else ""
                st.write(f"{i}. **{player_short}** {is_female} (Rating: {ratings.get(player, 0)})")
            st.metric("Total Rating", f"{ml_rating:.1f} / {get_rating_limit(ml_lineup, female_players):.1f}")
            st.metric("Expected GD/min", f"{ml_gd:.3f}")
        else:
            st.warning(f"No data for Canada vs {opponent}")

    st.markdown("---")

    # Custom Lineup Selection
    st.markdown("## Custom Lineup Selection")

    # Initialize session state
    if "custom_lineup" not in st.session_state:
        st.session_state.custom_lineup = optim_lineup if optim_lineup else []

    custom_selection = st.multiselect(
        "Select 4 Players",
        options=canada_player_list,
        default=st.session_state.custom_lineup[:4] if st.session_state.custom_lineup else [],
        max_selections=4,
        format_func=lambda x: f"{x.replace('Canada_', '')} (Rating: {ratings.get(x, 0)})"
    )

    if custom_selection:
        custom_rating = sum(ratings.get(p, 0) for p in custom_selection)
        custom_limit = get_rating_limit(custom_selection, female_players)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Players", len(custom_selection))
        with col2:
            delta_color = "normal" if custom_rating <= custom_limit else "inverse"
            st.metric(
                "Total Rating",
                f"{custom_rating:.1f}",
                delta=f"Limit: {custom_limit:.1f}",
                delta_color=delta_color
            )
        with col3:
            if custom_rating <= custom_limit and len(custom_selection) == 4:
                st.success("✅ Valid lineup!")
            elif len(custom_selection) < 4:
                st.info(f"Select {4 - len(custom_selection)} more player(s)")
            else:
                st.error(f"⚠️ Over rating limit by {custom_rating - custom_limit:.1f}")

    st.markdown("---")

    # Effectiveness Heatmap
    st.markdown("## Player Effectiveness Heatmap")
    st.caption("Shows expected GD/min contribution for each player against each opponent")

    heatmap_fig = create_effectiveness_heatmap(player_coefficients)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.warning("Could not generate heatmap - no coefficient data available")

    st.markdown("---")

    # Player Details Table
    st.markdown("## Canada Player Details")

    # Build details table
    player_details = []
    for _, row in canada_players.iterrows():
        player = row["player"]

        # Get GD vs selected opponent
        opp_stats = player_opponent_stats[
            (player_opponent_stats["player"] == player) &
            (player_opponent_stats["opponent"] == opponent)
        ]
        gd_vs_opp = opp_stats["plus_minus_per_min"].values[0] if not opp_stats.empty else 0

        player_details.append({
            "Player": player.replace("Canada_", ""),
            "Rating": row["rating"],
            "Female": "Yes" if player in female_players else "No",
            f"GD/min vs {opponent}": round(gd_vs_opp, 3),
            "WOWY": round(wowy_dict.get(player, 0), 2),
            f"Stint {stint_num} GD": round(stint_gd.get(player, {}).get(stint_num, 0), 2)
        })

    details_df = pd.DataFrame(player_details)
    details_df = details_df.sort_values(f"GD/min vs {opponent}", ascending=False)

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True
    )

    # Footer
    st.markdown("---")
    st.caption(
        "Dashboard combines optimization model (stint GD + WOWY) and "
        "ML model (Ridge regression coefficients) for lineup recommendations. "
        "Rating limit: 8.0 base + 0.5 per female player selected."
    )


if __name__ == "__main__":
    main()
