# Lineup prediction + fatigue feature + lineup search (RATING CAP ENFORCED)
# Cap: sum(rating) for the 4-player lineup MUST be <= 8.0 (always enforced)

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

PLAYER_CSV = "./data/player_data.csv"
STINT_CSV  = "./data/stint_data.csv"

HOME_COLS = ["home1", "home2", "home3", "home4"]
AWAY_COLS = ["away1", "away2", "away3", "away4"]

RATING_CAP = 8.0


# ---------------------------
# 1) Load
# ---------------------------
players = pd.read_csv(PLAYER_CSV)
stints  = pd.read_csv(STINT_CSV)

required_player_cols = {"player", "team", "rating", "wowy"}
required_stint_cols = {"game_id", "minutes", "h_goals", "a_goals", *HOME_COLS, *AWAY_COLS}

missing_p = required_player_cols - set(players.columns)
missing_s = required_stint_cols - set(stints.columns)
if missing_p:
    raise ValueError(f"player_data.csv missing columns: {sorted(missing_p)}")
if missing_s:
    raise ValueError(f"stint_data.csv missing columns: {sorted(missing_s)}")


# ---------------------------
# 2) Timeline + fatigue
# ---------------------------
def add_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Assumes rows are ordered chronologically within each game."""
    out = df.copy()
    out["minutes"] = out["minutes"].astype(float)
    out["start_min"] = out.groupby("game_id")["minutes"].cumsum() - out["minutes"]
    out["end_min"]   = out["start_min"] + out["minutes"]
    return out


def compute_fatigue_features(df: pd.DataFrame, tau_seconds: float = 90.0) -> pd.DataFrame:
    """
    Fatigue state per player:
        F_new = exp(-rest/tau) * F_old + shift_len_seconds

    Aggregated into lineup-level mean/max/min fatigue for home and away each stint.
    """
    fatigue = defaultdict(float)            # (game_id, player) -> fatigue
    last_end = defaultdict(lambda: 0.0)     # (game_id, player) -> last end time (seconds)

    feat_rows = []
    for _, row in df.iterrows():
        gid = row["game_id"]
        start_s = float(row["start_min"]) * 60.0
        end_s   = float(row["end_min"]) * 60.0
        shift_len_s = float(row["minutes"]) * 60.0

        home_players = [row[c] for c in HOME_COLS]
        away_players = [row[c] for c in AWAY_COLS]
        all_players  = home_players + away_players

        per_player_f = {}
        for p in all_players:
            key = (gid, p)
            rest = max(0.0, start_s - last_end[key])
            decay = float(np.exp(-rest / tau_seconds))
            new_f = decay * fatigue[key] + shift_len_s
            fatigue[key] = new_f
            last_end[key] = end_s
            per_player_f[p] = new_f

        home_f = np.array([per_player_f[p] for p in home_players], dtype=float)
        away_f = np.array([per_player_f[p] for p in away_players], dtype=float)

        feat_rows.append({
            "fatigue_home_mean": float(home_f.mean()),
            "fatigue_home_max":  float(home_f.max()),
            "fatigue_home_min":  float(home_f.min()),
            "fatigue_away_mean": float(away_f.mean()),
            "fatigue_away_max":  float(away_f.max()),
            "fatigue_away_min":  float(away_f.min()),
        })

    feats = pd.DataFrame(feat_rows, index=df.index)
    return pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)


st = add_timeline(stints)
st = compute_fatigue_features(st, tau_seconds=90.0)


# ---------------------------
# 3) Player priors + target
# ---------------------------
rating = dict(zip(players["player"], players["rating"]))
wowy   = dict(zip(players["player"], players["wowy"]))

def add_lineup_priors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def agg(vals):
        a = np.array(vals, dtype=float)
        return float(a.sum()), float(a.mean())

    h_r_sum, h_r_mean, a_r_sum, a_r_mean = [], [], [], []
    h_w_sum, h_w_mean, a_w_sum, a_w_mean = [], [], [], []

    for _, r in out.iterrows():
        hp = [r[c] for c in HOME_COLS]
        ap = [r[c] for c in AWAY_COLS]

        hrat = [rating.get(p, 0.0) for p in hp]
        arat = [rating.get(p, 0.0) for p in ap]
        hw   = [wowy.get(p, 0.0)   for p in hp]
        aw   = [wowy.get(p, 0.0)   for p in ap]

        s, m = agg(hrat); h_r_sum.append(s); h_r_mean.append(m)
        s, m = agg(arat); a_r_sum.append(s); a_r_mean.append(m)
        s, m = agg(hw);   h_w_sum.append(s); h_w_mean.append(m)
        s, m = agg(aw);   a_w_sum.append(s); a_w_mean.append(m)

    out["home_rating_sum"]  = h_r_sum
    out["home_rating_mean"] = h_r_mean
    out["away_rating_sum"]  = a_r_sum
    out["away_rating_mean"] = a_r_mean

    out["home_wowy_sum"]  = h_w_sum
    out["home_wowy_mean"] = h_w_mean
    out["away_wowy_sum"]  = a_w_sum
    out["away_wowy_mean"] = a_w_mean
    return out

st = add_lineup_priors(st)
st["net_goals_per60"] = (st["h_goals"] - st["a_goals"]) / st["minutes"].clip(lower=1e-6) * 60.0


# ---------------------------
# 4) Encode lineups + numeric features
# ---------------------------
home_lists = st[HOME_COLS].values.tolist()
away_lists = st[AWAY_COLS].values.tolist()

mlb_home = MultiLabelBinarizer()
mlb_away = MultiLabelBinarizer()

X_home = mlb_home.fit_transform(home_lists)
X_away = mlb_away.fit_transform(away_lists)

home_feature_names = [f"home_{p}" for p in mlb_home.classes_]
away_feature_names = [f"away_{p}" for p in mlb_away.classes_]

num_cols = [
    "minutes",
    "fatigue_home_mean", "fatigue_home_max", "fatigue_home_min",
    "fatigue_away_mean", "fatigue_away_max", "fatigue_away_min",
    "home_rating_sum", "home_rating_mean", "away_rating_sum", "away_rating_mean",
    "home_wowy_sum", "home_wowy_mean", "away_wowy_sum", "away_wowy_mean",
]

X_num = st[num_cols].astype(float).values
X = np.hstack([X_home, X_away, X_num])

y = st["net_goals_per60"].astype(float).values
groups = st["game_id"].values

feature_names = home_feature_names + away_feature_names + num_cols


# ---------------------------
# 5) Train/test split by game + train model
# ---------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.08,
    max_iter=400,
    l2_regularization=0.2,
    random_state=42,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 :", r2_score(y_test, pred))

# Optional: permutation importance
perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
imp = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)
print("\nTop 15 features:")
print(imp.head(15).to_string(index=False))


# ---------------------------
# 6) Lineup scoring + best-lineup search (CAP ALWAYS ENFORCED)
# ---------------------------
def lineup_priors(player_list):
    r = np.array([rating.get(p, 0.0) for p in player_list], dtype=float)
    w = np.array([wowy.get(p, 0.0)   for p in player_list], dtype=float)
    return float(r.sum()), float(r.mean()), float(w.sum()), float(w.mean())

def build_feature_row(home_players, away_players, minutes: float, fatigue_context: dict):
    home_oh = mlb_home.transform([home_players])
    away_oh = mlb_away.transform([away_players])

    hr_sum, hr_mean, hw_sum, hw_mean = lineup_priors(home_players)
    ar_sum, ar_mean, aw_sum, aw_mean = lineup_priors(away_players)

    num = np.array([[
        float(minutes),
        float(fatigue_context["fatigue_home_mean"]),
        float(fatigue_context["fatigue_home_max"]),
        float(fatigue_context["fatigue_home_min"]),
        float(fatigue_context["fatigue_away_mean"]),
        float(fatigue_context["fatigue_away_max"]),
        float(fatigue_context["fatigue_away_min"]),
        hr_sum, hr_mean, ar_sum, ar_mean,
        hw_sum, hw_mean, aw_sum, aw_mean,
    ]], dtype=float)

    return np.hstack([home_oh, away_oh, num]), hr_sum


def best_lineup_for_stint(
    team: str,
    opponent_lineup: list,
    minutes: float = 4.0,
    fatigue_context: dict | None = None,
):
    """
    Returns best 4-player lineup for `team` vs a known opponent lineup of 4 players,
    with hard constraint:
        sum(rating of the 4 players) <= RATING_CAP
    """
    roster = players.loc[players["team"] == team, "player"].tolist()
    if len(roster) < 4:
        raise ValueError(f"Team {team} has only {len(roster)} players in player_data.csv")

    if fatigue_context is None:
        fatigue_context = {
            "fatigue_home_mean": 180.0,
            "fatigue_home_max":  240.0,
            "fatigue_home_min":  120.0,
            "fatigue_away_mean": 180.0,
            "fatigue_away_max":  240.0,
            "fatigue_away_min":  120.0,
        }

    best = None
    for home_lineup in itertools.combinations(roster, 4):
        xrow, hr_sum = build_feature_row(list(home_lineup), opponent_lineup, float(minutes), fatigue_context)

        if hr_sum > RATING_CAP:
            continue

        yhat = float(model.predict(xrow)[0])
        if (best is None) or (yhat > best["pred_net_goals_per60"]):
            best = {
                "team": team,
                "home_lineup": list(home_lineup),
                "opponent_lineup": opponent_lineup,
                "minutes": float(minutes),
                "pred_net_goals_per60": yhat,
                "home_rating_sum": hr_sum,
            }

    if best is None:
        raise ValueError(f"No feasible 4-player lineup found for team={team} under rating cap {RATING_CAP}")

    # Safety check
    assert best["home_rating_sum"] <= RATING_CAP + 1e-9
    return best


# ---------------------------
# 7) Example usage
# ---------------------------
print("\nTeams in player_data.csv:", sorted(players["team"].unique())[:30], "...")
example_row = st.iloc[0]
opp_lineup = [example_row[c] for c in AWAY_COLS]
print("Example opponent lineup:", opp_lineup)

TEAM = "Canada"  # change if needed, e.g. "CAN"
best = best_lineup_for_stint(TEAM, opp_lineup, minutes=float(example_row["minutes"]))
print("\nBest lineup (cap enforced):")
print(best)