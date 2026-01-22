# SportsModule

Analytics tools for stint- and lineup-level wheelchair rugby (murderball) analysis, with:
- Data prep and normalization (`player_stats.ipynb`)
- Line-level stats (`line_stats.ipynb`)
- Lineup optimization with Gurobi MILP (`lineup_optimization.ipynb`)
- Interactive dashboard (`streamlit_dashboard.py`)
- Linear model exploration (`linearmodel.ipynb`)
- Ad hoc analysis (`dataAnalysis.ipynb`)

## Data
Input CSVs (in `data/`):
- `stint_data.csv`: raw stint records (teams, players, goals, minutes)
- `player_data.csv`: per-player ratings, WOWY, stint_gd columns (stint1_gd … stint16_gd)

Outputs commonly written:
- `player_data.csv` (updated) from `player_stats.ipynb`

## Environment
Python 3.11+ recommended.

Install base deps (matches `requirements.txt`):
```bash
pip install -r requirements.txt
```

### Gurobi
`lineup_optimization.ipynb` uses Gurobi exclusively. Install Gurobi and set up a license (academic/limited OK):
```bash
pip install gurobipy
grbgetkey <your-license-key>   # per Gurobi instructions
```
If Gurobi is missing, the notebook will error; install it before running optimization.

## Workflows

### 1) Normalize player-level metrics (`player_stats.ipynb`)
- Expands `stint_data` to player-stint rows.
- Computes team strength from standings, z-scores it, builds expected GD, and z-scores residual goal margins to produce `normalized_gd`.
- Aggregates per-player stint GD (`stint_gd`), basic stats, and WOWY.
- Adds stint_gd columns back into `player_data.csv`.
- Visualizations: WOWY bars (Canada), best players per stint (Canada/Japan).

Run cells top-to-bottom; ensure `data/stint_data.csv` and `data/player_data.csv` are present.

### 2) Lineup optimization with Gurobi (`lineup_optimization.ipynb`)
- Loads `player_data.csv` (expects stint_gd and wowy columns).
- Gurobi MILP: select 4 players per stint maximizing `stint_gd + WOWY * wowy_factor` under rating cap (default 8).
- Prints optimized lineup per stint for the chosen team (default Canada).
- Requires Gurobi installed and licensed.

### 3) Interactive dashboard (`streamlit_dashboard.py`)
- Streamlit app to tweak lineups, compare optimizer suggestion vs ML (GD/min) suggestion, and view metrics/visuals.
- Uses stint_gd + WOWY weighting (slider for WOWY factor) and opponent GD/min from `linearmodel`-style aggregation.

Run:
```bash
streamlit run streamlit_dashboard.py
```

### 4) Linear model exploration (`linearmodel.ipynb`)
- Ridge regression over player-opponent interaction features to estimate GD/min contributions.
- Provides opponent-specific lineup suggestions based on coefficients.

### 5) Additional analysis (`dataAnalysis.ipynb`)
- Miscellaneous exploratory work; may be slow/large.

## Notes on normalization
- Team strengths are z-scored from standings.
- Expected GD per stint = z(team_strength_home) – z(team_strength_away).
- `normalized_gd` = z-score of residual (goal_margin – expected_gd), unit variance.
- Strong teams meeting expectation land near 0; overperformance >0, underperformance <0.

## Troubleshooting
- Missing Gurobi: install `gurobipy` and set a license.
- Missing Streamlit: `pip install streamlit`.
- Permission errors from prior attempts: ensure you run commands outside restricted sandboxes and with correct env activated.
