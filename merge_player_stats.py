import pandas as pd

# Define data directory
DATA_DIR = 'data/'

# Load the datasets
player_data = pd.read_csv(DATA_DIR + 'player_data.csv')
stint_data = pd.read_csv(DATA_DIR + 'stint_data.csv')

# Function to extract team from player name
def extract_team(player):
    return '_'.join(player.split('_')[:-1])

# Prepare player data
player_data['team'] = player_data['player'].apply(extract_team)

# Create a list to hold the expanded data
expanded_rows = []

# For each stint, expand to individual players
for _, row in stint_data.iterrows():
    game_id = row['game_id']
    h_team = row['h_team']
    a_team = row['a_team']
    minutes = row['minutes']
    h_goals = row['h_goals']
    a_goals = row['a_goals']
    
    # Home players
    for i in range(1, 5):
        player = row[f'home{i}']
        expanded_rows.append({
            'player': player,
            'team': h_team,
            'minutes': minutes,
            'goals_for': h_goals,
            'goals_against': a_goals
        })
    
    # Away players
    for i in range(1, 5):
        player = row[f'away{i}']
        expanded_rows.append({
            'player': player,
            'team': a_team,
            'minutes': minutes,
            'goals_for': a_goals,
            'goals_against': h_goals
        })

# Create expanded dataframe
expanded_df = pd.DataFrame(expanded_rows)

# Group by player and sum
player_stats = expanded_df.groupby('player').agg({
    'goals_for': 'sum',
    'goals_against': 'sum',
    'minutes': 'sum'
}).reset_index()

# Merge with player data to get rating
player_stats = player_stats.merge(player_data[['player', 'rating']], on='player', how='left')

# Compute plus/minus per minute
player_stats['plus_minus_per_min'] = (player_stats['goals_for'] - player_stats['goals_against']) / player_stats['minutes']

# Select required columns
output_df = player_stats[['player', 'plus_minus_per_min', 'rating']]

# Save to CSV
output_df.to_csv('player_stats.csv', index=False)

print("Player stats CSV created: player_stats.csv")