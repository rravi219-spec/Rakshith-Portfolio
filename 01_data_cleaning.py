"""
01_DATA_CLEANING.py
Cleans and prepares football player data from multiple sources
"""

import pandas as pd
import numpy as np

print("="*80)
print("STEP 1: DATA CLEANING - Top 5 European Leagues")
print("="*80)

# Load datasets
print("\n📂 Loading datasets...")
df_standard = pd.read_csv('data/raw/player_standard_stats.csv')
df_shooting = pd.read_csv('data/raw/player_shooting.csv')
df_passing = pd.read_csv('data/raw/player_passing.csv')
df_defense = pd.read_csv('data/raw/player_defense.csv')

print(f"✓ Standard: {len(df_standard)} records")
print(f"✓ Shooting: {len(df_shooting)} records")
print(f"✓ Passing: {len(df_passing)} records")
print(f"✓ Defense: {len(df_defense)} records")

# Merge all datasets
print("\n🔗 Merging datasets...")
merge_keys = ['player', 'season', 'comp', 'squad']

combined = df_standard.merge(df_shooting, on=merge_keys, how='left', suffixes=('', '_shoot'))
combined = combined.merge(df_passing, on=merge_keys, how='left', suffixes=('', '_pass'))
combined = combined.merge(df_defense, on=merge_keys, how='left', suffixes=('', '_def'))

# Remove duplicate columns
print("\n🧹 Cleaning duplicate columns...")
cols_to_drop = [col for col in combined.columns if col.endswith(('_shoot', '_pass', '_def'))]
combined.drop(columns=cols_to_drop, inplace=True)

# Filter by minimum minutes (450 = ~5 matches)
print("\n⏱️ Filtering by minimum minutes...")
combined = combined[combined['min'] >= 450]

# Keep only current players (2024 season)
print("\n👥 Keeping only current players...")
current_players = combined[combined['season'] == 2024]['player'].unique()
combined = combined[combined['player'].isin(current_players)]

# Handle missing values
print("\n🔧 Handling missing values...")
numeric_cols = combined.select_dtypes(include=[np.number]).columns
combined[numeric_cols] = combined[numeric_cols].fillna(0)
combined['nation'].fillna('Unknown', inplace=True)

# Calculate per 90 stats
print("\n📊 Calculating per 90 statistics...")
combined['ninety_mins_played'] = combined['min'] / 90
combined['goals_per_90'] = combined['goals'] / combined['ninety_mins_played'].replace(0, 1)
combined['assists_per_90'] = combined['assists'] / combined['ninety_mins_played'].replace(0, 1)
combined['xg_per_90'] = combined['xg'] / combined['ninety_mins_played'].replace(0, 1)
combined['xag_per_90'] = combined['xa'] / combined['ninety_mins_played'].replace(0, 1)
combined['shots_per_90'] = combined['shots'] / combined['ninety_mins_played'].replace(0, 1)
combined['progressive_actions_per_90'] = (combined['progressive_passes'] + combined['progressive_carries']) / combined['ninety_mins_played'].replace(0, 1)
combined['defensive_actions_per_90'] = (combined['tackles'] + combined['interceptions']) / combined['ninety_mins_played'].replace(0, 1)

# Summary
print("\n" + "="*80)
print("✅ DATA CLEANING COMPLETE")
print("="*80)
print(f"Total records: {len(combined):,}")
print(f"Unique players: {combined['player'].nunique():,}")
print(f"Seasons: {combined['season'].min()} - {combined['season'].max()}")
print(f"Features: {len(combined.columns)}")

# Save
output_path = 'data/processed/big5_leagues_cleaned.csv'
combined.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")
print("="*80)
