"""
02_FEATURE_ENGINEERING.py
Creates position-specific features for ML models
"""

import pandas as pd
import numpy as np

print("="*80)
print("STEP 2: FEATURE ENGINEERING - Position-Specific Metrics")
print("="*80)

# Load cleaned data
df = pd.read_csv('data/processed/big5_leagues_cleaned.csv')
print(f"\n✓ Loaded {len(df):,} records")

# ============================================================================
# FORWARD FEATURES
# ============================================================================
print("\n⚽ Creating Forward Features...")

# Shot Efficiency
df['shot_efficiency'] = df['goals'] / df['shots'].replace(0, 1)

# Poacher Index (0-100)
df['goals_norm'] = (df['goals'] / df['goals'].max()) * 100
df['xg_norm'] = (df['xg'] / df['xg'].max()) * 100
df['poacher_index'] = (df['goals_norm'] * 0.5 + df['xg_norm'] * 0.5)
df.drop(columns=['goals_norm', 'xg_norm'], inplace=True)

# Complete Forward Index (0-100)
df['assists_norm'] = (df['assists'] / df['assists'].max()) * 100
df['complete_forward_index'] = (df['poacher_index'] * 0.5 + df['assists_norm'] * 0.5)
df.drop(columns=['assists_norm'], inplace=True)

# Clinical Finisher Score
df['clinical_finisher_score'] = ((df['goals'] - df['xg']) / df['xg'].replace(0, 1) * 50 + 
                                   df['shots_on_target_pct'] / 2).clip(0, 100)

print("  ✓ Created 4 forward features")

# ============================================================================
# MIDFIELDER FEATURES
# ============================================================================
print("\n⚙️ Creating Midfielder Features...")

# Final Third Involvement
df['final_third_involvement'] = (df['passes_into_final_third'] + 
                                   df['passes_into_penalty_area'] + 
                                   df['key_passes'])
df['final_third_involvement_per_90'] = df['final_third_involvement'] / df['ninety_mins_played'].replace(0, 1)

# Creative Output per 90
df['creative_output_per_90'] = (df['key_passes'] + df['xa']) / df['ninety_mins_played'].replace(0, 1)

# Progressive Midfielder Score (0-100)
prog_max = df['progressive_passes'].max()
final_third_max = df['passes_into_final_third'].max()
df['progressive_midfielder_score'] = (
    (df['progressive_passes'] / prog_max * 60) +
    (df['passes_into_final_third'] / final_third_max * 30)
).clip(0, 100)

# Attacking Midfielder Index (0-100)
key_max = df['key_passes'].max()
xag_max = df['xa'].max()
df['attacking_midfielder_index'] = (
    (df['key_passes'] / key_max * 50) +
    (df['xa'] / xag_max * 50)
).clip(0, 100)

# Defensive Midfielder Index (0-100)
tackles_max = df['tackles'].max()
int_max = df['interceptions'].max()
df['defensive_midfielder_index'] = (
    (df['tackles'] / tackles_max * 50) +
    (df['interceptions'] / int_max * 50)
).clip(0, 100)

# Box-to-Box Score (0-100)
df['box_to_box_midfielder_score'] = (
    df['progressive_midfielder_score'] * 0.5 +
    df['defensive_midfielder_index'] * 0.5
)

print("  ✓ Created 9 midfielder features")

# ============================================================================
# DEFENDER FEATURES
# ============================================================================
print("\n🛡️ Creating Defender Features...")

# Pure Defender Index (0-100)
df['pure_defender_index'] = (
    (df['tackles'] / tackles_max * 50) +
    (df['interceptions'] / int_max * 50)
).clip(0, 100)

# Ball-Playing Defender Index (0-100)
pass_max = df['pass_completion_pct'].max()
prog_pass_max = df['progressive_passes'].max()
df['ball_playing_defender_index'] = (
    (df['pass_completion_pct'] / pass_max * 30) +
    (df['progressive_passes'] / prog_pass_max * 30) +
    (df['pure_defender_index'] * 0.4)
).clip(0, 100)

# Attacking Fullback Index (0-100)
crosses_max = df['crosses_into_penalty_area'].max()
carries_max = df['progressive_carries'].max()
df['attacking_fullback_index'] = (
    (df['crosses_into_penalty_area'] / crosses_max * 50) +
    (df['progressive_carries'] / carries_max * 50)
).clip(0, 100)

# TRACKING BACK FEATURES (KEY INNOVATION!)
df['tracking_back_tackles'] = df['mid_third_tackles'] + df['att_third_tackles']
df['tracking_back_rate'] = (df['tracking_back_tackles'] / df['tackles'].replace(0, 1) * 100).clip(0, 100)
df['tracking_back_index'] = (df['tracking_back_tackles'] / df['tracking_back_tackles'].max() * 100).clip(0, 100)

# Defensive Work Rate (THE GAME CHANGER!)
df['defensive_work_rate'] = (
    df['def_third_tackles'] * 1.0 +
    df['mid_third_tackles'] * 1.2 +  # Tracking back!
    df['att_third_tackles'] * 1.5 +  # High press!
    df['interceptions'] * 0.8
)
df['defensive_work_rate_per_90'] = df['defensive_work_rate'] / df['ninety_mins_played'].replace(0, 1)

print("  ✓ Created 8 defender features (including tracking back!)")

# ============================================================================
# GENERAL FEATURES
# ============================================================================
print("\n📊 Creating General Features...")

# Consistency Score
df['starts_per_match'] = df['starts'] / df['mp'].replace(0, 1)
df['consistency_score'] = (
    df['starts_per_match'] * 60 +
    (df['min'] / df['mp'].replace(0, 1) / 90) * 40
).clip(0, 100)

# Discipline Score
df['cards_per_90'] = (df['yellow_cards'] + df['red_cards'] * 3) / df['ninety_mins_played'].replace(0, 1)
df['discipline_score'] = (100 - df['cards_per_90'] * 10).clip(0, 100)

print("  ✓ Created 4 general features")

# Summary
print("\n" + "="*80)
print("✅ FEATURE ENGINEERING COMPLETE")
print("="*80)
print(f"Total features created: 25+")
print(f"Final dataset shape: {df.shape}")
print(f"Features: {len(df.columns)}")

# Save
output_path = 'data/processed/big5_leagues_with_features.csv'
df.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")
print("="*80)
