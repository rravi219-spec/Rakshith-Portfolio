"""
FIXED CLASSIFICATION - ADDING WINGERS (RW/LW)
Properly separating:
- Forwards (Strikers): Poachers & False 9s
- Wingers: RW & LW (wide attackers)
- Midfielders: CAM, CM, CDM, Box-to-Box
"""

import pandas as pd
import numpy as np

print("="*80)
print("⚡ FIXED CLASSIFICATION - ADDING WINGERS")
print("="*80)

# Load dataset
df = pd.read_csv('/mnt/user-data/outputs/big5_leagues_WITH_TRACKING_BACK.csv')
df_2024 = df[df['season'] == 2024].copy()

# ============================================================================
# STEP 1: SEPARATE WINGERS FROM FORWARDS
# ============================================================================
print("\n" + "="*80)
print("🎯 STEP 1: IDENTIFYING WINGERS vs STRIKERS")
print("="*80)

forwards = df_2024[df_2024['general_position'] == 'Forward'].copy()
print(f"\nTotal 'Forwards' in dataset: {len(forwards)}")

# WINGER characteristics:
# - High progressive carries (run with ball down wings)
# - High crosses into penalty area
# - Moderate goals (not pure strikers)
# - High dribbles (wide players beat defenders)
# - Moderate assists (create from wide)

# STRIKER characteristics:
# - High goals per 90
# - Central positioning (low crosses)
# - High shots
# - High xG

# Use stats to identify wingers
def identify_winger_or_striker(row):
    # Winger indicators
    crosses_score = row['crosses_into_penalty_area']
    carries_score = row['progressive_carries']
    
    # Striker indicators  
    goals_score = row['goals_per_90']
    shots_score = row['shots_per_90']
    
    # If high crosses + carries = likely winger
    winger_score = (crosses_score * 3) + (carries_score * 2)
    
    # If high goals + shots + low crosses = likely striker
    striker_score = (goals_score * 50) + (shots_score * 20) - (crosses_score * 2)
    
    # Classify
    if winger_score > 30 and crosses_score >= 3:  # Clear winger
        return 'WINGER'
    elif striker_score > 50:  # Clear striker
        return 'STRIKER'
    elif crosses_score >= 5 or carries_score >= 30:  # Borderline winger
        return 'WINGER'
    else:
        return 'STRIKER'  # Default to striker

forwards['player_type'] = forwards.apply(identify_winger_or_striker, axis=1)

print(f"\n📊 Player Type Distribution:")
print(forwards['player_type'].value_counts())

# Split into strikers and wingers
strikers = forwards[forwards['player_type'] == 'STRIKER'].copy()
wingers = forwards[forwards['player_type'] == 'WINGER'].copy()

print(f"\nStrikers: {len(strikers)}")
print(f"Wingers: {len(wingers)}")

# ============================================================================
# STEP 2: CLASSIFY STRIKERS (POACHER VS FALSE 9)
# ============================================================================
print("\n" + "="*80)
print("⚽ STEP 2: CLASSIFYING STRIKERS")
print("="*80)

if len(strikers) > 0:
    # Normalize metrics
    strikers['goals_norm'] = (strikers['goals_per_90'] / strikers['goals_per_90'].max()) * 100
    strikers['shot_eff_norm'] = (strikers['shot_efficiency'] / strikers['shot_efficiency'].max()) * 100
    strikers['key_pass_norm'] = (strikers['key_passes'] / strikers['key_passes'].max()) * 100
    strikers['assists_norm'] = (strikers['assists_per_90'] / strikers['assists_per_90'].max()) * 100
    
    # POACHER SCORE
    strikers['poacher_score'] = (
        strikers['goals_norm'] * 0.40 +
        strikers['shot_eff_norm'] * 0.30 +
        strikers['poacher_index'] * 0.30
    )
    
    # FALSE 9 SCORE
    strikers['false9_score'] = (
        strikers['key_pass_norm'] * 0.35 +
        strikers['assists_norm'] * 0.30 +
        strikers['complete_forward_index'] * 0.35
    )
    
    # Classify
    strikers['striker_role'] = strikers.apply(
        lambda x: 'POACHER' if x['poacher_score'] > x['false9_score'] else 'FALSE 9',
        axis=1
    )
    
    print(f"\n📊 Striker Role Distribution:")
    print(strikers['striker_role'].value_counts())
    
    # TOP POACHERS
    print(f"\n🎯 TOP 10 POACHERS (Pure Strikers):")
    print("-" * 80)
    top_poachers = strikers.nlargest(10, 'poacher_score')[
        ['player', 'squad', 'comp', 'poacher_score', 'goals_per_90', 'striker_role']
    ]
    print(top_poachers.to_string(index=False))
    
    # TOP FALSE 9s
    print(f"\n🎨 TOP 10 FALSE 9s (Creative Strikers):")
    print("-" * 80)
    top_false9 = strikers.nlargest(10, 'false9_score')[
        ['player', 'squad', 'comp', 'false9_score', 'key_passes', 'assists_per_90', 'striker_role']
    ]
    print(top_false9.to_string(index=False))

# ============================================================================
# STEP 3: CLASSIFY WINGERS (RW/LW - can't distinguish without spatial data)
# ============================================================================
print("\n" + "="*80)
print("⚡ STEP 3: CLASSIFYING WINGERS")
print("="*80)

if len(wingers) > 0:
    # WINGER SCORE (general effectiveness)
    wingers['goals_norm_w'] = (wingers['goals_per_90'] / wingers['goals_per_90'].max()) * 100
    wingers['assists_norm_w'] = (wingers['assists_per_90'] / wingers['assists_per_90'].max()) * 100
    wingers['crosses_norm_w'] = (wingers['crosses_into_penalty_area'] / 
                                  wingers['crosses_into_penalty_area'].max()) * 100
    wingers['carries_norm_w'] = (wingers['progressive_carries'] / 
                                  wingers['progressive_carries'].max()) * 100
    wingers['key_pass_norm_w'] = (wingers['key_passes'] / wingers['key_passes'].max()) * 100
    
    # Overall winger effectiveness score
    wingers['winger_score'] = (
        wingers['goals_norm_w'] * 0.25 +
        wingers['assists_norm_w'] * 0.25 +
        wingers['crosses_norm_w'] * 0.20 +
        wingers['carries_norm_w'] * 0.15 +
        wingers['key_pass_norm_w'] * 0.15
    )
    
    # Can't distinguish RW vs LW without spatial data
    wingers['winger_role'] = 'RW/LW'
    
    print(f"\n⚡ TOP 15 WINGERS (RW/LW Combined):")
    print("-" * 80)
    top_wingers = wingers.nlargest(15, 'winger_score')[
        ['player', 'squad', 'comp', 'winger_score', 'goals_per_90', 'assists_per_90',
         'crosses_into_penalty_area', 'progressive_carries']
    ]
    print(top_wingers.to_string(index=False))

# ============================================================================
# STEP 4: RE-CLASSIFY MIDFIELDERS (MORE GRANULAR)
# ============================================================================
print("\n" + "="*80)
print("⚙️ STEP 4: RE-CLASSIFYING MIDFIELDERS")
print("="*80)

midfielders = df_2024[df_2024['general_position'] == 'Midfielder'].copy()
print(f"\nMidfielders in 2024: {len(midfielders)}")

# Recalculate with better thresholds
# CAM Score
midfielders['cam_score'] = (
    midfielders['attacking_midfielder_index'] * 0.50 +
    (midfielders['final_third_involvement_per_90'] / 
     midfielders['final_third_involvement_per_90'].max() * 100) * 0.30 +
    (midfielders['creative_output_per_90'] / 
     midfielders['creative_output_per_90'].max() * 100) * 0.20
)

# CDM Score
midfielders['cdm_score'] = (
    midfielders['defensive_midfielder_index'] * 0.50 +
    (midfielders['defensive_actions_per_90'] / 
     midfielders['defensive_actions_per_90'].max() * 100) * 0.30 +
    (midfielders['pass_completion_pct'] / 100 * 100) * 0.20
)

# CM (Central Midfielder) - Balanced
midfielders['cm_score'] = midfielders['progressive_midfielder_score']

# Box-to-Box Score
midfielders['b2b_score'] = midfielders['box_to_box_midfielder_score']

# Classify with better logic
def classify_midfielder_detailed(row):
    # Strong CAM (high attacking)
    if row['attacking_midfielder_index'] > 25:
        return 'CAM'
    
    # Strong CDM (high defensive)
    elif row['defensive_midfielder_index'] > 30:
        return 'CDM'
    
    # Box-to-Box (balanced, high on both)
    elif row['b2b_score'] > 20 and row['attacking_midfielder_index'] > 15 and row['defensive_midfielder_index'] > 15:
        return 'BOX-TO-BOX'
    
    # Default to CM (central midfielder)
    else:
        return 'CM'

midfielders['midfielder_role'] = midfielders.apply(classify_midfielder_detailed, axis=1)

print(f"\n📊 Midfielder Role Distribution:")
print(midfielders['midfielder_role'].value_counts())

# TOP BY ROLE
for role in ['CAM', 'CM', 'CDM', 'BOX-TO-BOX']:
    role_players = midfielders[midfielders['midfielder_role'] == role]
    if len(role_players) > 0:
        print(f"\n🎯 TOP 10 {role}s:")
        print("-" * 80)
        
        sort_col = {
            'CAM': 'cam_score',
            'CM': 'cm_score',
            'CDM': 'cdm_score',
            'BOX-TO-BOX': 'b2b_score'
        }[role]
        
        top_role = role_players.nlargest(10, sort_col)[
            ['player', 'squad', 'comp', sort_col, 'tackles', 'key_passes', 'progressive_passes']
        ]
        print(top_role.to_string(index=False))

# ============================================================================
# SAVE ALL CLASSIFICATIONS
# ============================================================================
print("\n" + "="*80)
print("💾 SAVING UPDATED CLASSIFICATIONS")
print("="*80)

# Save strikers
if len(strikers) > 0:
    strikers_output = strikers[[
        'player', 'squad', 'comp', 'age', 'striker_role',
        'poacher_score', 'false9_score', 'goals_per_90', 'key_passes', 'assists_per_90'
    ]].copy()
    strikers_path = '/mnt/user-data/outputs/STRIKERS_CLASSIFICATION.csv'
    strikers_output.to_csv(strikers_path, index=False)
    print(f"✓ Saved strikers: {strikers_path}")

# Save wingers
if len(wingers) > 0:
    wingers_output = wingers[[
        'player', 'squad', 'comp', 'age', 'winger_role', 'winger_score',
        'goals_per_90', 'assists_per_90', 'crosses_into_penalty_area',
        'progressive_carries', 'key_passes'
    ]].copy()
    wingers_path = '/mnt/user-data/outputs/WINGERS_CLASSIFICATION.csv'
    wingers_output.to_csv(wingers_path, index=False)
    print(f"✓ Saved wingers: {wingers_path}")

# Save midfielders
midfielders_output = midfielders[[
    'player', 'squad', 'comp', 'age', 'midfielder_role',
    'cam_score', 'cm_score', 'cdm_score', 'b2b_score',
    'key_passes', 'tackles', 'progressive_passes'
]].copy()
midfielders_path = '/mnt/user-data/outputs/MIDFIELDERS_UPDATED_CLASSIFICATION.csv'
midfielders_output.to_csv(midfielders_path, index=False)
print(f"✓ Saved midfielders: {midfielders_path}")

# Load defenders (already done)
defenders_df = pd.read_csv('/mnt/user-data/outputs/DEFENDERS_FINAL_CLASSIFICATION.csv')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 COMPLETE UPDATED CLASSIFICATION SUMMARY")
print("="*80)

print(f"\n⚽ STRIKERS (Central Forwards):")
if len(strikers) > 0:
    print(f"   Poachers: {len(strikers[strikers['striker_role'] == 'POACHER'])}")
    print(f"   False 9s: {len(strikers[strikers['striker_role'] == 'FALSE 9'])}")
    print(f"   Total: {len(strikers)}")

print(f"\n⚡ WINGERS (Wide Attackers):")
if len(wingers) > 0:
    print(f"   RW/LW: {len(wingers)}")
    print(f"   (Cannot distinguish Left vs Right without spatial data)")

print(f"\n⚙️ MIDFIELDERS:")
print(f"   CAMs: {len(midfielders[midfielders['midfielder_role'] == 'CAM'])}")
print(f"   CMs: {len(midfielders[midfielders['midfielder_role'] == 'CM'])}")
print(f"   CDMs: {len(midfielders[midfielders['midfielder_role'] == 'CDM'])}")
print(f"   Box-to-Box: {len(midfielders[midfielders['midfielder_role'] == 'BOX-TO-BOX'])}")
print(f"   Total: {len(midfielders)}")

print(f"\n🛡️ DEFENDERS:")
print(f"   Center Backs (CB): {len(defenders_df[defenders_df['defender_position'] == 'CB'])}")
print(f"   Fullbacks (RB/LB): {len(defenders_df[defenders_df['defender_position'] == 'FB'])}")
print(f"   Wing Backs (RWB/LWB): {len(defenders_df[defenders_df['defender_position'] == 'WB'])}")
print(f"   Total: {len(defenders_df)}")

total_classified = len(strikers) + len(wingers) + len(midfielders) + len(defenders_df)
print(f"\n🎯 TOTAL PLAYERS CLASSIFIED: {total_classified}")

print("\n" + "="*80)
print("✅ ALL UPDATED CLASSIFICATIONS SAVED!")
print("="*80)

print(f"\n📁 Output files:")
print(f"   • STRIKERS_CLASSIFICATION.csv (Poachers & False 9s)")
print(f"   • WINGERS_CLASSIFICATION.csv (RW/LW)")
print(f"   • MIDFIELDERS_UPDATED_CLASSIFICATION.csv (CAM, CM, CDM, B2B)")
print(f"   • DEFENDERS_FINAL_CLASSIFICATION.csv (CB, FB, WB)")
print("="*80)
