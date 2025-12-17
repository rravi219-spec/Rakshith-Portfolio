"""
XGBOOST MODEL TRAINING - COMPARISON WITH RANDOM FOREST
Training XGBoost for all 3 positions and comparing performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("XGBOOST MODEL TRAINING - ALL POSITIONS")
print("Comparing XGBoost vs Random Forest")
print("="*80)

# Load dataset
print("\n📂 Loading dataset...")
df = pd.read_csv('/mnt/user-data/outputs/big5_leagues_WITH_TRACKING_BACK.csv')
print(f"✓ Loaded {len(df)} records with {len(df.columns)} features")

# Store results for comparison
results = {
    'position': [],
    'model': [],
    'train_r2': [],
    'test_r2': [],
    'test_mae': [],
    'test_rmse': [],
    'cv_mean': [],
    'cv_std': []
}

# ============================================================================
# FORWARD MODEL - XGBOOST
# ============================================================================
print("\n" + "="*80)
print("⚡ PART 1: FORWARD MODEL - XGBOOST")
print("="*80)

forwards = df[df['general_position'] == 'Forward'].copy()
print(f"\nForwards in dataset: {len(forwards)}")

target_fw = 'goals_per_90'

forward_features = [
    'xg_per_90', 'shots_per_90', 'shots_on_target_pct', 'avg_shot_distance',
    'progressive_passes_received', 'key_passes', 'assists_per_90',
    'progressive_carries', 'att_third_tackles', 'non_penalty_xg_per_90',
    'passes_into_penalty_area', 'age', 'min_per_match',
    'poacher_index', 'complete_forward_index', 'pressing_forward_index',
    'clinical_finisher_score', 'shot_efficiency'
]

forward_features = [f for f in forward_features if f in forwards.columns]
print(f"Using {len(forward_features)} features")

# Prepare data
X_fw = forwards[forward_features].fillna(0)
y_fw = forwards[target_fw]

X_fw_train, X_fw_test, y_fw_train, y_fw_test = train_test_split(
    X_fw, y_fw, test_size=0.2, random_state=42
)

print(f"Train set: {len(X_fw_train)} | Test set: {len(X_fw_test)}")

# Scale (XGBoost doesn't require it, but we'll do it for consistency)
scaler_fw = StandardScaler()
X_fw_train_scaled = scaler_fw.fit_transform(X_fw_train)
X_fw_test_scaled = scaler_fw.transform(X_fw_test)

# Train XGBoost
print("\n🚀 Training XGBoost model...")
xgb_fw = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_fw.fit(X_fw_train_scaled, y_fw_train)
print("✓ XGBoost model trained!")

# Predictions
y_fw_pred_train_xgb = xgb_fw.predict(X_fw_train_scaled)
y_fw_pred_test_xgb = xgb_fw.predict(X_fw_test_scaled)

# Metrics
train_r2_fw_xgb = r2_score(y_fw_train, y_fw_pred_train_xgb)
test_r2_fw_xgb = r2_score(y_fw_test, y_fw_pred_test_xgb)
test_mae_fw_xgb = mean_absolute_error(y_fw_test, y_fw_pred_test_xgb)
test_rmse_fw_xgb = np.sqrt(mean_squared_error(y_fw_test, y_fw_pred_test_xgb))

print("\n📊 XGBOOST FORWARD MODEL PERFORMANCE:")
print("-" * 80)
print(f"Training R²: {train_r2_fw_xgb:.4f}")
print(f"Test R²: {test_r2_fw_xgb:.4f}")
print(f"Test MAE: {test_mae_fw_xgb:.4f}")
print(f"Test RMSE: {test_rmse_fw_xgb:.4f}")

cv_scores_fw_xgb = cross_val_score(xgb_fw, X_fw_train_scaled, y_fw_train, 
                                     cv=5, scoring='r2', n_jobs=-1)
print(f"5-Fold CV R²: {cv_scores_fw_xgb.mean():.4f} (+/- {cv_scores_fw_xgb.std():.4f})")

# Store results
results['position'].append('Forward')
results['model'].append('XGBoost')
results['train_r2'].append(train_r2_fw_xgb)
results['test_r2'].append(test_r2_fw_xgb)
results['test_mae'].append(test_mae_fw_xgb)
results['test_rmse'].append(test_rmse_fw_xgb)
results['cv_mean'].append(cv_scores_fw_xgb.mean())
results['cv_std'].append(cv_scores_fw_xgb.std())

# Also train Random Forest for comparison
print("\n🌲 Training Random Forest for comparison...")
rf_fw = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_fw.fit(X_fw_train_scaled, y_fw_train)

y_fw_pred_test_rf = rf_fw.predict(X_fw_test_scaled)
test_r2_fw_rf = r2_score(y_fw_test, y_fw_pred_test_rf)
test_mae_fw_rf = mean_absolute_error(y_fw_test, y_fw_pred_test_rf)
test_rmse_fw_rf = np.sqrt(mean_squared_error(y_fw_test, y_fw_pred_test_rf))

cv_scores_fw_rf = cross_val_score(rf_fw, X_fw_train_scaled, y_fw_train, 
                                    cv=5, scoring='r2', n_jobs=-1)

results['position'].append('Forward')
results['model'].append('Random Forest')
results['train_r2'].append(r2_score(y_fw_train, rf_fw.predict(X_fw_train_scaled)))
results['test_r2'].append(test_r2_fw_rf)
results['test_mae'].append(test_mae_fw_rf)
results['test_rmse'].append(test_rmse_fw_rf)
results['cv_mean'].append(cv_scores_fw_rf.mean())
results['cv_std'].append(cv_scores_fw_rf.std())

print(f"Random Forest Test R²: {test_r2_fw_rf:.4f}")

# Feature importance comparison
feature_importance_fw_xgb = pd.DataFrame({
    'feature': forward_features,
    'xgb_importance': xgb_fw.feature_importances_,
    'rf_importance': rf_fw.feature_importances_
}).sort_values('xgb_importance', ascending=False)

print("\n🔥 TOP 10 FEATURES (XGBOOST):")
print(feature_importance_fw_xgb[['feature', 'xgb_importance']].head(10).to_string(index=False))

# ============================================================================
# MIDFIELDER MODEL - XGBOOST
# ============================================================================
print("\n" + "="*80)
print("⚙️ PART 2: MIDFIELDER MODEL - XGBOOST")
print("="*80)

midfielders = df[df['general_position'] == 'Midfielder'].copy()
print(f"\nMidfielders in dataset: {len(midfielders)}")

target_mf = 'progressive_actions_per_90'

midfielder_features = [
    'progressive_passes', 'passes_into_final_third', 'pass_completion_pct',
    'key_passes', 'xag_per_90', 'tackles', 'interceptions',
    'progressive_carries', 'assists_per_90', 'defensive_actions_per_90',
    'long_pass_completion_pct', 'short_pass_completion_pct', 'age',
    'attacking_midfielder_index', 'defensive_midfielder_index',
    'box_to_box_midfielder_score', 'progressive_midfielder_score',
    'final_third_involvement_per_90', 'creative_output_per_90'
]

midfielder_features = [f for f in midfielder_features if f in midfielders.columns]
print(f"Using {len(midfielder_features)} features")

X_mf = midfielders[midfielder_features].fillna(0)
y_mf = midfielders[target_mf]

X_mf_train, X_mf_test, y_mf_train, y_mf_test = train_test_split(
    X_mf, y_mf, test_size=0.2, random_state=42
)

print(f"Train set: {len(X_mf_train)} | Test set: {len(X_mf_test)}")

scaler_mf = StandardScaler()
X_mf_train_scaled = scaler_mf.fit_transform(X_mf_train)
X_mf_test_scaled = scaler_mf.transform(X_mf_test)

# Train XGBoost
print("\n🚀 Training XGBoost model...")
xgb_mf = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_mf.fit(X_mf_train_scaled, y_mf_train)
print("✓ XGBoost model trained!")

y_mf_pred_test_xgb = xgb_mf.predict(X_mf_test_scaled)
test_r2_mf_xgb = r2_score(y_mf_test, y_mf_pred_test_xgb)
test_mae_mf_xgb = mean_absolute_error(y_mf_test, y_mf_pred_test_xgb)
test_rmse_mf_xgb = np.sqrt(mean_squared_error(y_mf_test, y_mf_pred_test_xgb))

print("\n📊 XGBOOST MIDFIELDER MODEL PERFORMANCE:")
print("-" * 80)
print(f"Test R²: {test_r2_mf_xgb:.4f}")
print(f"Test MAE: {test_mae_mf_xgb:.4f}")
print(f"Test RMSE: {test_rmse_mf_xgb:.4f}")

cv_scores_mf_xgb = cross_val_score(xgb_mf, X_mf_train_scaled, y_mf_train, 
                                     cv=5, scoring='r2', n_jobs=-1)
print(f"5-Fold CV R²: {cv_scores_mf_xgb.mean():.4f} (+/- {cv_scores_mf_xgb.std():.4f})")

results['position'].append('Midfielder')
results['model'].append('XGBoost')
results['train_r2'].append(r2_score(y_mf_train, xgb_mf.predict(X_mf_train_scaled)))
results['test_r2'].append(test_r2_mf_xgb)
results['test_mae'].append(test_mae_mf_xgb)
results['test_rmse'].append(test_rmse_mf_xgb)
results['cv_mean'].append(cv_scores_mf_xgb.mean())
results['cv_std'].append(cv_scores_mf_xgb.std())

# Random Forest comparison
print("\n🌲 Training Random Forest for comparison...")
rf_mf = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_mf.fit(X_mf_train_scaled, y_mf_train)

y_mf_pred_test_rf = rf_mf.predict(X_mf_test_scaled)
test_r2_mf_rf = r2_score(y_mf_test, y_mf_pred_test_rf)

cv_scores_mf_rf = cross_val_score(rf_mf, X_mf_train_scaled, y_mf_train, 
                                    cv=5, scoring='r2', n_jobs=-1)

results['position'].append('Midfielder')
results['model'].append('Random Forest')
results['train_r2'].append(r2_score(y_mf_train, rf_mf.predict(X_mf_train_scaled)))
results['test_r2'].append(test_r2_mf_rf)
results['test_mae'].append(mean_absolute_error(y_mf_test, y_mf_pred_test_rf))
results['test_rmse'].append(np.sqrt(mean_squared_error(y_mf_test, y_mf_pred_test_rf)))
results['cv_mean'].append(cv_scores_mf_rf.mean())
results['cv_std'].append(cv_scores_mf_rf.std())

print(f"Random Forest Test R²: {test_r2_mf_rf:.4f}")

# ============================================================================
# DEFENDER MODEL - XGBOOST
# ============================================================================
print("\n" + "="*80)
print("🛡️ PART 3: DEFENDER MODEL - XGBOOST")
print("="*80)

defenders = df[df['general_position'] == 'Defender'].copy()
print(f"\nDefenders in dataset: {len(defenders)}")

target_df = 'defensive_actions_per_90'

defender_features = [
    'tackles', 'interceptions', 'clearances', 'blocks', 'tackles_won',
    'def_third_tackles', 'mid_third_tackles', 'att_third_tackles',
    'pass_completion_pct', 'progressive_passes', 'errors', 'age', 'min_per_match',
    'pure_defender_index', 'ball_playing_defender_index',
    'attacking_fullback_index', 'tackles_won_rate', 'def_third_dominance',
    'tracking_back_tackles', 'tracking_back_rate', 'tracking_back_index',
    'recovery_ability', 'defensive_work_rate', 'defensive_work_rate_per_90',
    'high_press_defender_score'
]

defender_features = [f for f in defender_features if f in defenders.columns]
print(f"Using {len(defender_features)} features")

X_df = defenders[defender_features].fillna(0)
y_df = defenders[target_df]

X_df_train, X_df_test, y_df_train, y_df_test = train_test_split(
    X_df, y_df, test_size=0.2, random_state=42
)

print(f"Train set: {len(X_df_train)} | Test set: {len(X_df_test)}")

scaler_df = StandardScaler()
X_df_train_scaled = scaler_df.fit_transform(X_df_train)
X_df_test_scaled = scaler_df.transform(X_df_test)

# Train XGBoost
print("\n🚀 Training XGBoost model...")
xgb_df = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_df.fit(X_df_train_scaled, y_df_train)
print("✓ XGBoost model trained!")

y_df_pred_test_xgb = xgb_df.predict(X_df_test_scaled)
test_r2_df_xgb = r2_score(y_df_test, y_df_pred_test_xgb)
test_mae_df_xgb = mean_absolute_error(y_df_test, y_df_pred_test_xgb)
test_rmse_df_xgb = np.sqrt(mean_squared_error(y_df_test, y_df_pred_test_xgb))

print("\n📊 XGBOOST DEFENDER MODEL PERFORMANCE:")
print("-" * 80)
print(f"Test R²: {test_r2_df_xgb:.4f}")
print(f"Test MAE: {test_mae_df_xgb:.4f}")
print(f"Test RMSE: {test_rmse_df_xgb:.4f}")

cv_scores_df_xgb = cross_val_score(xgb_df, X_df_train_scaled, y_df_train, 
                                     cv=5, scoring='r2', n_jobs=-1)
print(f"5-Fold CV R²: {cv_scores_df_xgb.mean():.4f} (+/- {cv_scores_df_xgb.std():.4f})")

results['position'].append('Defender')
results['model'].append('XGBoost')
results['train_r2'].append(r2_score(y_df_train, xgb_df.predict(X_df_train_scaled)))
results['test_r2'].append(test_r2_df_xgb)
results['test_mae'].append(test_mae_df_xgb)
results['test_rmse'].append(test_rmse_df_xgb)
results['cv_mean'].append(cv_scores_df_xgb.mean())
results['cv_std'].append(cv_scores_df_xgb.std())

# Random Forest comparison
print("\n🌲 Training Random Forest for comparison...")
rf_df = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, random_state=42, n_jobs=-1
)
rf_df.fit(X_df_train_scaled, y_df_train)

y_df_pred_test_rf = rf_df.predict(X_df_test_scaled)
test_r2_df_rf = r2_score(y_df_test, y_df_pred_test_rf)

cv_scores_df_rf = cross_val_score(rf_df, X_df_train_scaled, y_df_train, 
                                    cv=5, scoring='r2', n_jobs=-1)

results['position'].append('Defender')
results['model'].append('Random Forest')
results['train_r2'].append(r2_score(y_df_train, rf_df.predict(X_df_train_scaled)))
results['test_r2'].append(test_r2_df_rf)
results['test_mae'].append(mean_absolute_error(y_df_test, y_df_pred_test_rf))
results['test_rmse'].append(np.sqrt(mean_squared_error(y_df_test, y_df_pred_test_rf)))
results['cv_mean'].append(cv_scores_df_rf.mean())
results['cv_std'].append(cv_scores_df_rf.std())

print(f"Random Forest Test R²: {test_r2_df_rf:.4f}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("🏆 FINAL MODEL COMPARISON: XGBOOST VS RANDOM FOREST")
print("="*80)

comparison_df = pd.DataFrame(results)

print("\n📊 COMPLETE RESULTS TABLE:")
print("="*80)
print(comparison_df.to_string(index=False))

print("\n" + "="*80)
print("🎯 HEAD-TO-HEAD COMPARISON (Test R² Scores)")
print("="*80)

for position in ['Forward', 'Midfielder', 'Defender']:
    xgb_score = comparison_df[(comparison_df['position'] == position) & 
                               (comparison_df['model'] == 'XGBoost')]['test_r2'].values[0]
    rf_score = comparison_df[(comparison_df['position'] == position) & 
                              (comparison_df['model'] == 'Random Forest')]['test_r2'].values[0]
    
    winner = "XGBoost" if xgb_score > rf_score else "Random Forest"
    diff = abs(xgb_score - rf_score)
    
    print(f"\n{position.upper()}:")
    print(f"  XGBoost:       {xgb_score:.4f}")
    print(f"  Random Forest: {rf_score:.4f}")
    print(f"  Winner: {winner} (+{diff:.4f})")

print("\n" + "="*80)
print("✅ MODEL COMPARISON COMPLETE!")
print("="*80)

# Determine overall winner
avg_xgb = comparison_df[comparison_df['model'] == 'XGBoost']['test_r2'].mean()
avg_rf = comparison_df[comparison_df['model'] == 'Random Forest']['test_r2'].mean()

print(f"\n🏆 OVERALL AVERAGE TEST R² SCORE:")
print(f"  XGBoost:       {avg_xgb:.4f}")
print(f"  Random Forest: {avg_rf:.4f}")

if avg_xgb > avg_rf:
    print(f"\n🥇 WINNER: XGBOOST! (+{(avg_xgb - avg_rf):.4f})")
else:
    print(f"\n🥇 WINNER: RANDOM FOREST! (+{(avg_rf - avg_xgb):.4f})")

print("="*80)
