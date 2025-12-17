"""
COMPREHENSIVE VISUALIZATION SUITE
Creates all charts and graphs for player performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("📊 CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*80)

# Create output directory for charts
import os
os.makedirs('/mnt/user-data/outputs/visualizations', exist_ok=True)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('/mnt/user-data/outputs/big5_leagues_WITH_TRACKING_BACK.csv')
df_2024 = df[df['season'] == 2024].copy()
print(f"✓ Loaded {len(df_2024)} players from 2024 season")

# Load classifications
strikers = pd.read_csv('/mnt/user-data/outputs/STRIKERS_CLASSIFICATION.csv')
wingers = pd.read_csv('/mnt/user-data/outputs/WINGERS_CLASSIFICATION.csv')
midfielders = pd.read_csv('/mnt/user-data/outputs/MIDFIELDERS_UPDATED_CLASSIFICATION.csv')
defenders = pd.read_csv('/mnt/user-data/outputs/DEFENDERS_ENHANCED_WITH_LB_RB.csv')

# Load value analysis
value_analysis = pd.read_csv('/mnt/user-data/outputs/VALUE_VS_PERFORMANCE_ANALYSIS.csv')

# ============================================================================
# CHART 1: MODEL PERFORMANCE COMPARISON
# ============================================================================
print("\n📊 Chart 1: Model Performance Comparison...")

fig, ax = plt.subplots(figsize=(12, 7))

models = ['Forwards', 'Midfielders', 'Defenders']
rf_scores = [0.9815, 0.8547, 0.9906]
xgb_scores = [0.9876, 0.9019, 0.9930]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='#e74c3c', alpha=0.8)

ax.set_ylabel('R² Score (Accuracy)', fontsize=12, fontweight='bold')
ax.set_xlabel('Position', fontsize=12, fontweight='bold')
ax.set_title('ML Model Performance Comparison\nXGBoost vs Random Forest', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim([0.8, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/01_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_model_comparison.png")

# ============================================================================
# CHART 2: PLAYER ROLE DISTRIBUTION
# ============================================================================
print("\n📊 Chart 2: Player Role Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Strikers
striker_counts = strikers['striker_role'].value_counts()
axes[0, 0].pie(striker_counts.values, labels=striker_counts.index, autopct='%1.1f%%',
               startangle=90, colors=['#3498db', '#e74c3c'], textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0, 0].set_title('Striker Roles\n(196 players)', fontsize=14, fontweight='bold', pad=15)

# Midfielders
mid_counts = midfielders['midfielder_role'].value_counts()
axes[0, 1].pie(mid_counts.values, labels=mid_counts.index, autopct='%1.1f%%',
               startangle=90, colors=['#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
               textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[0, 1].set_title('Midfielder Roles\n(462 players)', fontsize=14, fontweight='bold', pad=15)

# Defenders
def_counts = defenders['detailed_position'].value_counts()
axes[1, 0].pie(def_counts.values, labels=def_counts.index, autopct='%1.1f%%',
               startangle=90, colors=['#34495e', '#95a5a6', '#7f8c8d', '#e67e22', '#16a085', '#c0392b'],
               textprops={'fontsize': 9, 'fontweight': 'bold'})
axes[1, 0].set_title('Defender Positions\n(554 players)', fontsize=14, fontweight='bold', pad=15)

# Overall distribution
all_positions = {
    'Strikers': len(strikers),
    'Wingers': len(wingers),
    'Midfielders': len(midfielders),
    'Defenders': len(defenders)
}
axes[1, 1].bar(all_positions.keys(), all_positions.values(), color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], alpha=0.8)
axes[1, 1].set_title('Total Players by Position\n(1,336 classified)', fontsize=14, fontweight='bold', pad=15)
axes[1, 1].set_ylabel('Number of Players', fontsize=11, fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels
for i, (k, v) in enumerate(all_positions.items()):
    axes[1, 1].text(i, v, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('Player Classification Overview - 2024 Season', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/02_role_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_role_distribution.png")

# ============================================================================
# CHART 3: TOP 10 PLAYERS BY POSITION - COMPREHENSIVE VIEW
# ============================================================================
print("\n📊 Chart 3: Top 10 Players by Position...")

# Create 9 separate charts for all positions (including LB/RB)
fig = plt.figure(figsize=(20, 36))

# Top 10 Poachers (Strikers)
ax1 = plt.subplot(9, 1, 1)
top_poachers = strikers[strikers['striker_role'] == 'POACHER'].nlargest(10, 'poacher_score')
y_pos = np.arange(len(top_poachers))
ax1.barh(y_pos, top_poachers['poacher_score'].values, color='#e74c3c', alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_poachers.iterrows()], fontsize=10)
ax1.set_xlabel('Poacher Score', fontsize=11, fontweight='bold')
ax1.set_title('⚽ TOP 10 POACHERS (Goalscoring Strikers)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_poachers['poacher_score'].values):
    ax1.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 False 9s
ax2 = plt.subplot(9, 1, 2)
top_false9 = strikers[strikers['striker_role'] == 'FALSE 9'].nlargest(10, 'false9_score')
y_pos = np.arange(len(top_false9))
ax2.barh(y_pos, top_false9['false9_score'].values, color='#9b59b6', alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_false9.iterrows()], fontsize=10)
ax2.set_xlabel('False 9 Score', fontsize=11, fontweight='bold')
ax2.set_title('🎨 TOP 10 FALSE 9s (Creative Strikers)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_false9['false9_score'].values):
    ax2.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 Wingers
ax3 = plt.subplot(9, 1, 3)
top_wingers = wingers.nlargest(10, 'winger_score')
y_pos = np.arange(len(top_wingers))
ax3.barh(y_pos, top_wingers['winger_score'].values, color='#f39c12', alpha=0.8)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_wingers.iterrows()], fontsize=10)
ax3.set_xlabel('Winger Score', fontsize=11, fontweight='bold')
ax3.set_title('⚡ TOP 10 WINGERS (RW/LW)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_wingers['winger_score'].values):
    ax3.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 CAMs
ax4 = plt.subplot(9, 1, 4)
top_cams = midfielders.nlargest(10, 'cam_score')
y_pos = np.arange(len(top_cams))
ax4.barh(y_pos, top_cams['cam_score'].values, color='#3498db', alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_cams.iterrows()], fontsize=10)
ax4.set_xlabel('CAM Score', fontsize=11, fontweight='bold')
ax4.set_title('🎨 TOP 10 ATTACKING MIDFIELDERS (CAM)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_cams['cam_score'].values):
    ax4.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 CDMs
ax5 = plt.subplot(9, 1, 5)
top_cdms = midfielders.nlargest(10, 'cdm_score')
y_pos = np.arange(len(top_cdms))
ax5.barh(y_pos, top_cdms['cdm_score'].values, color='#1abc9c', alpha=0.8)
ax5.set_yticks(y_pos)
ax5.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_cdms.iterrows()], fontsize=10)
ax5.set_xlabel('CDM Score', fontsize=11, fontweight='bold')
ax5.set_title('🛡️ TOP 10 DEFENSIVE MIDFIELDERS (CDM)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_cdms['cdm_score'].values):
    ax5.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 Center Backs
ax6 = plt.subplot(9, 1, 6)
top_cbs = defenders[defenders['detailed_position'] == 'CB'].nlargest(10, 'cb_score')
y_pos = np.arange(len(top_cbs))
ax6.barh(y_pos, top_cbs['cb_score'].values, color='#2ecc71', alpha=0.8)
ax6.set_yticks(y_pos)
ax6.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_cbs.iterrows()], fontsize=10)
ax6.set_xlabel('Center Back Score', fontsize=11, fontweight='bold')
ax6.set_title('🏰 TOP 10 CENTER BACKS (CB)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_cbs['cb_score'].values):
    ax6.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 Left Backs
ax7 = plt.subplot(9, 1, 7)
top_lbs = defenders[defenders['detailed_position'] == 'LFB'].nlargest(10, 'lb_score')
y_pos = np.arange(len(top_lbs))
ax7.barh(y_pos, top_lbs['lb_score'].values, color='#e67e22', alpha=0.8)
ax7.set_yticks(y_pos)
ax7.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_lbs.iterrows()], fontsize=10)
ax7.set_xlabel('Left Back Score', fontsize=11, fontweight='bold')
ax7.set_title('⬅️ TOP 10 LEFT BACKS (LB)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax7.invert_yaxis()
ax7.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_lbs['lb_score'].values):
    ax7.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 Right Backs
ax8 = plt.subplot(9, 1, 8)
top_rbs = defenders[defenders['detailed_position'] == 'RFB'].nlargest(10, 'rb_score')
y_pos = np.arange(len(top_rbs))
ax8.barh(y_pos, top_rbs['rb_score'].values, color='#16a085', alpha=0.8)
ax8.set_yticks(y_pos)
ax8.set_yticklabels([f"{row['player']} ({row['squad']})" for _, row in top_rbs.iterrows()], fontsize=10)
ax8.set_xlabel('Right Back Score', fontsize=11, fontweight='bold')
ax8.set_title('➡️ TOP 10 RIGHT BACKS (RB)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax8.invert_yaxis()
ax8.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_rbs['rb_score'].values):
    ax8.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

# Top 10 Wing Backs (Combined LWB/RWB)
ax9 = plt.subplot(9, 1, 9)
top_wbs = defenders[defenders['detailed_position'].isin(['LWB', 'RWB'])].nlargest(10, 'wb_score')
y_pos = np.arange(len(top_wbs))
colors_wb = ['#c0392b' if pos == 'LWB' else '#27ae60' for pos in top_wbs['detailed_position']]
ax9.barh(y_pos, top_wbs['wb_score'].values, color=colors_wb, alpha=0.8)
ax9.set_yticks(y_pos)
ax9.set_yticklabels([f"{row['player']} ({row['squad']}) [{row['detailed_position']}]" for _, row in top_wbs.iterrows()], fontsize=10)
ax9.set_xlabel('Wing Back Score', fontsize=11, fontweight='bold')
ax9.set_title('🔥 TOP 10 WING BACKS (LWB/RWB)', fontsize=14, fontweight='bold', pad=15, loc='left')
ax9.invert_yaxis()
ax9.grid(axis='x', alpha=0.3)
for i, v in enumerate(top_wbs['wb_score'].values):
    ax9.text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold', fontsize=9)

plt.suptitle('🏆 TOP 10 PERFORMERS BY POSITION - 2024 SEASON', fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/03_top_players.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_top_players.png")

# ============================================================================
# CHART 4: VALUE VS PERFORMANCE SCATTER PLOT
# ============================================================================
print("\n📊 Chart 4: Value vs Performance Analysis...")

fig, ax = plt.subplots(figsize=(14, 10))

# Filter recent season
recent_value = value_analysis[value_analysis['season'].isin([2023, 2024])]

# Create scatter plot by position
positions = recent_value['general_position'].unique()
colors = {'Forward': '#e74c3c', 'Midfielder': '#3498db', 'Defender': '#2ecc71'}

for pos in positions:
    if pos in colors:
        pos_data = recent_value[recent_value['general_position'] == pos]
        ax.scatter(pos_data['value_score'], pos_data['performance_score'],
                  alpha=0.6, s=100, c=colors[pos], label=pos, edgecolors='black', linewidth=0.5)

# Add diagonal line (fair value)
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, linewidth=2, label='Fair Value Line')

# Add quadrant labels
ax.text(75, 25, 'OVERPRICED\n(High Value, Low Performance)', 
        ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.2))
ax.text(25, 75, 'BARGAIN!\n(Low Value, High Performance)', 
        ha='center', va='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.2))

ax.set_xlabel('Market Value Score (0-100)', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Score (0-100)', fontsize=12, fontweight='bold')
ax.set_title('Value vs Performance Analysis\nIdentifying Undervalued & Overpriced Players', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/04_value_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_value_vs_performance.png")

# ============================================================================
# CHART 5: LEAGUE VALUE ANALYSIS
# ============================================================================
print("\n📊 Chart 5: League Value Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calculate league averages
league_avg = value_analysis.groupby('comp').agg({
    'value_gap': 'mean',
    'performance_score': 'mean',
    'market_value_eur_mill': 'mean'
}).round(2)

league_avg = league_avg.sort_values('value_gap', ascending=False)

# Chart 1: Average Value Gap by League
leagues = league_avg.index
colors_league = ['#2ecc71' if x > 0 else '#e74c3c' for x in league_avg['value_gap'].values]

axes[0].bar(range(len(leagues)), league_avg['value_gap'].values, color=colors_league, alpha=0.8)
axes[0].set_xticks(range(len(leagues)))
axes[0].set_xticklabels(leagues, rotation=45, ha='right', fontsize=11)
axes[0].set_ylabel('Average Value Gap', fontsize=12, fontweight='bold')
axes[0].set_title('Best Value Leagues\n(Positive = Undervalued)', fontsize=14, fontweight='bold', pad=15)
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels
for i, v in enumerate(league_avg['value_gap'].values):
    axes[0].text(i, v, f'{v:+.1f}', ha='center', va='bottom' if v > 0 else 'top', 
                fontweight='bold', fontsize=10)

# Chart 2: Players per League
league_counts = value_analysis.groupby('comp')['player'].nunique().sort_values(ascending=False)
axes[1].barh(range(len(league_counts)), league_counts.values, 
             color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.8)
axes[1].set_yticks(range(len(league_counts)))
axes[1].set_yticklabels(league_counts.index, fontsize=11)
axes[1].set_xlabel('Number of Players Analyzed', fontsize=12, fontweight='bold')
axes[1].set_title('Players Analyzed per League', fontsize=14, fontweight='bold', pad=15)
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

# Add value labels
for i, v in enumerate(league_counts.values):
    axes[1].text(v, i, f' {v}', ha='left', va='center', fontweight='bold', fontsize=10)

plt.suptitle('League-Level Insights - Transfer Market Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/05_league_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_league_analysis.png")

# ============================================================================
# CHART 6: FEATURE IMPORTANCE (CONCEPTUAL)
# ============================================================================
print("\n📊 Chart 6: Feature Importance...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Forward features
fw_features = ['Shot Efficiency', 'xG per 90', 'Shots per 90', 'Clinical Finisher', 'Non-penalty xG']
fw_importance = [56, 18, 12, 7, 4]
axes[0].barh(fw_features, fw_importance, color='#e74c3c', alpha=0.8)
axes[0].set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Forward Model\nTop Features', fontsize=13, fontweight='bold', pad=10)
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(fw_importance):
    axes[0].text(v, i, f' {v}%', ha='left', va='center', fontweight='bold')

# Midfielder features
mf_features = ['Final Third\nInvolvement', 'Progressive\nCarries', 'Key Passes', 'Box-to-Box\nScore', 'Passes into\nFinal Third']
mf_importance = [71, 15, 2, 1, 1]
axes[1].barh(mf_features, mf_importance, color='#3498db', alpha=0.8)
axes[1].set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Midfielder Model\nTop Features', fontsize=13, fontweight='bold', pad=10)
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(mf_importance):
    axes[1].text(v, i, f' {v}%', ha='left', va='center', fontweight='bold')

# Defender features
df_features = ['Defensive Work\nRate per 90', 'Def Third\nDominance', 'Interceptions', 'Att Third\nTackles', 'Attacking FB\nIndex']
df_importance = [98, 1, 0.15, 0.15, 0.11]
axes[2].barh(df_features, df_importance, color='#2ecc71', alpha=0.8)
axes[2].set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
axes[2].set_title('Defender Model\nTop Features', fontsize=13, fontweight='bold', pad=10)
axes[2].grid(axis='x', alpha=0.3)
for i, v in enumerate(df_importance):
    axes[2].text(v, i, f' {v}%', ha='left', va='center', fontweight='bold', fontsize=9)

plt.suptitle('XGBoost Feature Importance - Position-Specific Models', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/06_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_feature_importance.png")

# ============================================================================
# CHART 7: TRACKING BACK IMPACT (BEFORE & AFTER)
# ============================================================================
print("\n📊 Chart 7: Tracking Back Impact...")

fig, ax = plt.subplots(figsize=(10, 7))

metrics = ['Training R²', 'Test R²', 'Cross-Val R²']
before = [0.9332, 0.6709, 0.6710]
after = [0.9981, 0.9906, 0.9912]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, before, width, label='Before (Without Tracking Back)', 
               color='#e74c3c', alpha=0.7)
bars2 = ax.bar(x + width/2, after, width, label='After (With Tracking Back)', 
               color='#2ecc71', alpha=0.7)

ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Impact of Tracking Back Metrics on Defender Model\n+32.2% Accuracy Improvement!', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0.6, 1.0])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add improvement annotation
ax.annotate('', xy=(1, after[1]), xytext=(1, before[1]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(1.3, (before[1] + after[1])/2, '+32.2%\nimprovement!',
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/visualizations/07_tracking_back_impact.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_tracking_back_impact.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("🎉 ALL VISUALIZATIONS CREATED!")
print("="*80)

print("\n📊 Created 7 comprehensive charts:")
print("   1. Model Performance Comparison (RF vs XGBoost)")
print("   2. Player Role Distribution (Pie charts)")
print("   3. Top Players by Position (Bar charts)")
print("   4. Value vs Performance Scatter Plot")
print("   5. League Value Analysis")
print("   6. Feature Importance by Position")
print("   7. Tracking Back Impact (Before & After)")

print(f"\n📁 All charts saved to:")
print(f"   /mnt/user-data/outputs/visualizations/")

print("\n💡 These visualizations are perfect for:")
print("   • GitHub README")
print("   • Portfolio presentations")
print("   • LinkedIn posts")
print("   • Interview discussions")
print("   • Project reports")

print("\n" + "="*80)
print("✅ VISUALIZATION SUITE COMPLETE!")
print("="*80)
