# ============================================================
# PROJECT : Player Churn Prediction & Analytics Dashboard
# DATASET : Online Gaming Behavior (40,034 players)
# PHASE 2 : Exploratory Data Analysis (EDA)
# TOOLS   : Python — pandas, seaborn, matplotlib
# AUTHOR  : Isfaq
# ============================================================
#
# BUSINESS QUESTION:
# Which player behaviors and attributes most strongly
# differentiate churned players from active players?
#
# CHARTS PRODUCED (saved to /visuals/):
#   chart1_kpi_overview.png
#   chart2_churn_by_genre.png
#   chart3_session_duration_dist.png
#   chart4_boxplots.png
#   chart5_correlation_heatmap.png
#   chart6_churn_by_engagement.png
# ============================================================


# ─────────────────────────────────────────────
# STEP 0 — IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                   # non-interactive backend for saving
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('visuals', exist_ok=True)   # folder for all output charts


# ─────────────────────────────────────────────
# STEP 1 — LOAD CLEAN MASTER DATASET
# (produced by Phase 1 SQL pipeline)
# ─────────────────────────────────────────────
df = pd.read_csv('gaming_master_clean.csv')

print("=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)
print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Churned (1)    : {df['churned'].sum():,}  ({df['churned'].mean()*100:.2f}%)")
print(f"Active  (0)    : {(df['churned']==0).sum():,}  ({(1-df['churned'].mean())*100:.2f}%)")
print()
print("Column types:")
print(df.dtypes)
print()
print("Missing values:")
print(df.isnull().sum())


# ─────────────────────────────────────────────
# STEP 2 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("DESCRIPTIVE STATS BY CHURN STATUS")
print("=" * 55)

# Separate active and churned for comparisons
active  = df[df['churned'] == 0]
churned = df[df['churned'] == 1]

# Numeric columns to compare
num_cols = [
    'PlayTimeHours',
    'SessionsPerWeek',
    'AvgSessionDurationMinutes',
    'InGamePurchases',
    'PlayerLevel',
    'AchievementsUnlocked',
    'HrsPerSession'
]

# Group-level means
stats_by_churn = df.groupby('churned')[num_cols].mean().round(2)
stats_by_churn.index = ['Active', 'Churned']
print(stats_by_churn.T.to_string())

# Churn rate by categorical columns
print("\n--- Churn Rate by Game Genre ---")
genre_churn = (df.groupby('GameGenre')['churned']
               .mean()
               .mul(100)
               .round(2)
               .sort_values(ascending=False))
print(genre_churn.to_string())

print("\n--- Churn Rate by Location ---")
location_churn = (df.groupby('Location')['churned']
                  .mean()
                  .mul(100)
                  .round(2)
                  .sort_values(ascending=False))
print(location_churn.to_string())

print("\n--- Churn Rate by Game Difficulty ---")
diff_churn = (df.groupby('GameDifficulty')['churned']
              .mean()
              .mul(100)
              .round(2)
              .sort_values(ascending=False))
print(diff_churn.to_string())


# ─────────────────────────────────────────────
# GLOBAL CHART STYLE SETTINGS
# ─────────────────────────────────────────────
ACTIVE_COLOR = '#2ECC71'    # green  → active players
CHURN_COLOR  = '#E74C3C'    # red    → churned players
BG_COLOR     = '#0F1117'    # dark background
CARD_COLOR   = '#1A1D27'    # chart panel background
TEXT_COLOR   = '#EAEAEA'    # primary text
MUTED_COLOR  = '#8A8FA8'    # secondary text / ticks
GRID_COLOR   = '#2A2D3A'    # gridlines
ACCENT       = '#F39C12'    # highlight / annotation

plt.rcParams.update({
    'figure.facecolor' : BG_COLOR,
    'axes.facecolor'   : CARD_COLOR,
    'axes.edgecolor'   : GRID_COLOR,
    'axes.labelcolor'  : TEXT_COLOR,
    'axes.titlecolor'  : TEXT_COLOR,
    'xtick.color'      : MUTED_COLOR,
    'ytick.color'      : MUTED_COLOR,
    'text.color'       : TEXT_COLOR,
    'grid.color'       : GRID_COLOR,
    'grid.alpha'       : 0.4,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ─────────────────────────────────────────────
# CHART 1 — KPI OVERVIEW DASHBOARD
# Purpose: Instant high-level summary for stakeholders
# ─────────────────────────────────────────────
print("\n[Generating Chart 1: KPI Overview...]")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('Player Churn — KPI Overview',
             fontsize=16, fontweight='bold', color=TEXT_COLOR, y=1.02)

kpis = [
    ('Total Players',   f'{len(df):,}',                    '#5B8CFF'),
    ('Churn Rate',      f'{df["churned"].mean()*100:.1f}%', CHURN_COLOR),
    ('Active Players',  f'{(df["churned"]==0).sum():,}',    ACTIVE_COLOR),
    ('Churned Players', f'{df["churned"].sum():,}',         ACCENT),
]

for ax, (label, val, color) in zip(axes, kpis):
    ax.set_facecolor(CARD_COLOR)
    ax.text(0.5, 0.62, val, ha='center', va='center',
            fontsize=30, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.28, label, ha='center', va='center',
            fontsize=11, color=MUTED_COLOR, transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('visuals/chart1_kpi_overview.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart1_kpi_overview.png")


# ─────────────────────────────────────────────
# CHART 2 — CHURN RATE BY GAME GENRE
# Business Q: Which game genre retains players best?
# ─────────────────────────────────────────────
print("\n[Generating Chart 2: Churn Rate by Game Genre...]")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)

genre_df = (df.groupby('GameGenre')['churned']
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index())
genre_df.columns = ['GameGenre', 'ChurnRate']

# Highest churn bar highlighted in red, rest in blue
bar_colors = [CHURN_COLOR if i == 0 else '#5B8CFF'
              for i in range(len(genre_df))]

bars = ax.barh(genre_df['GameGenre'], genre_df['ChurnRate'],
               color=bar_colors, height=0.55, edgecolor='none')

# Value labels on each bar
for bar, val in zip(bars, genre_df['ChurnRate']):
    ax.text(val + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            va='center', color=TEXT_COLOR,
            fontsize=11, fontweight='bold')

# Overall average reference line
overall_avg = df['churned'].mean() * 100
ax.axvline(overall_avg, color=ACCENT, linestyle='--',
           linewidth=1.5, alpha=0.8,
           label=f'Overall avg: {overall_avg:.1f}%')

ax.set_xlabel('Churn Rate (%)', fontsize=11)
ax.set_title('Churn Rate by Game Genre', fontsize=14,
             fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0)
ax.set_xlim(0, 32)
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor(CARD_COLOR)

plt.tight_layout()
plt.savefig('visuals/chart2_churn_by_genre.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart2_churn_by_genre.png")


# ─────────────────────────────────────────────
# CHART 3 — SESSION DURATION DISTRIBUTION
# Business Q: Do churned players have shorter sessions?
# ─────────────────────────────────────────────
print("\n[Generating Chart 3: Session Duration Distribution...]")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)

ax.hist(active['AvgSessionDurationMinutes'],
        bins=40, alpha=0.7, color=ACTIVE_COLOR,
        label=f'Active  (mean = {active["AvgSessionDurationMinutes"].mean():.0f} min)',
        edgecolor='none', density=True)

ax.hist(churned['AvgSessionDurationMinutes'],
        bins=40, alpha=0.7, color=CHURN_COLOR,
        label=f'Churned (mean = {churned["AvgSessionDurationMinutes"].mean():.0f} min)',
        edgecolor='none', density=True)

# Mean reference lines
ax.axvline(active['AvgSessionDurationMinutes'].mean(),
           color=ACTIVE_COLOR, linestyle='--', linewidth=2)
ax.axvline(churned['AvgSessionDurationMinutes'].mean(),
           color=CHURN_COLOR, linestyle='--', linewidth=2)

ax.set_xlabel('Avg Session Duration (minutes)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Session Duration Distribution: Active vs Churned Players',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(CARD_COLOR)

plt.tight_layout()
plt.savefig('visuals/chart3_session_duration_dist.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart3_session_duration_dist.png")


# ─────────────────────────────────────────────
# CHART 4 — BEHAVIORAL METRICS BOXPLOTS
# Business Q: How different are sessions/week and
#             session duration between groups?
# ─────────────────────────────────────────────
print("\n[Generating Chart 4: Behavioral Metrics Boxplots...]")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('Behavioral Metrics — Churned vs Active Players',
             fontsize=14, fontweight='bold', color=TEXT_COLOR)

df_plot = df.copy()
df_plot['Status'] = df_plot['churned'].map({0: 'Active', 1: 'Churned'})

metrics = [
    ('SessionsPerWeek',            'Sessions Per Week'),
    ('AvgSessionDurationMinutes',  'Avg Session Duration (min)'),
]

for ax, (col, label) in zip(axes, metrics):
    bp = ax.boxplot(
        [df_plot[df_plot['Status'] == 'Active'][col],
         df_plot[df_plot['Status'] == 'Churned'][col]],
        labels=['Active', 'Churned'],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color=MUTED_COLOR),
        capprops=dict(color=MUTED_COLOR),
        flierprops=dict(markerfacecolor=MUTED_COLOR, marker='o',
                        markersize=2, alpha=0.3, linestyle='none')
    )
    bp['boxes'][0].set_facecolor(ACTIVE_COLOR + 'AA')
    bp['boxes'][1].set_facecolor(CHURN_COLOR  + 'AA')
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor(CARD_COLOR)

plt.tight_layout()
plt.savefig('visuals/chart4_boxplots.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart4_boxplots.png")


# ─────────────────────────────────────────────
# CHART 5 — CORRELATION HEATMAP
# Purpose: Understand feature relationships,
#          detect multicollinearity before modeling
# ─────────────────────────────────────────────
print("\n[Generating Chart 5: Correlation Heatmap...]")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG_COLOR)

heatmap_cols = [
    'Age', 'PlayTimeHours', 'InGamePurchases',
    'SessionsPerWeek', 'AvgSessionDurationMinutes',
    'PlayerLevel', 'AchievementsUnlocked',
    'HrsPerSession', 'churned'
]

corr_matrix = df[heatmap_cols].corr()

# Lower triangle mask (removes redundant upper half)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(10, 145, as_cmap=True)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 9, 'color': TEXT_COLOR},
    linewidths=0.5,
    linecolor=BG_COLOR,
    cbar_kws={'shrink': 0.8},
    ax=ax
)

ax.set_title('Feature Correlation Heatmap',
             fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)
ax.set_facecolor(CARD_COLOR)
ax.figure.axes[-1].tick_params(colors=MUTED_COLOR)

plt.tight_layout()
plt.savefig('visuals/chart5_correlation_heatmap.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart5_correlation_heatmap.png")


# ─────────────────────────────────────────────
# CHART 6 — CHURN RATE BY ENGAGEMENT LEVEL
# Business Q: How does engagement level segment players?
# ─────────────────────────────────────────────
print("\n[Generating Chart 6: Churn by Engagement Level...]")

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG_COLOR)

eng_order = ['High', 'Medium', 'Low']
eng_data  = (df.groupby('EngagementLevel')['churned']
             .value_counts(normalize=True)
             .mul(100)
             .unstack()
             .reindex(eng_order))

# Stacked bar: active on bottom, churned on top
bars_active  = ax.bar(eng_order, eng_data[0],
                      color=ACTIVE_COLOR, label='Active', width=0.5)
bars_churned = ax.bar(eng_order, eng_data[1], bottom=eng_data[0],
                      color=CHURN_COLOR, label='Churned', width=0.5)

# Churn % label inside each churned bar
for bar, val in zip(bars_churned, eng_data[1]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}%',
            ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')

ax.set_ylabel('Player Share (%)', fontsize=11)
ax.set_xlabel('Engagement Level', fontsize=11)
ax.set_title('Churn Rate by Engagement Level',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(CARD_COLOR)

plt.tight_layout()
plt.savefig('visuals/chart6_churn_by_engagement.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart6_churn_by_engagement.png")


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("PHASE 2 EDA COMPLETE")
print("=" * 55)
print(f"  Total players analysed : {len(df):,}")
print(f"  Churn rate             : {df['churned'].mean()*100:.2f}%")
print(f"  Charts saved           : 6  →  visuals/")
print()
print("KEY FINDINGS:")
print(f"  • Churned players avg {churned['SessionsPerWeek'].mean():.1f} sessions/week")
print(f"    vs {active['SessionsPerWeek'].mean():.1f} for active  (Δ = {active['SessionsPerWeek'].mean()-churned['SessionsPerWeek'].mean():.1f})")
print(f"  • Churned avg session = {churned['AvgSessionDurationMinutes'].mean():.0f} min")
print(f"    vs {active['AvgSessionDurationMinutes'].mean():.0f} min for active  (Δ = {active['AvgSessionDurationMinutes'].mean()-churned['AvgSessionDurationMinutes'].mean():.0f} min)")
print(f"  • RPG has highest churn rate at {genre_df.iloc[0]['ChurnRate']:.1f}%")
print()
print("→ Proceed to phase2_hypothesis_testing.py")
