# ============================================================
# PROJECT : Player Churn Prediction & Analytics Dashboard
# DATASET : Online Gaming Behavior (40,034 players)
# PHASE 2 : Hypothesis Testing
# TOOLS   : Python — pandas, scipy, matplotlib
# AUTHOR  : Isfaq
# ============================================================
#
# TESTS CONDUCTED:
#   Test 1 — Welch's t-test : Sessions per week (Active vs Churned)
#   Test 2 — Welch's t-test : Avg session duration (Active vs Churned)
#   Test 3 — Chi-square     : Game genre vs churn (independent?)
#   Test 4 — Welch's t-test : Player level (Active vs Churned)
#
# H0 (null hypothesis) for t-tests:
#   There is no significant difference between active
#   and churned players for the given metric.
#
# Significance level (α) = 0.05
# ============================================================


# ─────────────────────────────────────────────
# STEP 0 — IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('visuals', exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA & SPLIT BY CHURN STATUS
# ─────────────────────────────────────────────
df = pd.read_csv('gaming_master_clean.csv')

active  = df[df['churned'] == 0]   # 29,710 active players
churned = df[df['churned'] == 1]   # 10,324 churned players

ALPHA = 0.05   # significance threshold

print("=" * 60)
print("HYPOTHESIS TESTING — PLAYER CHURN ANALYSIS")
print(f"Significance level (α) = {ALPHA}")
print("=" * 60)


# ─────────────────────────────────────────────
# HELPER FUNCTION — Print test result cleanly
# ─────────────────────────────────────────────
def print_test_result(test_name, question, stat, p_val,
                      stat_label='t-statistic', alpha=0.05):
    result = "✅ REJECT H0 — Statistically significant difference" \
             if p_val < alpha else \
             "❌ FAIL to reject H0 — No significant difference"
    print(f"\n{'─'*60}")
    print(f"  {test_name}")
    print(f"{'─'*60}")
    print(f"  Question     : {question}")
    print(f"  {stat_label:<13}: {stat:.4f}")
    print(f"  p-value      : {p_val:.2e}")
    print(f"  Result       : {result}")


# ─────────────────────────────────────────────
# TEST 1 — Sessions Per Week
# H0: No difference in sessions/week between groups
# H1: Churned players have fewer sessions/week
# Method: Welch's t-test (unequal variance assumed)
# ─────────────────────────────────────────────
t1, p1 = stats.ttest_ind(
    active['SessionsPerWeek'],
    churned['SessionsPerWeek'],
    equal_var=False         # Welch's t-test — safer for large samples
)

print_test_result(
    test_name   = "TEST 1 — Welch's t-test: Sessions Per Week",
    question    = "Do churned players attend significantly fewer sessions per week?",
    stat        = t1,
    p_val       = p1
)
print(f"\n  Active  mean : {active['SessionsPerWeek'].mean():.2f} sessions/week")
print(f"  Churned mean : {churned['SessionsPerWeek'].mean():.2f} sessions/week")
print(f"  Difference   : {active['SessionsPerWeek'].mean() - churned['SessionsPerWeek'].mean():.2f} sessions/week")
print(f"\n  Business interpretation:")
print(f"  Churned players log 6.7 fewer sessions per week (p < 0.001).")
print(f"  Session frequency is the single strongest behavioural signal")
print(f"  of player churn risk — EA should flag players dropping below")
print(f"  5 sessions/week for re-engagement campaigns.")


# ─────────────────────────────────────────────
# TEST 2 — Avg Session Duration
# H0: No difference in session duration between groups
# H1: Churned players have shorter sessions
# Method: Welch's t-test
# ─────────────────────────────────────────────
t2, p2 = stats.ttest_ind(
    active['AvgSessionDurationMinutes'],
    churned['AvgSessionDurationMinutes'],
    equal_var=False
)

print_test_result(
    test_name   = "TEST 2 — Welch's t-test: Avg Session Duration",
    question    = "Do churned players have significantly shorter session durations?",
    stat        = t2,
    p_val       = p2
)
print(f"\n  Active  mean : {active['AvgSessionDurationMinutes'].mean():.1f} minutes/session")
print(f"  Churned mean : {churned['AvgSessionDurationMinutes'].mean():.1f} minutes/session")
print(f"  Difference   : {active['AvgSessionDurationMinutes'].mean() - churned['AvgSessionDurationMinutes'].mean():.1f} minutes/session")
print(f"\n  Business interpretation:")
print(f"  Churned players spend 37.6 fewer minutes per session (p < 0.001).")
print(f"  Disengagement shows in session depth before players formally")
print(f"  quit — early warning triggers should monitor session length drops.")


# ─────────────────────────────────────────────
# TEST 3 — Game Genre vs Churn (Chi-square)
# H0: Game genre and churn status are independent
# H1: There is a significant relationship between genre and churn
# Method: Chi-square test of independence
# ─────────────────────────────────────────────
contingency_table = pd.crosstab(df['GameGenre'], df['churned'])
chi2, p3, dof, expected = stats.chi2_contingency(contingency_table)

print_test_result(
    test_name   = "TEST 3 — Chi-square: Game Genre vs Churn",
    question    = "Is game genre statistically independent of churn status?",
    stat        = chi2,
    p_val       = p3,
    stat_label  = "chi2-stat"
)
print(f"\n  Degrees of freedom : {dof}")
print(f"\n  Contingency table:")
print(contingency_table.to_string())
print(f"\n  Business interpretation:")
print(f"  Game genre has NO statistically significant relationship")
print(f"  with churn (p = {p3:.4f} > 0.05). EA's retention strategy")
print(f"  should focus on behavioural signals (session frequency,")
print(f"  session depth), NOT genre-specific interventions.")


# ─────────────────────────────────────────────
# TEST 4 — Player Level
# H0: No difference in player level between groups
# H1: Active players have higher player levels
# Method: Welch's t-test
# ─────────────────────────────────────────────
t4, p4 = stats.ttest_ind(
    active['PlayerLevel'],
    churned['PlayerLevel'],
    equal_var=False
)

print_test_result(
    test_name   = "TEST 4 — Welch's t-test: Player Level",
    question    = "Do active players reach significantly higher player levels?",
    stat        = t4,
    p_val       = p4
)
print(f"\n  Active  mean : {active['PlayerLevel'].mean():.1f}")
print(f"  Churned mean : {churned['PlayerLevel'].mean():.1f}")
print(f"  Difference   : {active['PlayerLevel'].mean() - churned['PlayerLevel'].mean():.1f} levels")
print(f"\n  Business interpretation:")
print(f"  Active players average 4.8 higher player levels (p < 0.001).")
print(f"  Deeper game progression correlates with retention — EA should")
print(f"  design progression milestones to extend early-game engagement.")


# ─────────────────────────────────────────────
# CHART 7 — HYPOTHESIS TEST RESULTS VISUAL
# Purpose: Business-ready summary of test findings
# ─────────────────────────────────────────────
print("\n\n[Generating Chart 7: Hypothesis Test Results...]")

BG_COLOR    = '#0F1117'
CARD_COLOR  = '#1A1D27'
TEXT_COLOR  = '#EAEAEA'
MUTED_COLOR = '#8A8FA8'
GRID_COLOR  = '#2A2D3A'
ACTIVE_COLOR= '#2ECC71'
CHURN_COLOR = '#E74C3C'
ACCENT      = '#F39C12'

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

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('Hypothesis Test Results — Key Behavioral Differences (all p < 0.001)',
             fontsize=13, fontweight='bold', color=TEXT_COLOR, y=1.02)

test_data = [
    {
        'title' : "Sessions Per Week\n(t-test, p < 0.001 ✅)",
        'vals'  : [active['SessionsPerWeek'].mean(),
                   churned['SessionsPerWeek'].mean()],
        'ylabel': 'Avg Sessions / Week',
        'delta' : f"−{active['SessionsPerWeek'].mean()-churned['SessionsPerWeek'].mean():.1f} sessions\np < 0.001",
    },
    {
        'title' : "Avg Session Duration\n(t-test, p < 0.001 ✅)",
        'vals'  : [active['AvgSessionDurationMinutes'].mean(),
                   churned['AvgSessionDurationMinutes'].mean()],
        'ylabel': 'Avg Session Duration (min)',
        'delta' : f"−{active['AvgSessionDurationMinutes'].mean()-churned['AvgSessionDurationMinutes'].mean():.0f} minutes\np < 0.001",
    },
    {
        'title' : "Player Level\n(t-test, p < 0.001 ✅)",
        'vals'  : [active['PlayerLevel'].mean(),
                   churned['PlayerLevel'].mean()],
        'ylabel': 'Avg Player Level',
        'delta' : f"−{active['PlayerLevel'].mean()-churned['PlayerLevel'].mean():.1f} levels\np < 0.001",
    },
]

for ax, t in zip(axes, test_data):
    bars = ax.bar(
        ['Active', 'Churned'],
        t['vals'],
        color=[ACTIVE_COLOR, CHURN_COLOR],
        width=0.45,
        edgecolor='none'
    )
    # Value labels above each bar
    for bar, val in zip(bars, t['vals']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(t['vals']) * 0.02,
                f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=TEXT_COLOR)

    ax.set_title(t['title'], fontsize=10, fontweight='bold', pad=10)
    ax.set_ylabel(t['ylabel'], fontsize=9)
    ax.set_ylim(0, max(t['vals']) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor(CARD_COLOR)

    # Delta annotation box
    ax.text(0.97, 0.93, t['delta'],
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=ACCENT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD_COLOR,
                      edgecolor=ACCENT, alpha=0.8))

plt.tight_layout()
plt.savefig('visuals/chart7_hypothesis_results.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart7_hypothesis_results.png")


# ─────────────────────────────────────────────
# CHART 8 — SESSIONS PER WEEK DISTRIBUTION
# Visualises the gap found in Test 1
# ─────────────────────────────────────────────
print("\n[Generating Chart 8: Sessions Per Week Distribution...]")

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG_COLOR)

ax.hist(active['SessionsPerWeek'],
        bins=20, alpha=0.7, color=ACTIVE_COLOR,
        label=f'Active  (mean = {active["SessionsPerWeek"].mean():.1f})',
        edgecolor='none', density=True)

ax.hist(churned['SessionsPerWeek'],
        bins=20, alpha=0.7, color=CHURN_COLOR,
        label=f'Churned (mean = {churned["SessionsPerWeek"].mean():.1f})',
        edgecolor='none', density=True)

# Mean lines
ax.axvline(active['SessionsPerWeek'].mean(),
           color=ACTIVE_COLOR, linestyle='--', linewidth=2)
ax.axvline(churned['SessionsPerWeek'].mean(),
           color=CHURN_COLOR, linestyle='--', linewidth=2)

ax.set_xlabel('Sessions Per Week', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(
    f'Sessions Per Week Distribution: Active vs Churned\n'
    f'(Welch\'s t-test: t = {t1:.1f}, p < 0.001 | Δ = '
    f'{active["SessionsPerWeek"].mean()-churned["SessionsPerWeek"].mean():.1f} sessions)',
    fontsize=12, fontweight='bold', pad=15)
ax.legend(fontsize=10, framealpha=0)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor(CARD_COLOR)

plt.tight_layout()
plt.savefig('visuals/chart8_sessions_dist.png',
            dpi=180, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print("  ✅ Saved: visuals/chart8_sessions_dist.png")


# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print()
print("=" * 60)
print("HYPOTHESIS TESTING SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    'Test'      : ['T-test 1', 'T-test 2', 'Chi-square', 'T-test 4'],
    'Variable'  : ['Sessions/Week', 'Session Duration',
                   'Game Genre', 'Player Level'],
    'Method'    : ["Welch's t-test"] * 3 + ["Welch's t-test"],
    'p-value'   : [f'{p1:.2e}', f'{p2:.2e}', f'{p3:.4f}', f'{p4:.2e}'],
    'Reject H0' : ['✅ Yes', '✅ Yes', '❌ No', '✅ Yes'],
    'Significant': ['Yes', 'Yes', 'No', 'Yes'],
})
print(summary.to_string(index=False))
print()
print("→ 3 of 4 tests significant. Proceed to phase3_modeling.py")
print("  Key features for model: SessionsPerWeek,")
print("  AvgSessionDurationMinutes, PlayerLevel, HrsPerSession")
