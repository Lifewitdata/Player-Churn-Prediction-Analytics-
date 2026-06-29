# Player-Churn-Prediction-Analytics
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Player%20Churn%20Prediction&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=End-to-End%20Data%20Science%20Pipeline%20%7C%20SQL%20%E2%86%92%20Python%20%E2%86%92%20ML&descAlignY=55&descSize=16"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge)

<br/>

> **Can behavioral data alone predict which players will abandon a game — before they do?**
>
> This project answers that question with a full end-to-end data science pipeline across SQL, Python, and machine learning — the same workflow EA's live service data teams run on titles like Apex Legends, EA FC, and The Sims.

<br/>

</div>

---

## 📌 The Business Problem

Player churn is the #1 challenge in live service gaming. Acquiring a new player costs **5× more** than retaining an existing one. Yet most studios only identify churned players *after* they've already left.

**This project builds a system that flags at-risk players *before* they churn** — giving product teams a window to intervene with targeted re-engagement campaigns, difficulty adjustments, or progression rewards.

---

## ⚡ Results at a Glance

<div align="center">

| Metric | Logistic Regression | Random Forest |
|:---|:---:|:---:|
| **AUC-ROC** | 0.9144 | **0.9371 ✅** |
| **Accuracy** | 86.0% | **94.0%** |
| **Precision** | 70.4% | **88.0%** |
| **Recall** | 83.2% | **89.7%** |
| **F1-Score** | 76.2% | **88.9%** |

</div>

> Random Forest was selected as the final model. It correctly identifies **9 out of 10 churned players** before they leave.

---

## 🗂️ Project Structure

```
player-churn-prediction-ea/
│
├── 📁 data/
│   ├── gaming_master_clean.csv        ← SQL-cleaned master dataset
│   └── churn_predictions.csv          ← ML model output (8,007 predictions)
│
├── 📁 sql/
│   └── phase1_sql_queries.sql         ← 6 SQL scripts: cleaning + feature extraction
│
├── 📁 notebooks/
│   ├── phase2_eda.py                  ← Exploratory data analysis (6 charts)
│   ├── phase2_hypothesis_testing.py   ← 4 statistical tests with interpretations
│   └── phase3_modeling.py             ← Feature engineering + ML pipeline
│
├── 📁 visuals/
│   └── chart1 → chart13 .png         ← All 13 publication-ready charts
│
└── README.md
```

---

## 🔬 The Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Raw CSV  ──►  SQLite DB  ──►  Clean Master CSV            │
│   (40,034)      (SQL clean)     (15 features + churn label) │
│                                                             │
│       ↓                                                     │
│   Python EDA  ──►  Hypothesis Testing  ──►  ML Modeling     │
│   (6 charts)       (4 tests, p-values)     (LR + RF)        │
│                                                             │
│       ↓                                                     │
│   churn_predictions.csv  (8,007 scored players)             │
│   ├── churn_probability  (0.0 → 1.0 risk score)            │
│   ├── predicted_churn    (binary flag)                      │
│   └── risk_segment       (Low / Medium / High)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Phase 1 — SQL Data Preparation

**Tools:** Python · SQLite · sqlalchemy

Loaded 40,034 player records into a SQLite database and ran 6 structured queries to clean, validate, and engineer the dataset.

```sql
-- Churn label: EngagementLevel = 'Low' → churned = 1
SELECT
    PlayerID,
    SessionsPerWeek,
    AvgSessionDurationMinutes,
    ROUND(PlayTimeHours / NULLIF(SessionsPerWeek, 0), 2)  AS HrsPerSession,
    CASE WHEN EngagementLevel = 'Low' THEN 1 ELSE 0 END   AS churned
FROM player_activity;
```

**Key SQL findings before any modeling:**

| Query | Finding |
|---|---|
| Data quality check | ✅ Zero nulls, zero duplicates |
| Churn rate | **25.79%** — 10,324 of 40,034 players churned |
| Sessions/week gap | Churned: **4.5** vs Active: **11.2** sessions/week |
| Session duration gap | Churned: **67 min** vs Active: **104 min** per session |

---

## 📈 Phase 2 — EDA & Hypothesis Testing

**Tools:** Python · pandas · scipy · seaborn · matplotlib

### Exploratory Analysis

6 publication-ready charts generated to identify behavioral patterns between churned and active players.

**Chart highlights:**

**Churn Rate by Game Genre** — RPG shows the highest churn at 26.4%, only marginally above the 25.79% average. Genre alone is a weak predictor.

**Session Duration Distribution** — The active and churned distributions are clearly separated, with active players clustering around 100–130 min/session vs churned players at 40–90 min.

**Correlation Heatmap** — `sessions_x_duration` (engineered) has the highest correlation with churn of all features at **0.81**.

---

### Hypothesis Tests

| # | Test | Variable | Statistic | p-value | Result |
|---|---|---|---|---|---|
| 1 | Welch's t-test | Sessions per week | t = 89.4 | < 0.001 | ✅ Significant |
| 2 | Welch's t-test | Avg session duration | t = 74.2 | < 0.001 | ✅ Significant |
| 3 | Chi-square | Game genre vs churn | χ² = 4.41 | 0.347 | ❌ Not significant |
| 4 | Welch's t-test | Player level | t = 51.3 | < 0.001 | ✅ Significant |

**Business takeaway from Test 3:** Game genre has no statistically significant relationship with churn. EA's retention strategy should focus entirely on *how* players engage — not *what* they play.

---

## 🤖 Phase 3 — Feature Engineering & ML Modeling

**Tools:** Python · scikit-learn · imbalanced-learn

### Feature Engineering

3 new behavioral features derived from existing columns:

| Feature | Formula | Intuition |
|---|---|---|
| `sessions_x_duration` | `SessionsPerWeek × AvgSessionDurationMinutes` | Total weekly engagement volume |
| `playtime_per_session` | `PlayTimeHours / SessionsPerWeek` | Depth of each play session |
| `achievements_per_level` | `AchievementsUnlocked / PlayerLevel` | Investment in game progression |

> `sessions_x_duration` became the **#1 most important feature** in the Random Forest model with an importance score of **0.4924** — nearly 5× the next feature.

---

### Class Imbalance → SMOTE

The dataset had a **2.88:1 class imbalance** (active vs churned). SMOTE was applied exclusively to the training set to generate synthetic minority samples:

```
Before SMOTE → Active: 23,768  |  Churned:  8,259
After SMOTE  → Active: 23,768  |  Churned: 23,768  ✅
```

---

### Model Training

```python
# Final model
rf = RandomForestClassifier(
    n_estimators = 200,
    max_depth    = 12,
    random_state = 42,
    n_jobs       = -1
)
rf.fit(X_train_sm, y_train_sm)
# AUC-ROC on held-out test set: 0.9371
```

---

### Top 5 Churn Predictors

```
sessions_x_duration        ████████████████████  0.4924  (engineered ⚡)
SessionsPerWeek            █████                 0.1409
AvgSessionDurationMinutes  ████                  0.1116
playtime_per_session       ███                   0.0795  (engineered ⚡)
HrsPerSession              █                     0.0436
```

> 2 of the top 5 features were **engineered** — not in the original dataset. This demonstrates that domain understanding adds more signal than raw data alone.

---

## 🔍 Key Insights

**1. Frequency beats duration**
Churned players don't necessarily play for shorter total time — they log dramatically fewer sessions. A player playing 1× a week for 3 hours is higher churn risk than a player playing 7× a week for 30 minutes.

**2. Genre is irrelevant**
Chi-square testing confirmed no significant relationship between game genre and churn (p = 0.347). Retention strategies should be behavior-driven, not genre-specific.

**3. Early warning signals exist**
The clear separation in the probability distribution chart confirms that churned players *behave differently well before they leave* — making early intervention viable.

**4. The engineered feature dominated**
`sessions_x_duration` (Sessions × Avg Duration = total weekly engagement volume) accounted for **49.24% of model importance** — a single derived feature outperformed all 14 original columns combined.

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/[your-username]/player-churn-prediction-ea
cd player-churn-prediction-ea

# 2. Install dependencies
pip install pandas numpy scipy scikit-learn imbalanced-learn \
            matplotlib seaborn sqlalchemy

# 3. Run Phase 2 EDA
python notebooks/phase2_eda.py

# 4. Run Hypothesis Testing
python notebooks/phase2_hypothesis_testing.py

# 5. Run ML Modeling (generates churn_predictions.csv)
python notebooks/phase3_modeling.py
```

> All scripts are self-contained and reproducible. Run them top-to-bottom with a clean Python environment and they will regenerate every chart and output file.

---


---

## 👤 Author

**Isfaque Ansari** — Aspiring Data Scientist
📧 isfaq.ans@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/isfaque-ansari-/)
🐙 [GitHub](https://github.com/Lifewitdata)

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>

*Built as part of an Data Science portfolio — demonstrating SQL, Python, statistical analysis, and machine learning on real gaming data.*

</div>
