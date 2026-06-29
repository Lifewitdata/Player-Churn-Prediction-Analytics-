-- ============================================================
-- PROJECT: Player Churn Prediction & Analytics Dashboard
-- DATASET: Online Gaming Behavior (40,034 players)
-- PHASE 1: SQL Data Preparation
-- TOOL: SQLite via Python sqlalchemy
-- ============================================================


-- ─────────────────────────────────────────────
-- QUERY 1: Data Quality Check
-- Purpose: Confirm no NULLs or duplicates exist
-- ─────────────────────────────────────────────
SELECT 
    COUNT(*)                                                            AS total_rows,
    COUNT(DISTINCT PlayerID)                                            AS unique_players,
    SUM(CASE WHEN PlayerID IS NULL THEN 1 ELSE 0 END)                  AS null_playerids,
    SUM(CASE WHEN EngagementLevel IS NULL THEN 1 ELSE 0 END)           AS null_engagement
FROM player_activity;

-- Result: 40,034 rows | 40,034 unique players | 0 NULLs | 0 duplicates ✅


-- ─────────────────────────────────────────────
-- QUERY 2: Create Churn Label
-- Business rule: EngagementLevel = 'Low' → churned = 1
-- ─────────────────────────────────────────────
SELECT 
    PlayerID,
    Age,
    Gender,
    Location,
    GameGenre,
    PlayTimeHours,
    InGamePurchases,
    GameDifficulty,
    SessionsPerWeek,
    AvgSessionDurationMinutes,
    PlayerLevel,
    AchievementsUnlocked,
    EngagementLevel,
    CASE WHEN EngagementLevel = 'Low' THEN 1 ELSE 0 END                AS churned
FROM player_activity;

-- Result: Churn rate = 25.79% (10,324 churned / 40,034 total)


-- ─────────────────────────────────────────────
-- QUERY 3: Feature Extraction Per Player
-- Purpose: Create engineered features for ML model
-- ─────────────────────────────────────────────
SELECT 
    PlayerID,
    ROUND(PlayTimeHours, 2)                                             AS total_playtime_hrs,
    SessionsPerWeek                                                     AS sessions_per_week,
    AvgSessionDurationMinutes                                           AS avg_session_mins,
    InGamePurchases                                                     AS made_purchase,
    ROUND(PlayTimeHours / NULLIF(SessionsPerWeek, 0), 2)               AS hrs_per_session,
    PlayerLevel                                                         AS player_level,
    AchievementsUnlocked                                                AS achievements,
    CASE WHEN EngagementLevel = 'Low' THEN 1 ELSE 0 END                AS churned
FROM player_activity;


-- ─────────────────────────────────────────────
-- QUERY 4: Churn Rate by Game Genre
-- Business question: Which game genre has highest churn?
-- ─────────────────────────────────────────────
SELECT 
    GameGenre,
    COUNT(*)                                                            AS total_players,
    SUM(CASE WHEN EngagementLevel = 'Low' THEN 1 ELSE 0 END)           AS churned_players,
    ROUND(
        100.0 * SUM(CASE WHEN EngagementLevel = 'Low' THEN 1 ELSE 0 END) 
        / COUNT(*), 2
    )                                                                   AS churn_rate_pct
FROM player_activity
GROUP BY GameGenre
ORDER BY churn_rate_pct DESC;

-- Result: RPG = 26.37% (highest) | Strategy = 25.04% (lowest)


-- ─────────────────────────────────────────────
-- QUERY 5: Avg Metrics — Churned vs Active
-- Business question: How do churned players behave differently?
-- ─────────────────────────────────────────────
SELECT 
    CASE WHEN EngagementLevel = 'Low' THEN 'Churned' ELSE 'Active' END AS player_status,
    COUNT(*)                                                            AS total_players,
    ROUND(AVG(InGamePurchases), 3)                                      AS avg_purchases,
    ROUND(AVG(PlayTimeHours), 2)                                        AS avg_playtime_hrs,
    ROUND(AVG(AvgSessionDurationMinutes), 1)                            AS avg_session_mins,
    ROUND(AVG(SessionsPerWeek), 1)                                      AS avg_sessions_per_week,
    ROUND(AVG(PlayerLevel), 1)                                          AS avg_player_level
FROM player_activity
GROUP BY player_status;

-- Key Finding: Churned players have 4.5 sessions/week vs 11.2 for active players
-- Key Finding: Churned players avg 66.9 min/session vs 104.5 for active players


-- ─────────────────────────────────────────────
-- QUERY 6: Final Master Dataset Export
-- Purpose: Clean dataset with churn label + engineered feature
-- ─────────────────────────────────────────────
SELECT 
    PlayerID,
    Age,
    Gender,
    Location,
    GameGenre,
    GameDifficulty,
    ROUND(PlayTimeHours, 2)                                             AS PlayTimeHours,
    InGamePurchases,
    SessionsPerWeek,
    AvgSessionDurationMinutes,
    PlayerLevel,
    AchievementsUnlocked,
    EngagementLevel,
    ROUND(PlayTimeHours / NULLIF(SessionsPerWeek, 0), 2)               AS HrsPerSession,
    CASE WHEN EngagementLevel = 'Low' THEN 1 ELSE 0 END                AS churned
FROM player_activity;

-- Output: gaming_master_clean.csv | 40,034 rows × 15 columns
