USE baseball;

CREATE OR REPLACE TABLE base_table_Home AS
SELECT tc.game_id, tc.team_id, tc.homeTeam, tc.Walk, tc.Hit, tc.inning, g.local_date, tc.Strikeout, tc.atBat, tc.Sac_Fly
    , tc.Home_Run, tc.toBase, tc.Hit_By_Pitch, tc.win, g.stadium_id, tc.finalScore, tc.opponent_finalScore
FROM team_batting_counts tc
    INNER JOIN game g ON tc.game_id = g.game_id
WHERE tc.homeTeam = 1
GROUP BY tc.game_id, tc.team_id
ORDER BY tc.game_id
;

CREATE OR REPLACE TABLE base_table_Away AS
SELECT g.game_id, tc.team_id, tc.homeTeam, tc.Walk, tc.Hit, tc.inning, g.local_date, tc.Strikeout, tc.atBat, tc.Sac_Fly
    , tc.Home_Run, tc.toBase, tc.Hit_By_Pitch
FROM team_batting_counts tc
    INNER JOIN game g ON tc.game_id = g.game_id
WHERE tc.awayTeam = 1
GROUP BY tc.game_id, tc.team_id
ORDER BY tc.game_id
;

ALTER TABLE base_table_Home ADD PRIMARY KEY (team_id, game_id);
ALTER TABLE base_table_Away ADD PRIMARY KEY (team_id, game_id);


CREATE OR REPLACE TABLE rolling_Home AS
SELECT b1.game_id, b1.team_id, b1.win, b1.stadium_id, b1.finalScore, b1.opponent_finalScore
    , (CASE WHEN SUM(b2.inning) > 0 THEN (SUM(b2.Walk) + SUM(b2.Hit)) / SUM(b2.inning) ELSE 0 END) AS WHIP, b2.inning, b2.local_date AS date2, b1.local_date AS date1
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) - SUM(b2.Home_Run)) / (SUM(b2.atBat) - SUM(b2.Strikeout) - b2.Home_Run + SUM(b2.Sac_Fly)) ELSE 0 END) AS BABIP
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) * b2.toBase) / (SUM(b2.atBat) + SUM(b2.Walk)) ELSE 0 END) AS RC
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) + SUM(b2.Hit_By_Pitch)) / (SUM(b2.atBat) + SUM(b2.Hit_By_Pitch) + SUM(b2.Walk) + SUM(b2.Sac_Fly)) ELSE 0 END) AS OBP -- on-base percentage
    , (CASE WHEN SUM(b2.inning) > 0 THEN SUM(b2.Hit) / (SUM(b2.inning)) ELSE 0 END) AS HIP
    , (CASE WHEN SUM(b2.Home_Run) > 0 THEN SUM(b2.atBat) / (SUM(b2.Home_Run)) ELSE 0 END) AS AB_HR
    , (CASE WHEN SUM(b2.Strikeout) > 0 THEN SUM(b2.Walk) / (SUM(b2.Strikeout)) ELSE 0 END) AS BB_K
FROM base_table_Home b1
    LEFT JOIN base_table_Home b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date
GROUP BY b1.game_id
ORDER BY b1.game_id
;

CREATE OR REPLACE TABLE rolling_Away AS
SELECT b1.game_id, b1.team_id, (CASE WHEN SUM(b2.inning) > 0 THEN (SUM(b2.Walk) + SUM(b2.Hit)) / SUM(b2.inning) ELSE 0 END) AS WHIP, b2.inning, b2.local_date AS date2, b1.local_date AS date1
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) - SUM(b2.Home_Run)) / (SUM(b2.atBat) - SUM(b2.Strikeout) - b2.Home_Run + SUM(b2.Sac_Fly)) ELSE 0 END) AS BABIP
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) * b2.toBase) / (SUM(b2.atBat) + SUM(b2.Walk)) ELSE 0 END) AS RC
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) + SUM(b2.Hit_By_Pitch)) / (SUM(b2.atBat) + SUM(b2.Hit_By_Pitch) + SUM(b2.Walk) + SUM(b2.Sac_Fly)) ELSE 0 END) AS OBP -- on-base percentage
    , (CASE WHEN SUM(b2.inning) > 0 THEN SUM(b2.Hit) / (SUM(b2.inning)) ELSE 0 END) AS HIP
    , (CASE WHEN SUM(b2.Home_Run) > 0 THEN SUM(b2.atBat) / (SUM(b2.Home_Run)) ELSE 0 END) AS AB_HR
    , (CASE WHEN SUM(b2.Strikeout) > 0 THEN SUM(b2.Walk) / (SUM(b2.Strikeout)) ELSE 0 END) AS BB_K
FROM base_table_Away b1
    LEFT JOIN base_table_Away b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date
GROUP BY b1.game_id
ORDER BY b1.game_id
;

ALTER TABLE rolling_Home ADD PRIMARY KEY (game_id);
ALTER TABLE rolling_Away ADD PRIMARY KEY (game_id);



-- ---------------------------------
-- starting pitcher
-- create base tables
CREATE OR REPLACE TABLE base_table_Home_pitcher AS
SELECT tc.game_id, tc.team_id, tc.homeTeam, g.local_date, ((tc.endingInning - tc.startingInning) + 1) AS Innings_Played, tc.Strikeout
    , tc.Hit, tc.Walk, tc.Home_Run
FROM pitcher_counts tc
    INNER JOIN game g ON tc.game_id = g.game_id
WHERE tc.homeTeam = 1 AND tc.startingPitcher = 1
GROUP BY tc.game_id, tc.team_id
ORDER BY tc.game_id
;

CREATE OR REPLACE TABLE base_table_Away_pitcher AS
SELECT tc.game_id, tc.team_id, tc.homeTeam, g.local_date, ((tc.endingInning - tc.startingInning) + 1) AS Innings_Played, tc.Strikeout
    , tc.Hit, tc.Walk, tc.Home_Run
FROM pitcher_counts tc
    INNER JOIN game g ON tc.game_id = g.game_id
WHERE tc.awayTeam = 1 AND tc.startingPitcher = 1
GROUP BY tc.game_id, tc.team_id
ORDER BY tc.game_id
;

ALTER TABLE base_table_Home_pitcher ADD PRIMARY KEY (team_id, game_id);
ALTER TABLE base_table_Away_pitcher ADD PRIMARY KEY (team_id, game_id);

-- create rolling table for last 100 days
CREATE OR REPLACE TABLE rolling_Home_pitcher AS
SELECT b1.game_id, b1.team_id, (CASE WHEN SUM(b2.Walk) > 0 THEN (SUM(b2.Walk) * 9) / (SUM(b2.Innings_Played)) ELSE 0 END) AS BB_9IP
    , (CASE WHEN SUM(b2.Strikeout) > 0 THEN (SUM(b2.Strikeout) * 9) / (SUM(b2.Innings_Played)) ELSE 0 END) AS K9_pitcher
    , (CASE WHEN SUM(b2.Strikeout) > 0 THEN (SUM(b2.Strikeout) + SUM(b2.Walk)) / (SUM(b2.Innings_Played)) ELSE 0 END) AS PFR  -- Power finesse ratio
    , (CASE WHEN SUM(b2.Innings_Played) > 0 THEN ((3 + (13 * b2.Home_Run + 3 * (b2.Walk + b2.Hit) - 2 * b2.Strikeout)) / b2.Innings_Played) ELSE 0 END) AS DICE -- Defense-Independent Component
FROM base_table_Home_pitcher b1
    LEFT JOIN base_table_Home_pitcher b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date
GROUP BY b1.game_id
ORDER BY b1.game_id
;

CREATE OR REPLACE TABLE rolling_Away_pitcher AS
SELECT b1.game_id, b1.team_id, (CASE WHEN SUM(b2.Walk) > 0 THEN (SUM(b2.Walk) * 9) / (SUM(b2.Innings_Played)) ELSE 0 END) AS BB_9IP
    , (CASE WHEN SUM(b2.Strikeout) > 0 THEN (SUM(b2.Strikeout) * 9) / (SUM(b2.Innings_Played)) ELSE 0 END) AS K9_pitcher
    , (CASE WHEN SUM(b2.Strikeout) > 0 THEN (SUM(b2.Strikeout) + SUM(b2.Walk)) / (SUM(b2.Innings_Played)) ELSE 0 END) AS PFR  -- Power finesse ratio
    , (CASE WHEN SUM(b2.Innings_Played) > 0 THEN ((3 + (13 * b2.Home_Run + 3 * (b2.Walk + b2.Hit) - 2 * b2.Strikeout)) / b2.Innings_Played) ELSE 0 END) AS DICE -- Defense-Independent Component
FROM base_table_Away_pitcher b1
    LEFT JOIN base_table_Away_pitcher b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY) AND b2.local_date < b1.local_date
GROUP BY b1.game_id
ORDER BY b1.game_id
;

ALTER TABLE rolling_Home_pitcher ADD PRIMARY KEY (game_id, team_id);
ALTER TABLE rolling_Away_pitcher ADD PRIMARY KEY (game_id, team_id);


-- ------------------------------
-- create two pre final tables to join later

CREATE OR REPLACE TABLE final_editable_team AS
SELECT rh.game_id, rh.team_id AS Home, ra.team_id AS Away, rh.win AS HomeTeamWins, rh.stadium_id, rh.finalScore, rh.opponent_finalScore
    , (rh.WHIP - ra.WHIP) AS Difference_WHIP
    , (rh.BABIP - ra.BABIP) AS Difference_BABIP
    , (rh.OBP - ra.OBP) AS Difference_OBP
    , (rh.RC - ra.RC) AS Difference_RC
    , (rh.AB_HR - ra.AB_HR) AS Difference_AB_HR
    , (rh.BB_K - ra.BB_K) AS Difference_BB_K
FROM rolling_Home rh
    JOIN rolling_Away ra ON rh.game_id = ra.game_id
GROUP BY ra.game_id
;

CREATE OR REPLACE TABLE final_editable_pitcher AS
SELECT rh.game_id, rh.team_id AS Home, ra.team_id AS Away, (rh.BB_9IP - ra.BB_9IP) AS Difference_BB_9IP
    , (rh.K9_pitcher - ra.K9_pitcher) AS Difference_K9_pitcher
    , (rh.PFR - ra.PFR) AS Difference_PFR
    , (rh.DICE - ra.DICE) AS Difference_Dice
FROM rolling_Home_pitcher rh
    JOIN rolling_Away_pitcher ra ON rh.game_id = ra.game_id
GROUP BY ra.game_id
;


ALTER TABLE final_editable_team ADD PRIMARY KEY (game_id);
ALTER TABLE final_editable_pitcher ADD PRIMARY KEY (game_id);

-- ------------------------------
-- Join both table to have stats of team and pitcher in one table
CREATE OR REPLACE TABLE final_Stats AS
SELECT rh.game_id, rh.HomeTeamWins, rh.Home, rh.stadium_id, rh.Away, rh.Difference_WHIP, rh.Difference_BABIP, rh.Difference_OBP, rh.Difference_RC, rh.Difference_AB_HR
    , rh.Difference_BB_K
    , ra.Difference_K9_pitcher, ra.Difference_BB_9IP, ra.Difference_PFR, ra.Difference_Dice
    , rh.finalScore, rh.opponent_finalScore
FROM final_editable_team rh
    JOIN final_editable_pitcher ra ON rh.game_id = ra.game_id
GROUP BY ra.game_id
;
