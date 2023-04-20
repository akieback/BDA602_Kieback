CREATE OR REPLACE TABLE base_table_Home AS
SELECT g.game_id, tc.team_id, tc.homeTeam, tc.Walk, tc.Hit, tc.inning, g.local_date, tc.Strikeout, tc.atBat, tc.Sac_Fly
    , tc.Home_Run, tc.toBase, tc.Hit_By_Pitch
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
SELECT b1.game_id, b1.team_id, (CASE WHEN SUM(b2.inning) > 0 THEN (SUM(b2.Walk) + SUM(b2.Hit)) / SUM(b2.inning) ELSE 0 END) AS WHIP, b2.inning, b2.local_date AS date2, b1.local_date AS date1
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) - SUM(b2.Home_Run)) / (SUM(b2.atBat) - SUM(b2.Strikeout) - b2.Home_Run + SUM(b2.Sac_Fly)) ELSE 0 END) AS BABIP
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) * b2.toBase) / (SUM(b2.atBat) + SUM(b2.Walk)) ELSE 0 END) AS RC
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) + SUM(b2.Hit_By_Pitch)) / (SUM(b2.atBat) + SUM(b2.Hit_By_Pitch) + SUM(b2.Walk) + SUM(b2.Sac_Fly)) ELSE 0 END) AS OBP -- on-base percentage
FROM base_table_Home b1
    LEFT JOIN base_table_Home b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date
GROUP BY b1.game_id
ORDER BY b1.game_id
;

-- SELECT * FROM base_table_Home;
-- SELECT * FROM rolling_Home;

CREATE OR REPLACE TABLE rolling_Away AS
SELECT b1.game_id, b1.team_id, (CASE WHEN SUM(b2.inning) > 0 THEN (SUM(b2.Walk) + SUM(b2.Hit)) / SUM(b2.inning) ELSE 0 END) AS WHIP, b2.inning, b2.local_date AS date2, b1.local_date AS date1
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) - SUM(b2.Home_Run)) / (SUM(b2.atBat) - SUM(b2.Strikeout) - b2.Home_Run + SUM(b2.Sac_Fly)) ELSE 0 END) AS BABIP
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) * b2.toBase) / (SUM(b2.atBat) + SUM(b2.Walk)) ELSE 0 END) AS RC
    , (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) + SUM(b2.Hit_By_Pitch)) / (SUM(b2.atBat) + SUM(b2.Hit_By_Pitch) + SUM(b2.Walk) + SUM(b2.Sac_Fly)) ELSE 0 END) AS OBP -- on-base percentage
FROM base_table_Away b1
    LEFT JOIN base_table_Away b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date
GROUP BY b1.game_id
ORDER BY b1.game_id
;

ALTER TABLE rolling_Home ADD PRIMARY KEY (game_id);
ALTER TABLE rolling_Away ADD PRIMARY KEY (game_id);


CREATE OR REPLACE TABLE final_editable AS
SELECT rh.game_id, rh.team_id AS Home, ra.team_id AS Away, (rh.WHIP - ra.WHIP) AS Difference_WHIP, rh.WHIP AS H_Whip, ra.WHIP AS A_Whip, (rh.BABIP - ra.BABIP) AS Difference_BABIP
    , (rh.OBP - ra.OBP) AS Difference_OBP, (rh.RC - ra.RC) AS Difference_RC
FROM rolling_Home rh
    JOIN rolling_Away ra ON rh.game_id = ra.game_id
GROUP BY ra.game_id
;
