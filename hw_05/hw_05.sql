USE baseball;

-- SHOW FIELDS FROM batter_counts;


-- get sum of bats and hits for each player over all time
CREATE OR REPLACE TABLE historical_average AS
SELECT batter, SUM(atBat) AS Bats, SUM(Hit) AS Hits, (CASE WHEN SUM(atBat) > 0 THEN SUM(Hit) / SUM(atBat) ELSE 0 END) AS Average
FROM batter_counts
GROUP BY batter
;


--  get avg bats for each year for each batter
CREATE OR REPLACE TABLE annual_average AS
SELECT bc.batter, EXTRACT(YEAR FROM g.local_date) AS years, (CASE WHEN SUM(bc.atBat) > 0 THEN SUM(bc.Hit) / SUM(bc.atBat) ELSE 0 END) AS Average, COUNT(*) AS cnt
FROM batter_counts bc
    JOIN game g ON bc.game_id = g.game_id
GROUP BY EXTRACT(YEAR FROM g.local_date), bc.batter --  extract inspired by this page https://learnsql.com/cookbook/how-to-group-by-year-in-sql/
HAVING COUNT(*) > 1   -- to eliminate players who only played once (Julien told me to implement that)
ORDER BY bc.batter DESC
;



CREATE OR REPLACE TABLE base_table AS
SELECT bc.batter, bc.game_id, g.local_date AS dates, bc.Hit, bc.atBat
FROM batter_counts bc
    INNER JOIN game g ON bc.game_id = g.game_id
-- WHERE bc.batter = 120074
GROUP BY bc.batter, g.local_date
ORDER BY g.local_date
;

CREATE INDEX ID ON base_table(batter);
ALTER TABLE base_table ADD PRIMARY KEY (batter, game_id);

-- Runtime: 2:10 min
CREATE OR REPLACE TABLE rolling_average AS
SELECT b1.game_id, b1.dates, b1.batter, (CASE WHEN SUM(b2.atBat) > 0 THEN SUM(b2.Hit) / SUM(b2.atBat) ELSE 0 END) AS batting_average1, SUM(b2.atBat) AS batting_summary
    , b1.atBat, b1.Hit
FROM base_table b1
    LEFT JOIN base_table b2 ON b1.batter = b2.batter AND b2.dates > DATE_SUB(b1.dates, INTERVAL 100 DAY) AND b2.dates < b1.dates -- must be in join
--  otherwise the code does not work
GROUP BY b1.dates, b1.batter
ORDER BY b1.dates, b1.batter
;








CREATE OR REPLACE TABLE base_table_Home AS
SELECT g.game_id, tc.team_id, tc.homeTeam, tc.Walk, tc.Hit, tc.inning, g.local_date, tc.Home_Run, tc.Sac_Fly, tc.Strikeout, tc.Hit_By_Pitch, tc.atBat
FROM team_batting_counts tc
    INNER JOIN game g ON tc.game_id = g.game_id
WHERE tc.homeTeam = 1
GROUP BY tc.game_id, tc.team_id
ORDER BY tc.game_id
;

CREATE OR REPLACE TABLE base_table_Away AS
SELECT g.game_id, tc.team_id, tc.homeTeam, tc.Walk, tc.Hit, tc.inning, g.local_date, tc.Home_Run, tc.Sac_Fly, tc.Strikeout, tc.Hit_By_Pitch, tc.atBat
FROM team_batting_counts tc
    INNER JOIN game g ON tc.game_id = g.game_id
WHERE tc.awayTeam = 1
GROUP BY tc.game_id, tc.team_id
ORDER BY tc.game_id
;

ALTER TABLE base_table_Home ADD PRIMARY KEY (team_id, game_id);
ALTER TABLE base_table_Away ADD PRIMARY KEY (team_id, game_id);


-- Runtime: 2:10 min
CREATE OR REPLACE TABLE rolling_Home AS
SELECT b1.game_id, b1.team_id, b2.inning, b2.local_date AS date2, b1.local_date AS dates, (CASE WHEN SUM(b2.inning) > 0 THEN (SUM(b2.Walk) + SUM(b2.Hit)) / SUM(b2.inning) ELSE 0 END) AS WHIP, (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) - SUM(b2.Home_Run)) / (SUM(b2.atBat) - SUM(b2.Strikeout) - SUM(b2.Home_Run) + SUM(b2.Sac_Fly)) ELSE 0 END) AS BABIP, (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) + SUM(b2.Hit_By_Pitch)) / (SUM(b2.atBat) + SUM(b2.Hit_By_Pitch) + SUM(b2.Walk) + SUM(b2.Sac_Fly)) ELSE 0 END) AS OBP -- on-base percentage
FROM base_table_Home b1
    LEFT JOIN base_table_Home b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date -- must be in join
--  otherwise the code does not work
GROUP BY b1.game_id
ORDER BY b1.game_id
;

CREATE OR REPLACE TABLE rolling_Away AS
SELECT b1.game_id, b1.team_id, b2.inning, b2.local_date AS date2, b1.local_date AS dates, (CASE WHEN SUM(b2.inning) > 0 THEN (SUM(b2.Walk) + SUM(b2.Hit)) / SUM(b2.inning) ELSE 0 END) AS WHIP, (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) - SUM(b2.Home_Run)) / (SUM(b2.atBat) - SUM(b2.Strikeout) - SUM(b2.Home_Run) + SUM(b2.Sac_Fly)) ELSE 0 END) AS BABIP, (CASE WHEN SUM(b2.atBat) > 0 THEN (SUM(b2.Hit) + SUM(b2.Walk) + SUM(b2.Hit_By_Pitch)) / (SUM(b2.atBat) + SUM(b2.Hit_By_Pitch) + SUM(b2.Walk) + SUM(b2.Sac_Fly)) ELSE 0 END) AS OBP -- on-base percentage
FROM base_table_Away b1
    LEFT JOIN base_table_Away b2 ON b1.team_id = b2.team_id AND b2.local_date > DATE_SUB(b1.local_date, INTERVAL 100 DAY)
        AND b2.local_date < b1.local_date -- must be in join
--  otherwise the code does not work
GROUP BY b1.game_id
ORDER BY b1.game_id
;

-- SELECT * FROM rolling_Away;

ALTER TABLE rolling_Home ADD PRIMARY KEY (team_id, game_id);
ALTER TABLE rolling_Away ADD PRIMARY KEY (team_id, game_id);

SELECT ra.game_id, ra.team_id, rh.team_id, (rh.WHIP - ra.WHIP) AS WHIP_DIFF
FROM rolling_Away ra
    JOIN rolling_Home rh ON ra.game_id = ra.game_id
;
