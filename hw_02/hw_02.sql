USE baseball;

-- SHOW FIELDS FROM batter_counts;


-- get sum of bats and hits for each player over all time
CREATE OR REPLACE TABLE historical_average AS
SELECT batter, SUM(atBat) AS Bats, SUM(Hit) AS Hits, SUM(Hit) / SUM(atBat) AS Average
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
