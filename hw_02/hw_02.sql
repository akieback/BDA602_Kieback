USE bASeball;

#SHOW FIELDS FROM batter_counts;


#SELECT *
#FROM batter_counts
#LIMIT 0,20
#;


#get sum of bats and hits for each player over all time
SELECT batter, SUM(atBat) AS Bats, SUM(Hit) AS Hits, SUM(Hit) / SUM(atBat) AS Average
FROM batter_counts
GROUP BY batter
LIMIT 0, 20
;


# get avg bats for each year for each batter
SELECT bc.batter, EXTRACT(YEAR FROM g.local_date) AS years, SUM(bc.Hit) / SUM(bc.atBat) AS Average, COUNT(*) AS cnt
FROM batter_counts bc
    JOIN game g ON bc.game_id = g.game_id
GROUP BY EXTRACT(YEAR FROM g.local_date), bc.batter # extract inspired by this page https://learnsql.com/cookbook/how-to-group-by-year-in-sql/
ORDER BY bc.batter DESC
;
#LIMIT 0, 20;
#having count > 1       self join
#window function



    #indexis on rolling table
