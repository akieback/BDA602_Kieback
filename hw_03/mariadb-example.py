# mariadb-example.py
from getpass import (  # https://stackoverflow.com/questions/9202224/getting-a-hidden-password-input
    getpass,
)

from pyspark import StorageLevel
from pyspark.sql import SparkSession

appName = "PySpark Example - MariaDB Example"
master = "local"
# Create Spark session
spark = (
    SparkSession.builder.appName(appName)
    .master(master)
    .config("spark.driver.extraClassPath", "./mariadb-java-client-3.1.2.jar")
    .getOrCreate()
)

database = "baseball"
user = "user"
password = getpass()
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"
sql_bc = "SELECT batter, game_id, Hit, atBat FROM baseball.batter_counts"
sql_g = "SELECT game_id, local_date FROM baseball.game"

# Create a data frame by reading data from Oracle via JDBC
df_bc = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql_bc)
    .option("user", user)
    .option("password", password)
    .option("driver", jdbc_driver)
    .load()
)
df_bc.show()

df_bc.createOrReplaceTempView("batter_counts")
df_bc.persist(StorageLevel.MEMORY_ONLY)

df_g = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql_g)
    .option("user", user)
    .option("password", password)
    .option("driver", jdbc_driver)
    .load()
)
df_g.show()

df_g.createOrReplaceTempView("game")
df_g.persist(StorageLevel.MEMORY_ONLY)

df_base_table = spark.sql(
    """
    SELECT bc.batter, bc.game_id, g.local_date AS dates, bc.Hit, bc.atBat
    FROM batter_counts bc
        INNER JOIN game g ON bc.game_id = g.game_id
    WHERE bc.batter = 120074
    GROUP BY bc.batter, g.local_date, bc.game_id, bc.Hit, bc.atBat
    ORDER BY g.local_date;
    """
)

df_base_table.createOrReplaceTempView("base_table")
df_base_table.persist(StorageLevel.MEMORY_ONLY)

df_rolling_average = spark.sql(
    """
    SELECT  b1.game_id, b1.dates, b1.batter,
            (CASE WHEN SUM(b2.atBat) > 0 THEN SUM(b2.Hit) / SUM(b2.atBat) ELSE 0 END) AS batting_average1,
            SUM(b2.atBat) AS batting_summary
            , b1.atBat, b1.Hit
    FROM base_table b1
        LEFT JOIN base_table b2 ON b1.batter = b2.batter AND b2.dates > DATE_SUB(b1.dates, 100)
        AND b2.dates < b1.dates -- must be in join
    --  otherwise the code does not work
    GROUP BY b1.game_id, b1.dates, b1.batter, b1.atBat, b1.Hit
    ORDER BY b1.dates, b1.batter
    """
)

df_rolling_average.createOrReplaceTempView("rolling_average")
df_rolling_average.persist(StorageLevel.MEMORY_ONLY)

df_count = spark.sql(
    """
    SELECT Count(*)
    FROM rolling_average
    """
)

df_count.show()

df_rolling_average.show()
