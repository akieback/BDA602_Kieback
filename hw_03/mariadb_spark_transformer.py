# mariadb_spark_transformer.py
import sys
from getpass import (  # https://stackoverflow.com/questions/9202224/getting-a-hidden-password-input
    getpass,
)

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession


class BatterAvgTransformer(Transformer):

    # concept from the lecture notes
    # initiate the class
    def __init__(self):
        super(BatterAvgTransformer, self).__init__()
        self.appName = "MariaDB Baseball Test"
        self.master = "local"
        self.driverpath = "spark.driver.extraClassPath"
        self.jarname = "./mariadb-java-client-3.1.2.jar"
        # Create Spark session
        self.spark = (
            SparkSession.builder.appName(self.appName)
            .master(self.master)
            .config(self.driverpath, self.jarname)
            .getOrCreate()
        )
        self.database = "baseball"
        self.user = input("input your user: ")
        self.server = "localhost"
        self.port = 3306
        return

    # build a spark connection and return a base table
    # which can later be used in the transformer
    def spark_con(self):
        password = getpass()  # getpass so the input is not visible
        # https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database
        jdbc_url = (
            f"jdbc:mysql://{self.server}:{self.port}/{self.database}?permitMysqlScheme"
        )
        jdbc_driver = "org.mariadb.jdbc.Driver"
        # get only needed table and columns
        sql_bc = "SELECT batter, game_id, Hit, atBat FROM baseball.batter_counts"
        sql_g = "SELECT game_id, local_date FROM baseball.game"

        # Create a data frame by reading data from Oracle via JDBC
        df_bc = (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", sql_bc)
            .option("user", self.user)
            .option("password", password)
            .option("driver", jdbc_driver)
            .load()
        )
        df_bc.show()

        df_bc.createOrReplaceTempView("batter_counts")
        df_bc.persist(StorageLevel.MEMORY_ONLY)  # storing everything only in memory

        df_g = (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", sql_g)
            .option("user", self.user)
            .option("password", password)
            .option("driver", jdbc_driver)
            .load()
        )
        # df_g.show()

        df_g.createOrReplaceTempView("game")
        df_g.persist(StorageLevel.MEMORY_ONLY)

        df_base_table = self.spark.sql(
            """
            SELECT bc.batter, bc.game_id, g.local_date AS dates, bc.Hit, bc.atBat
            FROM batter_counts bc
                INNER JOIN game g ON bc.game_id = g.game_id
            --WHERE bc.batter = 120074
            GROUP BY bc.batter, g.local_date, bc.game_id, bc.Hit, bc.atBat
            ORDER BY g.local_date;
            """
        )

        df_base_table.createOrReplaceTempView("base_table")
        df_base_table.persist(StorageLevel.MEMORY_ONLY)

        return df_base_table

    def _transform(self, df):

        # Calculates the average of hits per at-bat for each batter over the last 100 days

        df_rolling_average = self.spark.sql(
            """
            SELECT  b1.game_id, b1.dates, b1.batter,
                    (CASE WHEN SUM(b2.atBat) > 0 THEN SUM(b2.Hit) / SUM(b2.atBat) ELSE 0 END) AS batting_average1,
                    SUM(b2.atBat) AS batting_summary,
                    b1.atBat, b1.Hit
            FROM base_table b1
                LEFT JOIN base_table b2 ON b1.batter = b2.batter
                AND b2.dates > DATE_SUB(b1.dates, 100)
                AND b2.dates < b1.dates -- must be in join
            --  otherwise the code does not work
            GROUP BY b1.game_id, b1.dates, b1.batter, b1.atBat, b1.Hit
            ORDER BY b1.dates, b1.batter
            """
        )
        return df_rolling_average


def main():
    # create the transformer instance
    batter_avg_transformer = BatterAvgTransformer()

    # get base table to use in trsanformer
    base_df = batter_avg_transformer.spark_con()

    # transform the data
    df_transformed = batter_avg_transformer.transform(base_df)

    # show the result
    df_transformed.show()


if __name__ == "__main__":
    sys.exit(main())
