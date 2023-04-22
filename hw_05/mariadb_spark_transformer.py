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
        self.jarname = "mariadb-java-client-3.1.2.jar"
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
        final = "SELECT * FROM final_Stats"

        # Create a data frame by reading data from Oracle via JDBC
        df_Final = (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", final)
            .option("user", self.user)
            .option("password", password)
            .option("driver", jdbc_driver)
            .load()
        )
        df_Final.show()

        df_Final.createOrReplaceTempView("final_editable")
        df_Final.persist(StorageLevel.MEMORY_ONLY)  # storing everything only in memory

        return df_Final


'''
    def _transform(self, df_rolling_Home, df_rolling_Away):

        # Calculates the average of hits per at-bat for each batter over the last 100 days

        df_rolling_average = self.spark.sql(
            """
            SELECT rh.game_id, rh.team_id AS Home, ra.team_id AS Away, (rh.WHIP-ra.WHIP) AS Difference_WHIP,
            rh.WHIP as H_Whip, ra.WHIP AS A_Whip, (rh.BABIP - ra.BABIP) AS Difference_BABIP
            FROM rolling_Home rh
                JOIN rolling_Away ra ON rh.game_id = ra.game_id
            GROUP BY ra.game_id;
            """
        )
        return df_rolling_average
'''


def get_data():
    # create the transformer instance
    batter_avg_transformer = BatterAvgTransformer()

    # get base table to use in transformer
    df_Final = batter_avg_transformer.spark_con()

    # transform the data
    # df_transformed = batter_avg_transformer.transform(base_df_Home, base_df_Away)

    # show the result
    df_Final.show()
    predictors = []
    for col in df_Final.columns:
        predictors.append(col)
    non_predictors = ["game_id", "Home", "Away", "HomeTeamWins"]
    for drop in non_predictors:
        predictors.remove(drop)

    df_Final2 = (
        df_Final.toPandas()
    )  # https://sparkbyexamples.com/pandas/convert-pyspark-dataframe-to-pandas/
    response = "HomeTeamWins"
    df_Final2.dropna()
    return df_Final2, predictors, response


if __name__ == "__main__":
    sys.exit(get_data())
