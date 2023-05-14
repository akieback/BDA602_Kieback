import sys

import pandas as pd
import sqlalchemy


def get_data():
    # sql connection
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture04.html#/7/3
    db_user = "root"
    db_pass = "root"  # pragma: allowlist secret
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@mariadb:3306/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = "SELECT * FROM final_Stats"

    df_Final = pd.read_sql_query(query, sql_engine)
    predictors = []
    df_Final.head()

    for col in df_Final.columns:
        predictors.append(col)
    non_predictors = [
        "game_id",
        "Home",
        "Away",
        "HomeTeamWins",
        "finalScore",
        "opponent_finalScore",
    ]
    for drop in non_predictors:
        predictors.remove(drop)
    # https://sparkbyexamples.com/pandas/convert-pyspark-dataframe-to-pandas/
    response = "HomeTeamWins"
    df_Final.dropna()
    return df_Final, predictors, response


if __name__ == "__main__":
    sys.exit(get_data())
