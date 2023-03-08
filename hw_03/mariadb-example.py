# mariadb-example.py
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

sql = "select * from book"
database = "baseball"
user = "user"
password = "523265"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

# Create a data frame by reading data from Oracle via JDBC
df = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql)
    .option("user", user)
    .option("password", password)
    .option("driver", jdbc_driver)
    .load()
)

df.show()
