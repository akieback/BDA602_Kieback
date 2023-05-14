#!/bin/bash
# The baseball.sql has to be in the same folder than this file

set -e

# https://stackoverflow.com/questions/25503412/how-do-i-know-when-my-docker-mysql-container-is-up-and-mysql-is-ready-for-taking
# Wait for MariaDB to start
until mysqladmin ping -h mariadb -u root -proot -P3306 --silent; do
    echo "Waiting for MariaDB to start..."
    sleep 7
done
#mysql -h mariadb -u root -proot -P3306 -e "CREATE Database IF NOT EXISTS baseball;"
echo "..."

# Check if database exists
#DATABASE_EXISTS=$(mysql -h mariadb -u root -proot -P3306 -e "SHOW DATABASES LIKE 'baseball';" | grep baseball)
#echo "$DATABASE_EXISTS"
# Check if database exists
# https://stackoverflow.com/questions/838978/how-to-check-if-mysql-database-exists
if [[ ! -z "`mysql -h mariadb -u root -proot -P3306 -e "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME='baseball'" 2>&1`" ]]; then
  echo "Database already exists"
else
  # Create the database
  mysql -h mariadb -u root -proot -P3306 -e "CREATE Database  IF NOT EXISTS baseball;"
  echo "Database created successfully1"
  #create database from baseball.sql file
  mysql -h mariadb -u root -proot -P3306 -D baseball < /app/baseball.sql
  echo "tables inserted"
fi
echo "check completed"

mysql -h mariadb -u root -proot -P3306 -D baseball < /app/hw_05.sql
echo "features created"
#export final table to results.txt
mysql -h mariadb -u root -proot -P3306 -D baseball -e "SELECT * FROM final_Stats" > /app/results.txt
echo "txt export done"

java -version

python3 hw_05_python.py
echo "Container done"
