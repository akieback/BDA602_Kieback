#!/bin/bash
# The baseball.sql has to be in the same folder than this file

set -e

# Wait for MariaDB to start
until mysqladmin ping -h mariadb -u root -proot -P3306 --silent; do
    echo "Waiting for MariaDB to start..."
    sleep 1
done
#mysql -h mariadb -u root -proot -P3306 -e "CREATE Database IF NOT EXISTS baseball;"
echo "..."

# Check if database exists
#DATABASE_EXISTS=$(mysql -h mariadb -u root -proot -P3306 -e "SHOW DATABASES LIKE 'baseball';" | grep baseball)
#echo "$DATABASE_EXISTS"
# Check if database exists
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
mysql -h mariadb -u root -proot -P3306 -D baseball -e "SELECT * FROM final_Stats" > /results.txt
echo "txt export done"
echo "Container done"