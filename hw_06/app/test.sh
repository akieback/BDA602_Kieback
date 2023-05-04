#!/bin/bash
# The baseball.sql has to be in the same folder than this file

set -e

# Wait for MariaDB to start
until mysqladmin ping -h mariadb -u root -proot -P3306 --silent; do
    echo "Waiting for MariaDB to start..."
    sleep 1
done

# Check if database exists
DATABASE_EXISTS=$(mysql -h mariadb -u root -proot -P3306 -e "SHOW DATABASES LIKE 'baseball';" | grep baseball)

# Check if database exists
if [ "$DATABASE_EXISTS" == "baseball" ]; then
  echo "Database already exists"
else
  # Create the database
  mysql -h mariadb -u root -proot -P3306 -e "CREATE Database baseball;"
  echo "Database created successfully1"
  #create database from baseball.sql file
  mysql -h mariadb -u root -proot -P3306 -D baseball < /app/baseball.sql
  echo "Database created successfully"
fi
echo "tables inserted"

sleep 15
mysql -h mariadb -u root -proot -P3306 -D baseball < /app/hw_05.sql
echo "features created"
#export final table to results.txt
mysql -h mariadb -u root -proot -P3306 -Dbaseball -e "SELECT * FROM baseball.final_Stats;" > /results.txt

echo "Database created successfully"
