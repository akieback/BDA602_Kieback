FROM ubuntu:latest

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install necessary packages
RUN apt-get update && \
    apt-get install -y mariadb-client libmariadb-dev python3.10 python3-pip openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/* \

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
# Copy SQL script to container
COPY baseball.sql .
COPY hw_05.sql .

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

COPY cat_correlation.py .
COPY dataset_loader.py .
COPY hw_05_python.py .
COPY mariadb_spark_transformer1.py .
COPY mariadb-java-client-3.1.2.jar .
# Set the entrypoint script
COPY test.sh /
RUN chmod +x /test.sh
CMD ["/test.sh"]