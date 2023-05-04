FROM python:3.6.9

ENV APP_HOME /app
WORKDIR $APP_HOME

# Install necessary packages
RUN apt-get update && apt-get install -y \
    mariadb-client \
    && rm -rf /var/lib/apt/lists/*

# Copy SQL script to container
COPY baseball.sql .
COPY hw_05.sql .

# Set the entrypoint script
COPY test.sh /
RUN chmod +x /test.sh
ENTRYPOINT ["/test.sh"]
