FROM python:3.6.9

ENV APP_HOME /app
WORKDIR $APP_HOME

#inspired by lecture notes
# Install necessary packages
RUN apt-get update && apt-get install -y \
    mariadb-client \
    && rm -rf /var/lib/apt/lists/*

# Copy SQL script to container
COPY baseball.sql .
COPY hw_05.sql .

# Get necessary python libraries
COPY requirements.txt .
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Set the entrypoint script
COPY test.sh /
# x for execution rights
RUN chmod +x /test.sh
ENTRYPOINT ["/test.sh"]
