# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Final report.

This is my report on my final project in my Machine Learning course. First of all, what is the goal of our
code? We have written code over the semester that should allow you to insert any dataset (after
cleaning of course) and the program will automatically find out if the predictors and responses are
continuous or categorical. Then the python program will look at each predictor/feature individually and
calculate the respective p-value, t-score and random forest score. In addition, various plots are created,
such as a difference mean of response plot and a violin plot for the distribution within a feature.
The correlation between the individual predictors is then calculated and presented in a table and a matrix.
Additionally, all predictors are analyzed with each other by brute force and the results are presented in a
table and a matrix. After that, several ML models are created and the corresponding ROC curves, confusion matrices
and cross validations. Finally, everything is exported to an HTML document and saved in the results folder. \
\
So now lets start with our final. For our final project, we have been given a baseball dataset to use to test our code. Furthermore, we are supposed to do feature engineering. \
What did I start with? I started by creating a SQL file that I can use to access the database and do my own calculations. With the help of this sql file I was able to create a variety
of features and collect them neatly in a final table. important to mention here is that i always used a rolling average. this means that every line with a gameid contains the average
data from the last 100 days. After calculating my features in sql i connect to the running docker mariadb server with sqlalchemy to get access to the organized and final table with python to use the features in python. \
My first steps in my python program is to find out which path the python program is currently in to create a results folder there. Right after that an elo-ranking is calculated automatically. This has to happen before all
other steps, because this will be one of our features.

The elo ranking system works by assigning a rating to each team,
which should reflect their skill level. After each game, the ranking will be adjusted based on the pre-game ranking, the opponent's
ranking and whether it was a win or a loss.
The calculation of the elo ranking is as follows:

`new_rating = old_rating + K * (score - expected_score)`

where:

- old_rating is the player's or team's rating before the game
- new_rating is the player's or team's rating after the game
- K is a constant that determines the weight of the game in the rating update (i.e., the importance of the game in determining the player's or team's new rating)
- score is the actual score achieved in the game (e.g., number of points scored, or win/loss/draw)
- expected_score is the expected score for the player or team based on their rating and the rating of their opponent(s)

The expected score for a player or team is calculated using the following formula:

`expected_score = 1 / (1 + 10^((opp_rating - player_rating) / 400))`

where:

- opp_rating is the rating of the player's or team's opponent(s)
- player_rating is the player's or team's own rating

The good thing about this ranking is that it takes into account
the ranking of the opposing team. I have the ranking but also c
alculated in 2 different ways. First, I calculated the normal elo ranking
as described above. Then I created a second ranking, which also takes into account the last final score.
After I have created the elo rankings, I clean my data. I do this using the interquartile range (IQR) as
a principle. It is defined as the difference between the 75th percentile and the 25th percentile of a data set.

Now lets get to the feature engineering.
