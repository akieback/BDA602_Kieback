#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def create_folder():  # create a folder for the output
    path = os.getcwd() + "/Plots"  # get current path + "/Plots"
    if os.path.exists(path):  # if path exists do nothing
        pass
    else:  # if path does not exist create a new output
        os.mkdir(path)
        print_heading("Output folder got created")


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # inspired by https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def load_data(data):
    # load data into pandas dataframe
    df = pd.read_csv(data)
    # adding column name to make it easier
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]
    return df


def create_summary(data):
    # load dataframe into numpy array
    numerical = data.drop(
        ["class"], axis=1
    )  # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html

    nd_array = np.array(numerical)
    # print(nd_array)

    print_heading("Statistical summary of Numpy Array")
    # axis = 0 means along the column
    print("Mean Numpy Array: ", np.mean(nd_array, axis=0))
    print("Min Numpy Array: ", np.amin(nd_array, axis=0))
    print("Max Numpy Array: ", np.amax(nd_array, axis=0))

    # https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
    print("25th Quantile: ", np.quantile(nd_array, 0.25, axis=0))  # one quarter
    print("50th Quantile: ", np.quantile(nd_array, 0.50, axis=0))  # half
    print("75th Quantile: ", np.quantile(nd_array, 0.75, axis=0))  # three quarter
    # get summary of dataframe with pandas


def generate_plots(df):
    # inspired by the plotly website
    fig1 = px.scatter(  # create scatter plot
        df,
        x="sepal_width",
        y="sepal_length",
        color="class",
        size="petal_length",
        hover_data=["petal_width"],
        symbol="class",
    )
    fig1.write_html(file="Plots/scatter.html", include_plotlyjs="cdn")  # save plot

    fig2 = px.scatter_matrix(  # create scatter matrix plot
        df,
        dimensions=[
            "sepal_width",
            "sepal_length",
            "petal_width",
            "petal_length",
        ],
        color="class",
    )
    fig2.write_html(file="Plots/scatter_matrix.html", include_plotlyjs="cdn")

    fig3 = px.density_contour(  # create density contour plot
        df, x="sepal_width", y="sepal_length", color="class"
    )
    fig3.write_html(file="Plots/density_contour.html", include_plotlyjs="cdn")

    fig4 = px.histogram(  # create histogram plot
        df, x="sepal_width", y="sepal_length", color="class"
    )
    fig4.write_html(file="Plots/histogram.html", include_plotlyjs="cdn")

    for i in range(4):  # create 4 violin plots
        fig5 = px.violin(df, x="class", y=df.columns[i], color="class")
        fig5.write_html(
            file="Plots/violin_chart_{}.html".format(df.columns[i]),
            include_plotlyjs="cdn",
        )


def build_Models(df):
    # Model building
    print_heading("Model")

    X_orig = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ].values  # set X_origin values

    y = df[["class"]].values

    # creating scaler object
    s_scaler = StandardScaler()
    scale = s_scaler.fit_transform(X_orig)

    random_forest = RandomForestClassifier(random_state=1234)
    # A column-vector y was passed when a 1d array was expected.
    # Please change the shape of y to (n_samples,), for example using ravel().
    random_forest.fit(scale, y.ravel())
    prediction = random_forest.predict(scale)
    probability = random_forest.predict_proba(scale)

    print_heading("Random Forest Without Pipeline")
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")

    print_heading("Random Forest With Pipeline")
    pipeline_rf = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline_rf.fit(X_orig, y.ravel())

    probability2 = pipeline_rf.predict_proba(X_orig)
    prediction2 = pipeline_rf.predict(X_orig)
    print(f"Probability: {probability2}")
    print(f"Predictions: {prediction2}")

    # Nearest neighbor
    print_heading("Nearest neighbor with Pipeline")
    pipeline_nn = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            (
                "NearestNeighbor",
                KNeighborsClassifier(),
            ),
        ]
    )

    pipeline_nn.fit(X_orig, y.ravel())

    probability_nn = pipeline_nn.predict_proba(X_orig)
    prediction_nn = pipeline_nn.predict(X_orig)
    print(f"Probability: {probability_nn}")
    print(f"Predictions: {prediction_nn}")

    # Decision Tree
    print_heading("Decision Tree with Pipeline")
    pipeline_dt = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
        ]
    )

    pipeline_dt.fit(X_orig, y.ravel())

    probability_dt = pipeline_dt.predict_proba(X_orig)
    prediction_dt = pipeline_dt.predict(X_orig)
    print(f"Probability: {probability_dt}")
    print(f"Predictions: {prediction_dt}")


def mean_of_difference(df):
    df["is_setosa"] = (df["class"] == "Iris-setosa").astype(int)  # set boolean values
    df["is_versicolor"] = (df["class"] == "Iris-versicolor").astype(int)
    df["is_virginica"] = (df["class"] == "Iris-virginica").astype(int)

    count = len(
        df.index
    )  # store length of df in count variable to later calculate mean line
    # Calculate the rate of response boolean for each bin
    # using for loop to only have the code once and easy to change
    for predictor in [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]:
        for response_boolean in ["is_setosa", "is_versicolor", "is_virginica"]:
            amount = df[df[response_boolean] == 1].shape[0]  # store count of each class
            horizontal = amount / count  # calculate rate for horizontal line
            hist, bin_edges = np.histogram(
                df[predictor], bins=10  # set number of bins
            )  # use numpy because its easier
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # calculate bin center
            binned_df = df.groupby(pd.cut(df[predictor], bins=bin_edges)).mean(
                numeric_only=True
            )  # calculate mean
            # The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version,
            # numeric_only will default to False. Thats why we specify it as True
            binned_df["bin_center"] = bin_centers
            binned_df["bin_count"] = hist

            # print_heading("Binned")
            # print(binned_df.head())
            # Create the bar plot with a line chart for the rate of each boolean response
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=binned_df["bin_center"],
                    y=binned_df["bin_count"],
                    name="Count of Iris",
                ),
                secondary_y=False,  # set one of 2 y-axis
            )
            fig.add_trace(
                go.Scatter(
                    x=binned_df["bin_center"],
                    y=binned_df[response_boolean],
                    name="µi - µpop",
                    line=dict(color="red"),
                    connectgaps=True,
                ),
                secondary_y=True,  # set 2nd of 2 y-axis
            )

            fig.add_trace(
                go.Scatter(
                    x=[min(df[predictor]), max(df[predictor])],
                    y=[horizontal, horizontal],
                    name="µpop",
                ),
                secondary_y=True,
            )

            fig.update_layout(
                title=response_boolean,
                xaxis_title=predictor,
            )
            fig.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
            fig.update_yaxes(title_text="<b>Response</b>", secondary_y=True)
            fig.write_html(
                file=f"Plots/diff_mean_{predictor}_{response_boolean}.html",
                include_plotlyjs="cdn",
            )


def main():
    create_folder()
    print_heading("Start of program")
    df = load_data("iris.data")
    create_summary(df)
    generate_plots(df)
    build_Models(df)
    mean_of_difference(df)


if __name__ == "__main__":
    sys.exit(main())
