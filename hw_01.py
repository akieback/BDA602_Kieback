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


def create_folder():
    path = os.getcwd() + "/Plots"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        print("Output folder got created")


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
        "sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm",
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
    # print(data.describe())


def generate_plots(df):
    # inspired by the plotly website
    fig1 = px.scatter(
        df,
        x="sepal width in cm",
        y="sepal length in cm",
        color="class",
        size="petal length in cm",
        hover_data=["petal width in cm"],
        symbol="class",
    )
    fig1.write_html(file="Plots/scatter.html", include_plotlyjs="cdn")

    fig2 = px.scatter_matrix(
        df,
        dimensions=[
            "sepal width in cm",
            "sepal length in cm",
            "petal width in cm",
            "petal length in cm",
        ],
        color="class",
    )
    fig2.write_html(file="Plots/scatter_matrix.html", include_plotlyjs="cdn")

    fig3 = px.density_contour(
        df, x="sepal width in cm", y="sepal length in cm", color="class"
    )
    fig3.write_html(file="Plots/density_contour.html", include_plotlyjs="cdn")

    fig4 = px.histogram(
        df, x="sepal width in cm", y="sepal length in cm", color="class"
    )
    fig4.write_html(file="Plots/histogram.html", include_plotlyjs="cdn")

    for i in range(4):
        fig5 = px.violin(df, x="class", y=df.columns[i], color="class")
        fig5.write_html(file="Plots/name{}.html".format(i), include_plotlyjs="cdn")
    """
    fig2 = px.parallel_coordinates(df, color="class", labels={"class": "Species",
                      "sepal width in cm": "Sepal Width", "sepal length in cm": "Sepal Length",
                      "petal width in cm": "Petal Width", "petal length in cm": "Petal Length", },
                        color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=2)

    fig2.write_html(file="Plots/parallel.html", include_ggplotly="cdn")

    fig1 = go.Figure()
    for column in df.columns:
        fig1.add_trace(go.Violin(x="class",
                                 y=df["class"][df["class"] == column],
                                 name=column,
                                 box_visible=True,
                                 meanline_visible=True))

    fig1.write_html(file="Plots/try.html", include_ggplotly="cdn")
    """


def build_Models(df):
    # Model building
    print_heading("Model")

    X_orig = df[
        [
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
        ]
    ].values

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
            ),  # NearestNeighbors.__init__() got an unexpected keyword argument
            # 'random_state'
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


def newtry2(df):
    df["is_setosa"] = (df["class"] == "Iris-setosa").astype(int)
    df["is_versicolor"] = (df["class"] == "Iris-versicolor").astype(int)
    df["is_virginica"] = (df["class"] == "Iris-virginica").astype(int)

    # Calculate the rate of `is_setosa` for each bin
    for predictor in [
        "sepal length in cm",
        "sepal width in cm",
        "petal length in cm",
        "petal width in cm",
    ]:
        for response_boolean in ["is_setosa", "is_versicolor", "is_virginica"]:
            bins = 10
            hist, bin_edges = np.histogram(df[predictor], bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            binned_df = df.groupby(pd.cut(df[predictor], bins=bin_edges)).mean()
            binned_df["bin_center"] = bin_centers
            binned_df["bin_count"] = hist

            # print_heading("Binned")
            # print(binned_df.head())
            # Create the bar plot with a line chart for the rate of `is_setosa`
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=binned_df["bin_center"],
                    y=binned_df["bin_count"],
                    name="histogram",
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=binned_df["bin_center"],
                    y=binned_df[response_boolean],
                    name=response_boolean,
                    line=dict(color="red"),
                ),
                secondary_y=True,
            )
            fig.update_layout(
                title=response_boolean,
                xaxis_title=predictor,
                yaxis_title=f"Rate of {response_boolean}",
            )
            fig.update_yaxes(title_text="<b>Main</b> Y - axis ", secondary_y=False)
            fig.update_yaxes(title_text="<b>secondary</b> Y - axis ", secondary_y=True)
            fig.show()


def main():
    create_folder()
    print_heading("Start of program")
    df = load_data("iris.data")
    create_summary(df)
    generate_plots(df)
    build_Models(df)
    # seite2(df)
    # try_some(df)
    newtry2(df)


if __name__ == "__main__":
    sys.exit(main())
