#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api
from dataset_loader import TestDatasets
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # inspired by https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def create_folder(name):  # create a folder for the output
    path = os.getcwd() + f"/{name}"  # get current path + "/Plots"
    if os.path.exists(path):  # if path exists do nothing
        pass
    else:  # if path does not exist create a new output
        os.mkdir(path)
        print_heading("Output folder got created")
    # inspired by my hw_01.py script


def get_dataset(name):
    """
    continuous response test_sets : ["mpg", "tips", "diabetes", "breast_cancer"]
    bool response test_sets : ["titanic", "breast_cancer"]
    """

    test_datasets = TestDatasets()
    df, predictors, response = test_datasets.get_test_data_set(data_set_name=name)
    df.dropna()

    return df, predictors, response


def get_response_predictor_type(
    df, predictors, response
):  # determine if response is boolean or continuous
    re_pr_type = {}  # inspired by Anna who explained me the concept to use a dict here
    # https://stackoverflow.com/questions/42449594/python-pandas-get-unique-count-of-column
    if len(df[response].unique()) == 2:  # inspired by office hours with Sean
        re_pr_type[response] = "boolean"
    else:
        re_pr_type[response] = "continuous"
    # determine if predictor is categorical or continuous
    for predictor in predictors:
        # https://www.geeksforgeeks.org/python-check-if-a-variable-is-string/
        if isinstance(df[predictor][0], str) or len(df[predictor].unique()) == 2:
            re_pr_type[predictor] = "categorical"
        else:
            re_pr_type[predictor] = "continuous"
    return re_pr_type


def create_plots(re_pr_type, df, predictors, response):
    return_list_plots = []
    for predictor in predictors:
        if re_pr_type[response] == "boolean" and re_pr_type[predictor] == "continuous":
            # inspired by my first assignment
            fig1 = px.violin(df, x=response, y=predictor, color=response)
            fig1.write_html(
                file=f"Plots/violin_bool_con_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(f"Plots/violin_bool_con_chart_{predictor}.html")
        elif (
            re_pr_type[response] == "boolean" and re_pr_type[predictor] == "categorical"
        ):
            fig2 = px.density_heatmap(df, x=response, y=predictor)
            fig2.write_html(
                file=f"Plots/heat_bool_cat_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(f"Plots/heat_bool_cat_chart_{predictor}.html")
        elif (
            re_pr_type[response] == "continuous"
            and re_pr_type[predictor] == "continuous"
        ):
            # trend-line inspired by code from lecture
            fig3 = px.scatter(df, x=response, y=predictor)
            fig3.write_html(
                file=f"Plots/scatter_con_con_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(f"Plots/scatter_con_con_chart_{predictor}.html")
        elif (
            re_pr_type[response] == "continuous"
            and re_pr_type[predictor] == "categorical"
        ):
            # inspired lecture
            fig4 = px.violin(df, x=response, y=predictor, color=response)
            fig4.write_html(
                file=f"Plots/violin_con_cat_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(f"Plots/violin_con_cat_chart_{predictor}.html")
    return return_list_plots


def get_pt_value_score(df, predictors, response, re_pr_type):
    # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
    return_list_score = []
    create_folder("Plots/P_T_Value")
    # else null so table is not fucked up
    for predictor in predictors:
        X = df[predictor]
        Y = df[response]

        if (
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/10/2
            # Regression: Continuous response
            re_pr_type[response] == "continuous"
            and re_pr_type[predictor] == "continuous"
        ):
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
            predictor_lin = statsmodels.api.add_constant(X)
            linear_regression_model = statsmodels.api.OLS(Y, predictor_lin).fit()
            print(f"Variable: {predictor}")
            print(linear_regression_model.summary())

            # Get the stats
            t_value = round(linear_regression_model.tvalues[1], 6)
            p_value = "{:.6e}".format(linear_regression_model.pvalues[1])

            return_list_score.append([t_value, p_value])
        elif (
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/10/2
            # Logistic Regression: Boolean response
            re_pr_type[response] == "boolean"
            and re_pr_type[predictor] == "continuous"
        ):
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
            predictor_log = statsmodels.api.add_constant(X)
            logistic_regression_model = statsmodels.api.Logit(Y, predictor_log)
            logistic_regression_model_fitted = logistic_regression_model.fit()
            print(f"Variable: {predictor}")
            print(logistic_regression_model_fitted.summary())
            # https://www.statsmodels.org/stable/discretemod.html

            t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
            p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

            return_list_score.append([t_value, p_value])
        else:
            return_list_score.append([0, 0])

    return return_list_score


def diff_mean_response(df, predictors, response, re_pr_type, bins_amount):
    return_list = []
    path_list = []
    create_folder("Plots/diff_mean")
    count = len(
        df.index
    )  # store length of df in count variable to later calculate mean line
    amount = df[df[response] == 1].shape[0]  # store count of each class
    horizontal = amount / count  # calculate rate for horizontal line
    for predictor in predictors:
        if re_pr_type[predictor] == "continuous":
            hist, bin_edges = np.histogram(
                df[predictor], bins=bins_amount
            )  # set number of bins
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # calculate bin center
            binned_df = df.groupby(pd.cut(df[predictor], bins=bin_edges)).mean(
                numeric_only=True
            )  # calculate mean

            binned_df["bin_center"] = bin_centers
            binned_df["bin_count"] = hist

            # get weighted and unweighted mean
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/0/10
            pre_mean_squared = 0
            y = 0
            weighted_mean = 0
            # formulas from lecture notes
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/3/8
            for i in binned_df[response]:
                # if value is na than skip that
                # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
                if pd.isna(i):
                    pass
                else:
                    pre_mean_squared += (i - horizontal) ** 2
                    weight = hist[y] / binned_df["bin_count"].sum()
                    weighted_mean += weight * ((i - horizontal) ** 2)
                    y += 1
            mean_squared = pre_mean_squared * 0.1

            # https://www.geeksforgeeks.org/how-to-implement-weighted-mean-square-error-in-python/
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/3/8

            fig7 = make_subplots(specs=[[{"secondary_y": True}]])
            fig7.add_trace(
                go.Bar(
                    x=binned_df["bin_center"],
                    y=binned_df["bin_count"],
                    name="Count",
                ),
                secondary_y=False,  # set one of 2 y-axis
            )
            fig7.add_trace(
                go.Scatter(
                    x=binned_df["bin_center"],
                    y=binned_df[response],
                    name="µi - µpop",
                    line=dict(color="red"),
                    connectgaps=True,
                ),
                secondary_y=True,  # set 2nd of 2 y-axis
            )

            fig7.add_trace(
                go.Scatter(
                    x=[min(df[predictor]), max(df[predictor])],
                    y=[horizontal, horizontal],
                    name="µpop",
                ),
                secondary_y=True,
            )

            fig7.update_layout(
                title=response,
                xaxis_title=predictor,
            )
            fig7.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
            fig7.update_yaxes(title_text="<b>Response</b>", secondary_y=True)
            fig7.write_html(
                file=f"Plots/diff_mean/diff_mean_{predictor}_{response}.html",
                include_plotlyjs="cdn",
            )
            path_list.append(f"Plots/diff_mean/diff_mean_{predictor}_{response}.html")
            return_list.append([mean_squared, weighted_mean])
        else:
            # if its categorical each class is its own bin
            # https://www.sharpsightlabs.com/blog/pandas-value_counts/
            bin_values = np.sort(df[predictor].unique())

            # Get the mean survival rate for each class
            mean_response = []
            bin_count = []
            for c in bin_values:
                mean_response.append(df[df[predictor] == c][response].mean())
                bin_count.append(df[df[predictor] == c][predictor].count())

            # get weighted and unweighted mean
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/0/10
            pre_mean_squared = 0
            weighted_mean = 0
            for i in range(len(bin_values)):
                pre_mean_squared += (mean_response[i] - horizontal) ** 2
                weight = bin_count[i] / sum(bin_count)
                weighted_mean += weight * ((mean_response[i] - horizontal) ** 2)
            mean_squared = pre_mean_squared * (1 / bins_amount)

            bin_counts = df[predictor].value_counts()

            fig8 = make_subplots(specs=[[{"secondary_y": True}]])
            fig8.add_trace(
                go.Bar(
                    x=bin_values,
                    y=bin_counts,
                    name="Count",
                ),
                secondary_y=False,  # set one of 2 y-axis
            )
            fig8.add_trace(
                go.Scatter(
                    x=bin_values,
                    y=mean_response,
                    name="µi - µpop",
                    line=dict(color="red"),
                    connectgaps=True,
                ),
                secondary_y=True,  # set 2nd of 2 y-axis
            )

            fig8.add_trace(
                go.Scatter(
                    x=[min(df[predictor]), max(df[predictor])],
                    y=[horizontal, horizontal],
                    name="µpop",
                ),
                secondary_y=True,
            )

            fig8.update_layout(
                title=response,
                xaxis_title=predictor,
            )
            fig8.update_yaxes(title_text="<b>Count</b>", secondary_y=False)
            fig8.update_yaxes(title_text="<b>Response</b>", secondary_y=True)
            fig8.write_html(
                file=f"Plots/diff_mean/diff_mean_{predictor}_{response}.html",
                include_plotlyjs="cdn",
            )
            return_list.append([mean_squared, weighted_mean])
            path_list.append(f"Plots/diff_mean/diff_mean_{predictor}_{response}.html")
    return return_list, path_list


def rand_forest_ranking(df, predictors, response, re_pr_type):
    return_list_forest = []
    print_heading("Random Forest")
    continuous_predictors = []

    for predictor in predictors:
        # https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical
        if df[predictor].dtype != "object":
            continuous_predictors.append(predictor)

    X = df[continuous_predictors].values
    Y = df[response].values

    # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/10/2
    # Regression: Continuous response
    if re_pr_type[response] == "continuous":
        # Fit RandomForestRegressor
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        rf_model = RandomForestRegressor(random_state=1234)
        rf_model.fit(X, Y.ravel())

        # Get feature importances
        rf_importances = rf_model.feature_importances_

        return_list_forest.append(rf_importances)

    elif (
        # inspired by first assignment
        # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/10/2
        # Logistic Regression: Boolean response
        re_pr_type[response]
        == "boolean"
    ):
        # Fit RandomForestClassifier
        s_scaler = StandardScaler()
        scale = s_scaler.fit_transform(X)
        rf_model = RandomForestClassifier(random_state=1234)
        rf_model.fit(scale, Y.ravel())

        # Get feature importances
        rf_importances = rf_model.feature_importances_

        return_list_forest.append(rf_importances)
    else:
        return_list_forest.append([0, 0])

    final_forest = []
    z = 0
    for predictor in predictors:
        if (
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/10/2
            # Regression: Continuous response
            re_pr_type[predictor]
            == "continuous"
        ):
            final_forest.append(return_list_forest[0][z])
            z += 1
        else:
            final_forest.append("NaN")

    return final_forest


def turn_path_to_link(paths):
    links = []
    for i in paths:
        link = f'<a href="{i}">link</a>'
        links.append(link)

    return links


def create_output_df(predictors, paths, plots, re_pr_type, scores, means, forest):

    """
    :param predictors: list of all predictors
    :param paths: list of initial plot paths
    :param plots: list of Diff of mean plot paths
    :param re_pr_type: dict of predictor types
    :param scores: p-value and t-score
    :param means: un- and weighted means
    :param forest: list of Randomforest
    :return: a DataFrame to export to html
    """

    # create new Dataframe for html output (advise by Sean)
    df_html_output = pd.DataFrame(
        columns=[
            "Predictor",
            "Type",
            "Plot",
            "p_value",
            "t_value",
            "mean_squared",
            "weighted_mean",
            "Mean_Plots",
            "RandomForest",
        ]
    )

    i = 0
    link1 = turn_path_to_link(paths)
    link2 = turn_path_to_link(plots)
    for predictor in predictors:
        # https://www.geeksforgeeks.org/how-to-add-one-row-in-an-existing-pandas-dataframe/
        df_html_output.loc[len(df_html_output.index)] = [
            predictors[i],
            re_pr_type[predictor],
            link1[i],
            scores[i][1],
            scores[i][0],
            means[i][0],
            means[i][1],
            link2[i],
            forest[i],
        ]
        i += 1

    return df_html_output


def output_all_to_html(df):
    f = open("html_output.html", "w")
    output = df.to_html(
        render_links=True,
        escape=False,
    )
    f.write(output)
    f.close()


def main():
    create_folder("Plots")
    df, predictors, response = get_dataset("titanic")
    re_pr_type = get_response_predictor_type(df, predictors, response)
    paths = create_plots(re_pr_type, df, predictors, response)
    # p-value & t-score (continuous predictors only) along with it`s plot
    scores = get_pt_value_score(df, predictors, response, re_pr_type)
    means, plots = diff_mean_response(df, predictors, response, re_pr_type, 10)
    forest = rand_forest_ranking(df, predictors, response, re_pr_type)

    df_html_output = create_output_df(
        predictors, paths, plots, re_pr_type, scores, means, forest
    )

    output_all_to_html(df_html_output)


if __name__ == "__main__":
    sys.exit(main())
