# -*- coding: utf-8 -*-
import os
import sys

import cat_correlation as cor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api

# from dataset_loader import TestDatasets
from mariadb_spark_transformer1 import get_data
from plotly import express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # inspired by https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def create_folder(name):  # create a folder for the output plots
    path = os.getcwd() + f"/{name}"  # get current path + "/Plots"
    if os.path.exists(path):  # if path exists do nothing
        pass
    else:  # if path does not exist create a new output
        os.mkdir(path)
        print_heading("Output folder got created")
    # inspired by my hw_01.py script
    return path


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

    continuous_vars = []
    categorical_vars = []
    for k, v in re_pr_type.items():
        if v == "boolean" or v == "categorical":
            categorical_vars.append(k)
        else:
            continuous_vars.append(k)
    # re_pr_type.re.remove(response)
    return re_pr_type, continuous_vars, categorical_vars


def create_plots(re_pr_type, df, predictors, response):
    return_list_plots = []
    for predictor in predictors:
        if re_pr_type[response] == "boolean" and re_pr_type[predictor] == "continuous":
            # inspired by my first assignment
            fig1 = px.violin(df, x=response, y=predictor, color=response)
            fig1.write_html(
                file=f"results/Plots/violin_bool_con_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(
                f"results/Plots/violin_bool_con_chart_{predictor}.html"
            )
        elif (
            re_pr_type[response] == "boolean" and re_pr_type[predictor] == "categorical"
        ):
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/4/2
            fig2 = px.density_heatmap(df, x=response, y=predictor)
            fig2.write_html(
                file=f"results/Plots/heat_bool_cat_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(
                f"results/Plots/heat_bool_cat_chart_{predictor}.html"
            )
        elif (
            re_pr_type[response] == "continuous"
            and re_pr_type[predictor] == "continuous"
        ):
            # trend-line inspired by code from lecture
            fig3 = px.scatter(df, x=response, y=predictor)
            fig3.write_html(
                file=f"results/Plots/scatter_con_con_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(
                f"results/Plots/scatter_con_con_chart_{predictor}.html"
            )
        elif (
            re_pr_type[response] == "continuous"
            and re_pr_type[predictor] == "categorical"
        ):
            # inspired lecture and first assignment
            fig4 = px.violin(df, x=response, y=predictor, color=response)
            fig4.write_html(
                file=f"results/Plots/violin_con_cat_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return_list_plots.append(
                f"results/Plots/violin_con_cat_chart_{predictor}.html"
            )
    return return_list_plots


def get_pt_value_score(df, predictors, response, re_pr_type):
    # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
    return_list_score = []
    create_folder("results/Plots/P_T_Value")
    for predictor in predictors:
        X = df[predictor]
        Y = df[response]

        # https://www.geeksforgeeks.org/logistic-regression-using-statsmodels/
        if (
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/10/2
            # Regression: Continuous response
            re_pr_type[response] == "continuous"
            and re_pr_type[predictor] == "continuous"
        ):
            # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
            # https://www.statology.org/statsmodels-linear-regression-p-value/
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
    path_dict = {}
    create_folder("results/Plots/diff_mean")
    count = len(
        df.index
    )  # store length of df in count variable to later calculate mean line
    amount = sum(df[response])  # store count of each class
    horizontal = amount / count  # calculate rate for horizontal line
    for predictor in predictors:
        if re_pr_type[predictor] == "continuous":
            if len(df[predictor].unique()) <= 10:
                bins_amount = len(df[predictor].unique())

            bins = np.percentile(df[predictor], [5, 95])

            # Create the histogram using the custom bin edges

            hist, bin_edges = np.histogram(
                df[predictor], bins=np.linspace(bins[0], bins[1], 10)
            )  # set number of bins
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # calculate bin center
            for p in range(0, len(bin_edges) - 1):
                bin_edges[p] -= 0.00000001  # so the lower bound is included
            bin_edges[-1] += 0.00000001
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
            mean_squared = pre_mean_squared * (1 / bins_amount)

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
                    x=[bin_centers[0], bin_centers[-1]],
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
                file=f"results/Plots/diff_mean/diff_mean_{predictor}_{response}.html",
                include_plotlyjs="cdn",
            )
            path_dict[
                predictor
            ] = f"results/Plots/diff_mean/diff_mean_{predictor}_{response}.html"
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
                file=f"results/Plots/diff_mean/diff_mean_{predictor}_{response}.html",
                include_plotlyjs="cdn",
            )
            return_list.append([mean_squared, weighted_mean])
            path_dict[
                predictor
            ] = f"results/Plots/diff_mean/diff_mean_{predictor}_{response}.html"

    return return_list, path_dict


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
        # inspired by lecture notes
        # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/9/1/7
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
        # https://www.freecodecamp.org/news/how-to-use-html-to-open-link-in-new-tab/
        link = f'<a href="{i}" target="_blank">link</a>'
        links.append(link)

    return links


def create_output_df_brute_force(
    predictor_list,
    paths,
    scores,
    col1,
    col2,
):
    # create new Dataframe for html output (advise by Sean)
    df_html_output = pd.DataFrame(
        columns=[
            col1,
            col2,
            "mean_squared",
            "weighted_mean",
            "Mean_Plots",
        ]
    )

    i = 0
    link1 = turn_path_to_link(paths)

    for predictor in predictor_list:
        # https://www.geeksforgeeks.org/how-to-add-one-row-in-an-existing-pandas-dataframe/
        df_html_output.loc[len(df_html_output.index)] = [
            predictor[0],
            predictor[1],
            scores[i][0],
            scores[i][1],
            link1[i],
        ]
        i += 1

    final = df_html_output.sort_values(by=["weighted_mean"], ascending=[False])

    return final

    return df_html_output


def create_output_df_corr(
    predictor_list,
    col1,
    col2,
    plots,
):
    """
    :param predictor_list: list of both predictors and the correlation
    :param col1: specify column name
    :param col2: specify column name
    :param plots: dict of plots and predictor
    :return: html string
    """
    # create new Dataframe for html output (advise by Sean)
    df_html_output = pd.DataFrame(
        columns=[col1, col2, "corr", "Predictor 1", "Predictor 2"]
    )

    i = 0

    for predictor in predictor_list:
        path1 = plots[predictor[0]]
        path2 = plots[predictor[1]]
        link1 = f'<a href="{path1}" target="_blank">link</a>'
        link2 = f'<a href="{path2}" target="_blank">link</a>'

        # https://www.geeksforgeeks.org/how-to-add-one-row-in-an-existing-pandas-dataframe/
        df_html_output.loc[len(df_html_output.index)] = [
            predictor[0],
            predictor[1],
            predictor[2],
            link1,
            link2,
        ]
        i += 1

    final = df_html_output.sort_values(by=["corr"], ascending=[False])

    return final


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
    link2 = turn_path_to_link(plots.values())
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


def get_html_string(df):
    output = df.to_html(
        render_links=True,
        escape=False,
    )
    return output


def output_all_to_html(html):
    f = open("results/html_output.html", "w")
    f.write(html)
    f.close()


def get_matrix_con_con(df, continuous_vars):
    # Calculate the correlation coefficients
    corr_pair = []
    corr_matrix = pd.DataFrame(columns=continuous_vars, index=continuous_vars)
    for var1 in continuous_vars:
        for var2 in continuous_vars:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
            corr, _ = stats.pearsonr(df[var1], df[var2])
            corr_matrix.loc[var1, var2] = corr
            corr_pair.append([var1, var2, corr])

    # Print the correlation matrix
    print_heading("Con Con matrix")
    print(corr_matrix)

    # Plot the correlation matrix
    fig9 = px.imshow(corr_matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu")
    path = "matrix_con_con.html"
    fig9.write_html(
        file=f"results/Plots/{path}",
        include_plotlyjs="cdn",
    )
    return path, corr_pair


def get_matrix_con_cat(df, continuous_vars, categorical_vars):
    # Calculate the correlation coefficients
    corr_pair = []
    corr_matrix = pd.DataFrame(columns=categorical_vars, index=continuous_vars)
    for var1 in categorical_vars:
        for var2 in continuous_vars:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
            corr = cor.cat_cont_correlation_ratio(df[var1], df[var2])
            corr_matrix.loc[var2, var1] = corr
            corr_pair.append([var1, var2, corr])

    # Print the correlation matrix
    print_heading("Con Con matrix")
    print(corr_matrix)

    # Plot the correlation matrix
    fig9 = px.imshow(corr_matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu")
    path = "matrix_con_cat.html"
    fig9.write_html(
        file=f"results/Plots/{path}",
        include_plotlyjs="cdn",
    )
    return path, corr_pair


def get_matrix_cat_cat(
    df,
    x,
    y,
):
    corr_pair = []
    # Calculate the correlation coefficients
    corr_matrix = pd.DataFrame(columns=x, index=y)
    for var1 in x:
        for var2 in y:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
            corr = cor.cat_correlation(df[var1], df[var2])
            corr_matrix.loc[var2, var1] = corr
            corr_pair.append([var1, var2, corr])

    # Print the correlation matrix
    print_heading("Con Con matrix")
    print(corr_matrix)

    # Plot the correlation matrix
    fig9 = px.imshow(corr_matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu")
    path = "matrix_cat_cat.html"
    fig9.write_html(
        file=f"results/Plots/{path}",
        include_plotlyjs="cdn",
    )
    return path, corr_pair


def brute_force_con(df, response, cont_list):
    return_list = []
    path_list = []
    predictor_list = []
    already_done = []
    for predictor1 in cont_list:
        for predictor2 in cont_list:
            if (
                predictor1 == predictor2
                or [predictor2, predictor1] in already_done
                or [predictor1, predictor2] in already_done
            ):
                pass
            else:
                bin_count = 10
                fare_hist, bin_edges_fare = np.histogram(df[predictor2], bins=bin_count)

                # Create 8 bins for the age feature
                age_hist, bin_edges_age = np.histogram(df[predictor1], bins=bin_count)

                for p in range(0, len(bin_edges_fare) - 1):
                    bin_edges_fare[p] -= 0.00000001  # so the lower bound is included
                bin_edges_fare[-1] += 0.00000001
                for p in range(0, len(bin_edges_age) - 1):
                    bin_edges_age[p] -= 0.00000001  # so the lower bound is included
                bin_edges_age[-1] += 0.00000001

                # Calculate the means for each combination of age bin and fare bin
                grouped = df.groupby(
                    [
                        pd.cut(df[predictor1], bins=bin_edges_age),
                        (pd.cut(df[predictor2], bins=bin_edges_fare)),
                    ]
                )

                means_grouped = grouped[response].mean(numeric_only=True)

                # from assignment 4
                create_folder("results/Plots/brute_force")
                count = len(
                    df.index
                )  # store length of df in count variable to later calculate mean line
                amount = sum(df[response])  # store count of each class
                horizontal = amount / count  # calculate rate for horizontal line

                # get weighted and unweighted mean
                # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/0/10
                pre_mean_squared = 0
                wmse = 0
                # formulas from lecture notes
                hist_combined = sum(age_hist) + sum(fare_hist)
                # https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/6/3/8
                for i in means_grouped:
                    # if value is na than skip that
                    # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
                    if pd.isna(i):
                        pass
                    else:
                        pre_mean_squared += (i - horizontal) ** 2
                mean_squared = pre_mean_squared * (1 / (bin_count * bin_count))

                # for loop because we need both hist
                weight_each_bin = [
                    (b + c) / hist_combined for b in age_hist for c in fare_hist
                ]

                # https://www.geeksforgeeks.org/python-iterate-multiple-lists-simultaneously/#
                no_nan = []
                for (a, b) in zip(weight_each_bin, list(means_grouped)):
                    if pd.isna(a) or pd.isna(b):
                        pass
                    else:
                        no_nan.append((a, b))

                for (weight, mean) in no_nan:
                    wmse += weight * ((mean - horizontal) ** 2)

                # mid has to be used because its float
                # https://stackoverflow.com/questions/61045348/given-a-list-of-x-number-of-floats-return-a-tuple-of-the-average-of-the-middle
                index1 = [index[0].mid for index in means_grouped.index]
                index2 = [index[1].mid for index in means_grouped.index]

                # https://stackoverflow.com/questions/22127771/transforming-a-list-to-a-pivot-table-with-python
                zipped = pd.DataFrame(
                    list(zip(index1, index2, list(means_grouped))),
                    columns=["pred1", "pred2", "mean"],
                )

                # Reshape the means into a matrix
                # https://www.geeksforgeeks.org/how-to-create-a-pivot-table-in-python-using-pandas/
                matrix = pd.pivot_table(
                    zipped, index="pred1", columns="pred2", values="mean"
                )

                # https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
                fig10 = px.imshow(
                    matrix,
                    color_continuous_scale="RdBu",
                    labels=dict(x=predictor2, y=predictor1, color="Mean"),
                    aspect="auto",
                )

                path = f"results/Plots/brute_force/matrix_brute_con_{predictor1}_{predictor2}.html"
                # Show the plot
                fig10.write_html(
                    file=path,
                    include_plotlyjs="cdn",
                )
                already_done.append([predictor2, predictor1])

                path_list.append(path)
                return_list.append([mean_squared, wmse])
                predictor_list.append([predictor1, predictor2])

    return return_list, path_list, predictor_list


def brute_force_cat(df, response, cat_list):
    return_list = []
    path_list = []
    predictor_list = []
    create_folder("results/Plots/brute_force_cat")
    count = len(
        df.index
    )  # store length of df in count variable to later calculate mean line
    amount = sum(df[response])  # store count of each class
    horizontal = amount / count  # calculate rate for horizontal line
    already_done = []
    for predictor1 in cat_list:
        for predictor2 in cat_list:
            if (
                predictor1 == predictor2
                or [predictor2, predictor1] in already_done
                or [predictor1, predictor2] in already_done
            ):
                pass
            else:
                bin_amount_pred1 = len(np.sort(df[predictor1].unique()))
                bin_amount_pred2 = len(np.sort(df[predictor2].unique()))

                grouped = df.groupby([df[predictor1], df[predictor2]])

                bin_mean = grouped[response].mean()
                bin_count = grouped[response].count()

                hist_combined = bin_amount_pred1 * bin_amount_pred2
                pre_mean_squared = 0
                for i in bin_mean:
                    # if value is na than skip that
                    # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
                    # if pd.isna(i):
                    # pass
                    # else:
                    pre_mean_squared += (i - horizontal) ** 2
                mean_squared = pre_mean_squared * (
                    1 / (bin_amount_pred1 * bin_amount_pred2)
                )

                hist_combined = sum(bin_count)
                weight_each_bin = [b / hist_combined for b in bin_count]

                wmse = 0
                for (weight, mean) in zip(weight_each_bin, list(bin_mean)):
                    wmse += weight * ((mean - horizontal) ** 2)

                index1 = [index[1] for index in bin_mean.index]
                index2 = [index[0] for index in bin_mean.index]

                # https://stackoverflow.com/questions/22127771/transforming-a-list-to-a-pivot-table-with-python
                zipped = pd.DataFrame(
                    list(zip(index1, index2, list(bin_mean))),
                    columns=["pred1", "pred2", "mean"],
                )

                # Reshape the means into a matrix
                # https://www.geeksforgeeks.org/how-to-create-a-pivot-table-in-python-using-pandas/
                matrix = pd.pivot_table(
                    zipped, index="pred1", columns="pred2", values="mean"
                )

                # https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
                fig12 = px.imshow(
                    matrix,
                    color_continuous_scale="RdBu",
                    labels=dict(x=predictor1, y=predictor2, color="Mean"),
                    aspect="auto",
                )

                path = f"results/Plots/brute_force_cat/matrix_brute_cat_{predictor1}_{predictor2}.html"
                # Show the plot
                fig12.write_html(
                    file=path,
                    include_plotlyjs="cdn",
                )
                already_done.append([predictor2, predictor1])

                path_list.append(path)
                return_list.append([mean_squared, wmse])
                predictor_list.append([predictor1, predictor2])

    return return_list, path_list, predictor_list


def brute_force_cat_cont(df, response, cat_list, cont_list):
    return_list = []
    path_list = []
    predictor_list = []
    create_folder("results/Plots/brute_force_cat_cont")
    count = len(
        df.index
    )  # store length of df in count variable to later calculate mean line
    amount = sum(df[response])  # store count of each class
    horizontal = amount / count  # calculate rate for horizontal line

    for predictor_cat in cat_list:
        for predictor_con in cont_list:
            bin_amount_cat = len(np.sort(df[predictor_cat].unique()))
            bin_amount_cont = 10
            bin_total = bin_amount_cat * bin_amount_cont

            hist, bin_edges = np.histogram(df[predictor_con], bins=bin_amount_cont)

            for p in range(0, len(bin_edges) - 1):
                bin_edges[p] -= 0.00000001  # so the lower bound is included
            bin_edges[-1] += 0.00000001

            grouped = df.groupby(
                [(df[predictor_cat]), pd.cut(df[predictor_con], bins=bin_edges)]
            )

            means_grouped = grouped[response].mean(numeric_only=True)
            count_grouped = grouped[response].count()

            pre_mean_squared = 0
            for i in means_grouped:
                # if value is na than skip that
                # https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
                if pd.isna(i):
                    pass
                else:
                    pre_mean_squared += (i - horizontal) ** 2
            mean_squared = pre_mean_squared * (1 / bin_total)

            hist_combined = sum(count_grouped) + sum(hist)

            weight_each_bin = [
                (b + c) / hist_combined for b in hist for c in count_grouped
            ]

            wmse = 0
            no_nan = []

            for (a, b) in zip(weight_each_bin, list(means_grouped)):
                if pd.isna(a) or pd.isna(b):
                    pass
                else:
                    no_nan.append((a, b))

            for (weight, mean) in no_nan:
                wmse += weight * ((mean - horizontal) ** 2)

            index1 = [index[1].mid for index in means_grouped.index]
            index2 = [index[0] for index in means_grouped.index]

            # https://stackoverflow.com/questions/22127771/transforming-a-list-to-a-pivot-table-with-python
            zipped = pd.DataFrame(
                list(zip(index1, index2, list(means_grouped))),
                columns=["pred1", "pred2", "mean"],
            )

            # Reshape the means into a matrix
            # https://www.geeksforgeeks.org/how-to-create-a-pivot-table-in-python-using-pandas/
            matrix = pd.pivot_table(
                zipped, index="pred1", columns="pred2", values="mean"
            )

            # https://plotly.com/python-api-reference/generated/plotly.express.imshow.html
            fig12 = px.imshow(
                matrix,
                color_continuous_scale="RdBu",
                labels=dict(x=predictor_cat, y=predictor_con, color="Mean"),
                aspect="auto",
            )

            # Show the plot
            path = f"results/Plots/brute_force_cat_cont/matrix_brute_con_cat_{predictor_cat}_{predictor_con}.html"
            fig12.write_html(
                file=path,
                include_plotlyjs="cdn",
            )

            path_list.append(path)
            return_list.append([mean_squared, wmse])
            predictor_list.append([predictor_cat, predictor_con])

    return return_list, path_list, predictor_list


def matrix_html(path):
    # https://cscircles.cemc.uwaterloo.ca/3-comments-literals/
    header = ["con_con", "con_cat", "cat_cat"]
    html = ""
    for i in range(len(path)):
        html += f"<center><h1>{header[i]}</h1></center>"
        html += f'<center><iframe src="{path[i]}" width="800"height="600" frameBorder="0"></iframe></center>'
    return html


def add_header(text):
    html = f"<center><h2>{text}</h2></center>"
    return html


def style(html):
    x = html.replace("<table", "<center><table")
    y = x.replace("</table", "</table></center>")
    y += "</body> </html>"
    return y


def build_models(df, predictors, response):
    # inspired by hw_01
    X_orig = df[predictors].values  # set X_origin values

    Y = df[response].values

    # Split https://www.sharpsightlabs.com/blog/scikit-train_test_split/
    x_train, x_test, y_train, y_test = train_test_split(X_orig, Y, test_size=0.20)

    # Random Forest
    print_heading("Random Forest With Pipeline")
    pipeline_rf = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline_rf.fit(x_train, y_train.ravel())

    probability_rf = pipeline_rf.predict_proba(x_test)
    prediction_rf = pipeline_rf.predict(x_test)
    print(f"Probability: {probability_rf}")
    print(f"Predictions: {prediction_rf}")
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    score_rf = pipeline_rf.score(x_test, y_test)
    print(f"Score: {score_rf}")

    mse = mean_squared_error(y_test, prediction_rf)
    r2score = r2_score(y_test, prediction_rf)
    print("mean_squared_error: " + str(mse))
    print("r2_score: " + str(r2score))

    # https://www.projectpro.io/recipes/plot-roc-curve-in-python
    y_score = pipeline_rf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # https://mlhive.com/2023/02/create-heatmap-and-confusion-matrix-using-plotly-in-python
    cm = confusion_matrix(y_test, prediction_rf)
    heatmap = go.Heatmap(z=cm, x=["0", "1"], y=["0", "1"], colorscale="Blues")

    # create the layout
    layout = go.Layout(title="Confusion Metrix")

    # create the figure
    fig = go.Figure(data=[heatmap], layout=layout)

    # show the figure
    fig.show()

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    fig = px.line(
        roc_df, x="fpr", y="tpr", hover_data=["thresholds"], title="ROC Curve"
    )
    # Add the diagonal reference line
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="gray", dash="dash"),
            )
        ]
    )
    fig.show()

    # Decision Tree
    print_heading("Decision Tree with Pipeline")
    pipeline_dt = Pipeline(
        [
            ("StandardScaler", StandardScaler()),
            ("DecisionTree", DecisionTreeClassifier(random_state=1234)),
        ]
    )

    pipeline_dt.fit(x_train, y_train.ravel())

    probability_dt = pipeline_dt.predict_proba(x_test)
    prediction_dt = pipeline_dt.predict(x_test)
    print(f"Probability: {probability_dt}")
    print(f"Predictions: {prediction_dt}")
    score_dt = pipeline_dt.score(x_test, y_test)
    print(f"Score: {score_dt}")
    mse = mean_squared_error(y_test, prediction_dt)
    r2score = r2_score(y_test, prediction_dt)
    print("mean_squared_error: " + str(mse))
    print("r2_score: " + str(r2score))

    y_score = pipeline_dt.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)

    # https://mlhive.com/2023/02/create-heatmap-and-confusion-matrix-using-plotly-in-python
    cm = confusion_matrix(y_test, prediction_dt)
    heatmap = go.Heatmap(z=cm, x=["0", "1"], y=["0", "1"], colorscale="Blues")

    # create the layout
    layout = go.Layout(title="Confusion Metrix")

    # create the figure
    fig = go.Figure(data=[heatmap], layout=layout)

    # show the figure
    fig.show()

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    fig = px.line(
        roc_df, x="fpr", y="tpr", hover_data=["thresholds"], title="ROC Curve"
    )
    # Add the diagonal reference line
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="gray", dash="dash"),
            )
        ]
    )
    fig.show()


def main():
    # https://www.w3schools.com/html/html_table_borders.asp
    # https://www.color-hex.com/popular-colors.php
    html = (
        "<!DOCTYPE html><html><head><title>Midterm</title>"
        "<style> "
        "table, th, td { border: 1px solid white;"
        "border-collapse: collapse;} th, td { "
        "background-color: #81d8d0;} </style> </head> <body>"
        "<center><h1>Midterm Adrian Kieback</h1></center>"
    )
    create_folder("results/")
    curr_path = create_folder("results/Plots/")

    df, predictors, response = get_data()
    re_pr_type, continuous_vars, categorical_vars = get_response_predictor_type(
        df, predictors, response
    )
    categorical_vars.remove(response)
    pathss = create_plots(re_pr_type, df, predictors, response)

    scores = get_pt_value_score(df, predictors, response, re_pr_type)
    means, plots = diff_mean_response(df, predictors, response, re_pr_type, 10)
    forest = rand_forest_ranking(df, predictors, response, re_pr_type)

    df_html_output = create_output_df(
        predictors, pathss, plots, re_pr_type, scores, means, forest
    )
    html += add_header("Feature scores")
    html += get_html_string(df_html_output)

    paths_matrix = []

    # corr tables

    path, corr_list = get_matrix_con_con(df, continuous_vars)
    paths_matrix.append(curr_path + path)
    df_html_output2 = create_output_df_corr(corr_list, "con", "con", plots)
    html += add_header("Continuous/ Continuous")
    html += get_html_string(df_html_output2)

    path, corr_list = get_matrix_con_cat(df, continuous_vars, categorical_vars)
    paths_matrix.append(curr_path + path)
    df_html_output2 = create_output_df_corr(corr_list, "cat", "con", plots)
    html += add_header("Categorical/ Continuous")
    html += get_html_string(df_html_output2)

    path, corr_list = get_matrix_cat_cat(df, categorical_vars, categorical_vars)
    paths_matrix.append(curr_path + path)
    df_html_output2 = create_output_df_corr(corr_list, "cat", "cat", plots)
    html += add_header("Categorical/ Categorical")
    html += get_html_string(df_html_output2)

    # Get matrix paths
    matrix_paths = matrix_html(paths_matrix)

    scores2, paths2, predictor_list = brute_force_con(df, response, continuous_vars)

    df_html_output2 = create_output_df_brute_force(
        predictor_list, paths2, scores2, "cont", "cont"
    )
    html += add_header("Brute Force Continuous/ Continuous")
    html += get_html_string(df_html_output2)

    scores2, paths2, predictor_list = brute_force_cat(df, response, categorical_vars)

    df_html_output2 = create_output_df_brute_force(
        predictor_list, paths2, scores2, "cat", "cat"
    )
    html += add_header("Brute Force Categorical/ Categorical")
    html += get_html_string(df_html_output2)

    scores2, paths2, predictor_list = brute_force_cat_cont(
        df, response, categorical_vars, continuous_vars
    )

    df_html_output2 = create_output_df_brute_force(
        predictor_list, paths2, scores2, "cat", "cont"
    )
    html += add_header("Brute Force Categorical/ Continuous")
    html += get_html_string(df_html_output2)

    build_models(df, predictors, response)

    # add graphs to html in iframe
    html += matrix_paths
    final_html = style(html)
    output_all_to_html(final_html)

    print_heading("Program finished successfully")


"""
    we split the data 80/20 to have enough data to test our model.
    Both models did very bad... Probably the features are not really good.
"""

if __name__ == "__main__":
    sys.exit(main())
