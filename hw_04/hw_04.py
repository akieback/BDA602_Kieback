#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd
from dataset_loader import TestDatasets
from plotly import express as px


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def create_folder():  # create a folder for the output
    path = os.getcwd() + "/Plots"  # get current path + "/Plots"
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
    re_pr_type = {}  # inspired by Anna who explaine me the concept to use a dict here
    # https://stackoverflow.com/questions/15019830/check-if-object-is-a-number-or-boolean
    if df[response][0] is bool:
        re_pr_type[response] = "boolean"
    else:
        re_pr_type[response] = "continuous"
    # determine if predictor is categorical or continuous
    for predictor in predictors:
        # https://stackoverflow.com/questions/4843173/how-to-check-if-type-of-a-variable-is-string
        if isinstance(df[predictor][0], str) or df[predictor] is bool:
            re_pr_type[predictor] = "categorical"
        else:
            re_pr_type[predictor] = "continuous"

    return re_pr_type


def create_plots(df_data_types, df, predictors, response):
    for predictor in predictors:

        if (
            df_data_types[response] == "boolean"
            and df_data_types[predictor] == "continuous"
        ):
            # inspired by my first assignment
            fig1 = px.violin(df, x=response, y=predictor, color=response)
            fig1.write_html(
                file=f"Plots/violin_bool_con_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
        elif (
            df_data_types[response] == "boolean"
            and df_data_types[predictor] == "categorical"
        ):
            fig2 = px.density_heatmap(df, x=response, y=predictor, color=response)
            fig2.write_html(
                file=f"Plots/heat_bool_cat_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
        elif (
            df_data_types[response] == "continuous"
            and df_data_types[predictor] == "continuous"
        ):
            # trendline inspired by code from lecture
            fig2 = px.scatter(df, x=response, y=predictor)
            fig2.write_html(
                file=f"Plots/scatter_con_con_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )
        elif (
            df_data_types[response] == "continuous"
            and df_data_types[predictor] == "categorical"
        ):
            # inspired lecture
            fig1 = px.violin(df, x=response, y=predictor, color=response)
            fig1.write_html(
                file=f"Plots/violin_con_cat_chart_{predictor}.html",
                include_plotlyjs="cdn",
            )


def main():
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    create_folder()
    df, predictors, repsonse = get_dataset("breast_cancer")
    re_pr_type = get_response_predictor_type(df, predictors, repsonse)
    create_plots(re_pr_type, df, predictors, repsonse)

    pass


if __name__ == "__main__":
    sys.exit(main())
