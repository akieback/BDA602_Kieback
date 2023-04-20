import random
from typing import List, Tuple

import pandas
import seaborn
from sklearn import datasets


class TestDatasets:
    def __init__(self):
        self.seaborn_data_sets = ["mpg", "tips", "titanic"]
        self.sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
        self.all_data_sets = self.seaborn_data_sets + self.sklearn_data_sets

    TITANIC_PREDICTORS = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "embarked",
        "parch",
        "fare",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
        "class",
    ]

    def get_all_available_datasets(self) -> List[str]:
        return self.all_data_sets

    def get_test_data_set(
        self, data_set_name: str = None
    ) -> Tuple[pandas.DataFrame, List[str], str]:
        """Function to load a few test data sets

        :param:
        data_set_name : string, optional
            Data set to load

        :return:
        data_set : :class:`pandas.DataFrame`
            Tabular data, possibly with some preprocessing applied.
        predictors :list[str]
            List of predictor variables
        response: str
            Response variable
        """

        if data_set_name is None:
            data_set_name = random.choice(self.all_data_sets)
        else:
            if data_set_name not in self.all_data_sets:
                raise Exception(f"Data set choice not valid: {data_set_name}")

        if data_set_name in self.seaborn_data_sets:
            if data_set_name == "mpg":
                data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
                predictors = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "origin",
                ]
                response = "mpg"
            elif data_set_name == "tips":
                data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
                predictors = [
                    "total_bill",
                    "sex",
                    "smoker",
                    "day",
                    "time",
                    "size",
                ]
                response = "tip"
            elif data_set_name in ["titanic", "titanic_2"]:
                data_set = seaborn.load_dataset(name="titanic").dropna()
                data_set["alone"] = data_set["alone"].astype(str)
                data_set["class"] = data_set["class"].astype(str)
                data_set["deck"] = data_set["deck"].astype(str)
                data_set["pclass"] = data_set["pclass"].astype(str)
                predictors = self.TITANIC_PREDICTORS
                if data_set_name == "titanic":
                    response = "survived"
                elif data_set_name == "titanic_2":
                    response = "alive"
        elif data_set_name in self.sklearn_data_sets:
            if data_set_name == "boston":
                data = datasets.load_boston()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
                data_set["CHAS"] = data_set["CHAS"].astype(str)
            elif data_set_name == "diabetes":
                data = datasets.load_diabetes()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            elif data_set_name == "breast_cancer":
                data = datasets.load_breast_cancer()
                data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["target"] = data.target
            predictors = data.feature_names
            response = "target"

        # Change category dtype to string
        for predictor in predictors:
            if data_set[predictor].dtype in ["category"]:
                data_set[predictor] = data_set[predictor].astype(str)

        print(f"Data set selected: {data_set_name}")
        data_set.reset_index(drop=True, inplace=True)
        return data_set, predictors, response


if __name__ == "__main__":
    test_datasets = TestDatasets()
    for test in test_datasets.get_all_available_datasets():
        df, predictors, response = test_datasets.get_test_data_set(data_set_name=test)
