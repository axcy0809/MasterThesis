import pandas as pd
import os


class BuildingDataReader():
    """Face Landmarks dataset."""

    def __init__(self, csv_path, csv_name):
        """
        Args:
            csv_path (string): Path to the csv file.
            csv_name (string): name of dataset
        """
        self.csv_path = csv_path
        self.csv_name = csv_name

    def read_csv(self) -> pd.DataFrame:
        dataset = pd.read_csv(self.csv_path + self.csv_name)
        dataset = dataset.drop("Unnamed: 0", axis=1)
        dataset = self.dropNan(dataset)
        dataset = self.timeFormat(dataset)

        return dataset

    def dropNan(self, dataset) -> pd.DataFrame:
        dataset = dataset.dropna()
        return dataset

    def timeFormat(self, dataset) -> pd.DataFrame:
        dataset["Date"] = [ts[:-6] for ts in dataset["Date"]]
        return dataset

    def get_trainingData(self, percent) -> pd.DataFrame:
        dataset = self.read_csv()
        training_data = dataset[:-(round(len(dataset)*percent))]
        return training_data
