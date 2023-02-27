import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
from icecream import ic

class BuidingDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_name, root_dir, training_length, forecast_window):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        
        # load raw data file
        csv_file = os.path.join(root_dir, csv_name)
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = MinMaxScaler()
        self.T = training_length
        self.S = forecast_window

        print("BuidingDataset: ",self.df)

    def __len__(self):
        # return number of sensors
        return len(self.df.groupby(by=["Unnamed: 0"]))

    # Will pull an index between 0 and __len__. 
    def __getitem__(self, idx):
        print(idx)
        
        # Sensors are indexed from 1
        idx = idx+1

        # np.random.seed(0)

        start = np.random.randint(0, len(self.df[self.df["Unnamed: 0"]==idx]) - self.T - self.S) 
        print(start)
        #sensor_number = str(self.df[self.df["Unnamed: 0"]==idx][["Unnamed: 0"]][start:start+1].values.item())
        #index_in = torch.tensor([i for i in range(start, start+self.T)])
        #index_tar = torch.tensor([i for i in range(start + self.T, start + self.T + self.S)])
        _input = torch.tensor(self.df["Humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"])
        target = torch.tensor(self.df["Humidity", "sin_hour", "cos_hour", "sin_day", "cos_day", "sin_month", "cos_month"])

        # scalar is fit only to the input, to avoid the scaled values "leaking" information about the target range.
        # scalar is fit only for humidity, as the timestamps are already scaled
        # scalar input/output of shape: [n_samples, n_features].
        scaler = self.transform

        scaler.fit(_input[:,0].unsqueeze(-1))
        _input[:,0] = torch.tensor(scaler.transform(_input[:,0].unsqueeze(-1)).squeeze(-1))
        target[:,0] = torch.tensor(scaler.transform(target[:,0].unsqueeze(-1)).squeeze(-1))

        # save the scalar to be used later when inverse translating the data for plotting.
        dump(scaler, 'scalar_item.joblib')

        return  _input, target