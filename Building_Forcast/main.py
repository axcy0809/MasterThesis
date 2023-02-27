import argparse
import BuildingDataset as bd
import Preprocessing as pre
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
#from helpers import *
#from inference import *

def main(
    epoch = 10,
    batch_size = 1,
    training_length = 48,
    forecast_window = 24,
    dataset_name = "Office_Eddie.csv",
    path_to_save_model = "save_model/",
    device = "cpu"
):
    pre.process_data("data/" + dataset_name)

    #clean_directory()
    #bd.BuidingDataset(csv_name = dataset, root_dir = "data/", training_length = training_length, forecast_window = forecast_window)
    train_dataset, test_dataset = bd.BuidingDataset(csv_name = "pre_dataset.csv", root_dir = "data/", training_length = training_length, forecast_window = forecast_window)
    #train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    #test_dataset = SensorDataset(csv_name = test_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    #test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #best_model = transformer(train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)

    #inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--path_to_save_model",type=str,default="save_model/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        batch_size=args.batch_size,
        path_to_save_model=args.path_to_save_model,
        device=args.device,
    )