import argparse
import BuildingDataset as bd
import BuildingDataReader as dr
import Preprocessing as pre
import TransformerDataset as tfd
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
# from helpers import *
# from inference import *

# Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92  # length of input given to decoder
enc_seq_len = 153  # length of input given to encoder
# target sequence length. If hourly data and length = 48, you predict 2 days ahead
output_sequence_length = 48
# used to slice data into sub-sequences
window_size = enc_seq_len + output_sequence_length
step_size = 1  # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False


def main(
    epoch=10,
    batch_size=1,
    training_length=48,
    forecast_window=24,
    dataset_name="Office_Eddie.csv",
    path_to_save_model="save_model/",
    device="cpu"
):
    dataReader = dr.BuildingDataReader("data/", dataset_name)
    training_data = dataReader.get_trainingData(0.1)
    print(training_data)

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
    # Should be training data indices only
    training_indices = get_indices_entire_sequence(
        data=training_data,
        window_size=window_size,
        step_size=step_size)

    # Making instance of custom dataset class
    training_data = tfd.TransformerDataset(
        data=torch.tensor(training_data["energyConsumption"].values).float(),
        indices=training_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=output_sequence_length
    )
    # Making dataloader
    training_data = DataLoader(training_data, batch_size)
    print(training_data)

    # pre.process_data("data/" + dataset_name)

    # clean_directory()
    # bd.BuidingDataset(csv_name = dataset, root_dir = "data/", training_length = training_length, forecast_window = forecast_window)
    # train_dataset, test_dataset = bd.BuidingDataset(csv_name = "pre_dataset.csv", root_dir = "data/", training_length = training_length, forecast_window = forecast_window)
    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # test_dataset = SensorDataset(csv_name = test_csv, root_dir = "Data/", training_length = training_length, forecast_window = forecast_window)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # best_model = transformer(train_dataloader, epoch, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device)

    # inference(path_to_save_predictions, forecast_window, test_dataloader, device, path_to_save_model, best_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--path_to_save_model",
                        type=str, default="save_model/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    main(
        epoch=args.epoch,
        batch_size=args.batch_size,
        path_to_save_model=args.path_to_save_model,
        device=args.device,
    )
