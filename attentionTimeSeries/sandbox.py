"""
Showing how to use the model with some time series data.

NB! This is not a full training loop. You have to write the training loop yourself. 

I.e. this code is just a starting point to show you how to initialize the model and provide its inputs

If you do not know how to train a PyTorch model, it is too soon for you to dive into transformers imo :) 

You're better off starting off with some simpler architectures, e.g. a simple feed forward network, in order to learn the basics
"""

#import dataset as ds
import utils
from torch.utils.data import DataLoader
import torch
import datetime
#import transformer_timeseries as tst
import numpy as np
import dataset as ds
import attentionModel as atm

# Hyperparams
test_size = 0.1
batch_size = 128
target_col_name = "close"
timestamp_col = "dTime"
# Only use data from this date and onwards
cutoff_date = datetime.datetime(2017, 1, 1) 

## Params
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
dec_seq_len = 92 # length of input given to decoder
enc_seq_len = 153 # length of input given to encoder
output_sequence_length = 48 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len
batch_first = False

# Define input variables 
exogenous_vars = [] # should contain strings. Each string must correspond to a column name
input_variables = [target_col_name] + exogenous_vars
target_idx = 0 # index position of target in batched trg_y

input_size = len(input_variables)

def prepareInput():
    # Read data
    data = utils.read_data(timestamp_col_name=timestamp_col)

    # Remove test data from dataset
    training_data = data[:-(round(len(data)*test_size))]

    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
    # Should be training data indices only
    training_indices = utils.get_indices_entire_sequence(
        data=training_data, 
        window_size=window_size, 
        step_size=step_size)
    
    # Making instance of custom dataset class
    training_data = ds.TransformerDataset(
        data=torch.tensor(training_data[input_variables].values).float(),
        indices=training_indices,
        enc_seq_len=enc_seq_len,
        dec_seq_len=dec_seq_len,
        target_seq_len=output_sequence_length
        )
    # Making dataloader
    training_data = DataLoader(training_data, batch_size)

    i, batch = next(enumerate(training_data))

    src, trg, trg_y = batch

    # Make src mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, enc_seq_len]
    src_mask = utils.generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len
        )

    # Make tgt mask for decoder with size:
    # [batch_size*n_heads, output_sequence_length, output_sequence_length]
    tgt_mask = utils.generate_square_subsequent_mask( 
        dim1=output_sequence_length,
        dim2=output_sequence_length
        )
    
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 18714
    trg_vocab_size = 18714
    

    model = atm.Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx)

    out = model(src[0], trg[0][:, :-1])
    # output = model(
    #     src=src,
    #     tgt=trg,
    #     src_mask=src_mask,
    #     tgt_mask=tgt_mask
    #     )
    

if __name__ == "__main__":
    prepareInput()



