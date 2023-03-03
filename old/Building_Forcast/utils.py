import pandas as pd


def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences. 

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences. 

    Args:
        num_obs (int): Number of observations (time steps) in the entire 
                       dataset for which indices must be generated, e.g. 
                       len(data)

        window_size (int): The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50 
                           time steps, window_size = 100+50 = 150

        step_size (int): Size of each step as the data sequence is traversed 
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size], 
                         and the next will be [1:window_size].

    Return:
        indices: a list of tuples
    """

    stop_position = len(data)-1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0

    subseq_last_idx = window_size

    indices = []

    while subseq_last_idx <= stop_position:

        indices.append((subseq_first_idx, subseq_last_idx))

        subseq_first_idx += step_size

        subseq_last_idx += step_size

    return indices
