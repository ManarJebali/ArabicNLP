import numpy as np

def padding_(sequences, seq_len):
    """
    Pad sequences to the same length with zeros at the beginning.

    Args:
        sequences (list of list of int): The input sequences.
        seq_len (int): The desired sequence length.

    Returns:
        np.ndarray: Padded sequences as a NumPy array.
    """
    features = np.zeros((len(sequences), seq_len), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            trunc = np.array(seq)[-seq_len:]
            features[i, -len(trunc):] = trunc
    return features

# Example usage:
# x_train_pad = padding_(final_list_train, seq_len=50)
# x_test_pad  = padding_(final_list_test, seq_len=50)