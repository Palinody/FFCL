import os
import numpy as np

import numpy as np

def decode_txt(filepath, dtype=np.float32):
    data = np.loadtxt(filepath, dtype=dtype)
    return data

def decode_bin(filepath, n_features, dtype=np.float32):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(filepath, dtype=dtype)
    return scan.reshape((-1, n_features))

def auto_decode(filepath, dtype=np.float32, **kwargs):
    if filepath.endswith('.txt'):
        return decode_txt(filepath, dtype=dtype)
    
    elif filepath.endswith('.bin'):
        return decode_bin(filepath, **kwargs, dtype=dtype)
    
    else:
        raise ValueError("Invalid file extension. Supported extensions are '.txt' and '.bin'.")

def encode(data, filepath, dtype=np.float32):
    """Save the point cloud dataset to a binary file."""
    data_array = np.array(data, dtype=dtype)
    data_array.tofile(filepath)

def example():
  inputs_fn = r"/home/lucas/Programming/lucas_ws/src/github/cpp_clustering/bin/clustering/inputs/pointclouds_sequences/0000/000000.bin"
  prediction_fn = r"/home/lucas/Programming/lucas_ws/src/github/cpp_clustering/bin/clustering/predictions/pointclouds_sequences/0000/000000.bin"

  input = decode_bin(4, inputs_fn)
  predictions = decode_bin(1, prediction_fn, np.uint64)

  print(input.shape)
  print(predictions.shape)

  print(np.unique(predictions))   

if __name__ == "__name__":
  example()