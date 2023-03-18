import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KDTree

# from scipy.spatial import KDTree
import time
import timeit
import os


def read_dataset(filepath: str):
    return np.loadtxt(filepath, dtype=np.float32, delimiter=" ")


def cyclic_splitter(X, depth):
    """Select the splitting axis by cycling through the dimensions."""
    return depth % X.shape[1]


def TestKDTreeBuildTime(points: np.ndarray):
    np.random.shuffle(points)

    start_time = time.process_time()
    # start_time = timeit.default_timer()
    tree = KDTree(points, leaf_size=10)
    end_time = time.process_time()
    # end_time = timeit.default_timer()
    # print the elapsed time
    print("Elapsed time for KDTree construction:", end_time - start_time, "seconds")


def main():
    # where the datasets are placed (if they arent the should be generated from dataset_maker.py)
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "clustering/"
    )
    filename: str = "mnist.txt"

    input_path = root_folder + "inputs/" + filename

    dataset = read_dataset(input_path)

    print(dataset.shape)

    TestKDTreeBuildTime(dataset)


if __name__ == "__main__":
    main()
