import numpy as np

try:
    from sklearn.neighbors import KDTree
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "sklearn"])
    from sklearn.neighbors import KDTree

try:
    from pyclustering.container import kdtree
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyclustering"])
    from pyclustering.container import kdtree


try:
    import pyflann
except:
    # dont install automatically pyflann because the default is full of bugs with python3
    # the current tests have been made with a version fixed locally
    # see: https://github.com/primetang/pyflann/issues/1 to fix it yourself if you want to install
    pass
    """
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyflann"])
    import pyflann
    """
# from scipy.spatial import KDTree
import time
import timeit
import os
import sys

BUCKET_SIZE: int = 40


def read_dataset(filepath: str):
    return np.loadtxt(filepath, dtype=np.float32, delimiter=" ")


def cyclic_splitter(X, depth):
    """Select the splitting axis by cycling through the dimensions."""
    return depth % X.shape[1]


def TestPyclusteringKDTreeBuildTime(points: np.ndarray):
    np.random.shuffle(points)

    start_time = time.process_time()
    tree = kdtree.kdtree(points)
    end_time = time.process_time()
    # print the elapsed time
    print(
        "Elapsed time for KDTree construction (pyclustering):",
        end_time - start_time,
        "seconds",
    )


def TestSklearnKDTreeBuildTime(points: np.ndarray):
    np.random.shuffle(points)

    start_time = time.process_time()
    tree = KDTree(points, leaf_size=BUCKET_SIZE)
    end_time = time.process_time()
    # print the elapsed time
    print(
        "Elapsed time for KDTree construction (sklearn):",
        end_time - start_time,
        "seconds",
    )


def TestFlannKDTreeBuildTime(points: np.ndarray):
    """FLANN kdtree build modes
    "median": (default) split the points into two halves according to their median value in the chosen dimension.
    "mean": split the points into two halves according to their mean value in the chosen dimension.
    "rand": split the points randomly in the chosen dimension.
    "gini": split the points based on Gini impurity, a measure of node purity commonly used in decision trees.
    "entropy": split the points based on information gain, a measure of node purity commonly used in decision trees.
    "threshold": split the points using a fixed threshold value in the chosen dimension.
    "best": choose the best splitting method automatically based on the data and other parameters.
    """
    np.random.shuffle(points)

    start_time = time.process_time()
    flann = pyflann.FLANN()
    params = flann.build_index(
        points,
        algorithm="kdtree",
        split_method="median",
        copy_data=False,
        cores=1,
        trees=1,
        leaf_max_size=BUCKET_SIZE,
        sample_fraction=0.1,
        checks=-1,
        build_weight=0.01,
    )
    end_time = time.process_time()
    print(
        "Elapsed time for KDTree construction (pyflann):",
        end_time - start_time,
        "seconds",
    )

    # print("FLANN build_index parameters:")
    # print(params)


def run_all():
    """noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "clustering/inputs/"
    )

    file_names = os.listdir(root_folder)

    for filename in file_names:
        input_path = root_folder + filename

        dataset = read_dataset(input_path)

        print("---")
        print(filename)
        print(dataset.shape)

        TestPyclusteringKDTreeBuildTime(dataset)

        TestSklearnKDTreeBuildTime(dataset)

        try:
            TestFlannKDTreeBuildTime(dataset)
        except:
            print(dir(pyflann))


def run_mnist():
    # where the datasets are placed (if they arent the should be generated from dataset_maker.py)
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "clustering/"
    )
    # filename: str = "noisy_circles.txt"
    filename: str = "mnist.txt"

    input_path = root_folder + "inputs/" + filename

    dataset = read_dataset(input_path)

    print(dataset.shape)

    # TestPyclusteringKDTreeBuildTime(dataset)

    TestSklearnKDTreeBuildTime(dataset)

    try:
        TestFlannKDTreeBuildTime(dataset)
    except:
        print(dir(pyflann))


if __name__ == "__main__":
    run_all()
