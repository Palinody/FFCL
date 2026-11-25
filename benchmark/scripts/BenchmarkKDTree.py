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
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyflann-py3"])
    import pyflann

    # dont install automatically "pyflann" because the default is full of bugs with python3
    # the current tests have been made with a version fixed locally
    # see: https://github.com/primetang/pyflann/issues/1 to fix it yourself if you want to install
    """
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyflann"])
    import pyflann
    """
# from scipy.spatial import KDTree
import time
import os
import sys
import math

BUCKET_SIZE: int = 40
RADIUS: int = 1
N_NEIGHBORS: int = 5


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
    tree = KDTree(points, leaf_size=math.sqrt(points.shape[0]))
    end_time = time.process_time()
    # print the elapsed time
    print(
        "Elapsed time for KDTree construction (sklearn):",
        end_time - start_time,
        "seconds",
    )
    # Find k nearest neighbors
    k = N_NEIGHBORS
    start_time = time.process_time()

    for query_point in points:
        distances, indices = tree.query([query_point], k=k)

    end_time = time.process_time()
    print(
        f"Elapsed time for KDTree {k} nearest neighbor (scikit-learn):",
        end_time - start_time,
        "seconds",
    )

    # Find points within a radius
    radius = RADIUS
    start_time = time.process_time()

    for query_point in points:
        distances, indices = tree.query_radius(
            [query_point], r=radius, return_distance=True, count_only=False
        )

    end_time = time.process_time()
    print(
        f"Elapsed time for KDTree {radius} radius count (scikit-learn):",
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
    # np.random.shuffle(points)

    # random_query_index = random.randint(0, points.shape[0] - 1)

    # query_index = 13201
    # query_point = points[query_index, :]
    # np.delete(points, query_index, axis=0),

    start_time = time.process_time()
    flann = pyflann.FLANN()
    params = flann.build_index(
        points,
        algorithm="kdtree",
        split_method="median",
        copy_data=False,
        cores=1,
        trees=1,
        leaf_max_size=int(math.sqrt(points.shape[0])),  # BUCKET_SIZE,
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
    # flann.get_indexed_data()
    # print("FLANN build_index parameters:")
    # print(params)
    k = N_NEIGHBORS

    start_time = time.process_time()

    for query_point in points:
        result = flann.nn_index(query_point, k)

    end_time = time.process_time()
    print(
        f"Elapsed time for KDTree {k} nearest neighbor (pyflann):",
        end_time - start_time,
        "seconds",
    )

    radius = RADIUS

    start_time = time.process_time()

    for query_point in points:
        indices, distances = flann.nn_radius(query_point, radius)

    end_time = time.process_time()
    print(
        f"Elapsed time for KDTree {radius} radius count (pyflann):",
        end_time - start_time,
        "seconds",
    )
    print(f"num_neighbors: {indices.shape[0]}, radius: {radius}")


def run_all():
    """noisy_circles.txt, noisy_moons.txt, varied.txt, aniso.txt, blobs.txt, no_structure.txt, unbalanced_blobs.txt"""
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "clustering/inputs/"
    )

    file_names = [
        "noisy_circles.txt",
        "noisy_moons.txt",
        "varied.txt",
        "aniso.txt",
        "blobs.txt",
        "no_structure.txt",
        "unbalanced_blobs.txt",
    ]  # os.listdir(root_folder)

    for filename in file_names:
        input_path = root_folder + filename

        dataset = read_dataset(input_path)

        print("---")
        print(filename)
        print(dataset.shape)

        TestPyclusteringKDTreeBuildTime(dataset)

        TestSklearnKDTreeBuildTime(dataset)

        # try:
        TestFlannKDTreeBuildTime(dataset)
        # except:
        # print(dir(pyflann))


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

    TestPyclusteringKDTreeBuildTime(dataset)

    TestSklearnKDTreeBuildTime(dataset)

    try:
        TestFlannKDTreeBuildTime(dataset)
    except:
        print(dir(pyflann))


if __name__ == "__main__":
    run_all()
