import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

import os

import py_helpers.IO as IO

try:
    import hdbscan
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan"])
    import hdbscan

def plot_predictions(datapath, filename, axis=None):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + filename + ".txt")

    data = IO.auto_decode(
        datapath + "inputs/" + filename + ".txt",
        dtype=np.float32,
        n_features=n_features,
    )

    predictions = IO.auto_decode(
        datapath + "predictions/" + filename + ".txt", dtype=np.int32, n_features=1
    )

    # Separate the points based on their predictions
    noise_points = data[predictions == 0]

    axis.set_facecolor('lightgray')

    axis.scatter(
        noise_points[:, 0],
        noise_points[:, 1],
        color="black",
        marker="x",
        label="Noise",
    )

    # Assign a unique color to each cluster index
    unique_predictions = np.unique(predictions)

    colors = plt.cm.inferno(np.linspace(0, 1, len(unique_predictions)))

    for label, color in zip(unique_predictions, colors):
        if label > 0:
            cluster_points_label = data[predictions == label]
            axis.scatter(
                cluster_points_label[:, 0],
                cluster_points_label[:, 1],
                color=color,
                label=f"Cluster {label}",
            )

    axis.set_title(f"FFCL {filename}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.legend()

    if standalone_plot:
        plt.show()

def plot_hdbscan_predictions(datapath, filename, axis=None):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + filename + ".txt")

    data = IO.auto_decode(
        datapath + "inputs/" + filename + ".txt",
        dtype=np.float32,
        n_features=n_features,
    ).astype(np.float32)

    mst = IO.auto_decode(
        datapath + "predictions/" + filename + ".txt",
        dtype=np.float32,
        n_features=3,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=1,
        cluster_selection_epsilon=0,
        gen_min_span_tree=True,
        approx_min_span_tree=False,
        core_dist_n_jobs=1,
        allow_single_cluster=False
    )
    predictions = clusterer.fit_predict(data)

    # Separate the points based on their predictions
    noise_points = data[predictions == -1]

    axis.set_facecolor('lightgray')

    axis.scatter(
        noise_points[:, 0],
        noise_points[:, 1],
        color="black",
        marker="x",
        label="Noise",
    )

    # Assign a unique color to each cluster index
    unique_predictions = np.unique(predictions)

    colors = plt.cm.inferno(np.linspace(0, 1, len(unique_predictions)))

    for label, color in zip(unique_predictions, colors):
        if label != -1:
            cluster_points_label = data[predictions == label]
            axis.scatter(
                cluster_points_label[:, 0],
                cluster_points_label[:, 1],
                color=color,
                label=f"Cluster {label}",
            )

    axis.set_title(f"hdbscan lib: {filename}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.legend()

    if standalone_plot:
        plt.show()

def main():
    """noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "single_linkage_cluster_tree/"
    )

    file_names = os.listdir(root_folder)

    datapath = r"./clustering/"

    for filename in file_names:
        stem = os.path.splitext(filename)[0]

        print(stem)

        
        fig = plt.figure(num=1, figsize=(24, 12))
        gs = gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        plot_predictions(datapath, stem, ax1)
        plot_hdbscan_predictions(datapath, stem, ax2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    main()
