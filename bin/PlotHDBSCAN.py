import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

import MakeClusteringDatasets
from py_helpers import IO


try:
    import hdbscan
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan"])
    import hdbscan

datapath = r"./clustering/"


def plot_mst(dataset, mst, axis=None):
    standalone_plot = axis is None

    mst_data = mst[:, :2].astype(np.int32)
    mst_distances = mst[:, 2].astype(np.float32)
    # Normalize the distances to fit the colormap
    norm = Normalize(vmin=min(mst_distances), vmax=max(mst_distances))

    if standalone_plot:
        fig, axis = plt.subplots()

    axis.scatter(*dataset.T, color="black", alpha=0.4, label="Dataset Points")

    samples_1 = dataset[mst_data[:, 0]]
    samples_2 = dataset[mst_data[:, 1]]
    distances = norm(mst_distances)
    cmap = plt.get_cmap("viridis")
    colors = cmap(distances)

    arrowprops = dict(arrowstyle="<|-|>", mutation_scale=20, shrinkA=0)
    arrows = [
        FancyArrowPatch(start, end, color=color, **arrowprops)
        for start, end, color in zip(samples_1, samples_2, colors)
    ]

    for arrow in arrows:
        axis.add_patch(arrow)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axis, label="Edge Distance")
    axis.set_xlabel("X-axis")
    axis.set_ylabel("Y-axis")
    axis.set_title("Minimum Spanning Tree with Edge Distance Gradient")
    axis.legend()

    if standalone_plot:
        plt.show()


def plot_hdbscan_mst(clusterer, axis=None):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    clusterer.minimum_spanning_tree_.plot(
        axis=axis, edge_cmap="viridis", edge_alpha=0.6, node_size=80, edge_linewidth=2
    )

    if standalone_plot:
        plt.show()


def plot_hdbscan_predictions(inputs, predictions, axis=None):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    norm = Normalize(vmin=min(predictions), vmax=max(predictions))
    normalized_predictions = norm(predictions)

    cmap = plt.get_cmap("viridis")
    colors = cmap(normalized_predictions)

    axis.scatter(*inputs.T, color=colors, alpha=0.4, label="HDBSCAN predictions")

    if standalone_plot:
        plt.show()


def plot_hdbscan_single_linkage_tree(clusterer, axis=None):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    clusterer.single_linkage_tree_.plot(axis=axis, cmap="viridis", colorbar=True)

    if standalone_plot:
        plt.show()


def plot_hdbscan_condensed_tree(clusterer, axis=None):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    clusterer.condensed_tree_.plot(axis=axis, select_clusters=True, colorbar=True)

    if standalone_plot:
        plt.show()


for dataset_name in MakeClusteringDatasets.datasets_names + ["unbalanced_blobs"]:
    print(dataset_name)
    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + dataset_name + ".txt")

    data = IO.auto_decode(
        datapath + "inputs/" + dataset_name + ".txt",
        dtype=np.float32,
        n_features=n_features,
    ).astype(np.float32)

    mst = IO.auto_decode(
        datapath + "predictions/" + dataset_name + ".txt",
        dtype=np.float32,
        n_features=3,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        gen_min_span_tree=True,
        approx_min_span_tree=False,
        core_dist_n_jobs=1,
        allow_single_cluster=True
    )
    predictions = clusterer.fit_predict(data)
    """
    # plots from the c++ generated data
    fig = plt.figure(num=1, figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    plot_mst(dataset=data, mst=mst, axis=ax)
    """

    # plots from the hdbscan python library
    fig = plt.figure(num=1, figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    plot_hdbscan_mst(clusterer=clusterer, axis=ax1)
    plot_hdbscan_predictions(inputs=data, predictions=predictions, axis=ax2)
    plot_hdbscan_single_linkage_tree(clusterer=clusterer, axis=ax3)
    plot_hdbscan_condensed_tree(clusterer=clusterer, axis=ax4)

    plt.tight_layout()
    plt.show()
