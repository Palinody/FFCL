import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import MakeClusteringDatasets
from py_helpers import IO
from PlotKDTree import KDTree, plot_bbox, plot_2dtree
from typing import List

datapath = r"./clustering/"


def plot_closest_edge(
    dataset, labels, axis=None, x_lim: List[float] = None, y_lim: List[float] = None
):
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    labels_int = labels.astype(np.int32)

    queries_samples = dataset[labels_int == 0]
    reference_samples = dataset[labels_int == 1]
    closest_pair_query_point = dataset[labels_int == 2]
    closest_pair_reference_point = dataset[labels_int == 3]

    axis.scatter(*queries_samples.T, color="blue", alpha=0.4, label="Queries samples")
    axis.scatter(
        *reference_samples.T, color="black", alpha=1, label="Reference samples"
    )
    axis.scatter(
        *closest_pair_query_point.T, color="red", label="Closest pair query element"
    )
    axis.scatter(
        *closest_pair_reference_point.T,
        color="green",
        label="Closest pair reference element",
    )
    distance = np.linalg.norm(closest_pair_query_point - closest_pair_reference_point)

    axis.plot(
        [closest_pair_query_point[0, 0], closest_pair_reference_point[0, 0]],
        [closest_pair_query_point[0, 1], closest_pair_reference_point[0, 1]],
        color="black",
        label=f"Closest pair of points, distance = {distance}",
    )

    if x_lim:
        axis.set_xlim(x_lim)

    if y_lim:
        axis.set_xlim(y_lim)

    axis.grid(True)
    axis.set_xlabel("X-axis")
    axis.set_ylabel("Y-axis")
    axis.set_title("FFCL dual tree closest pair of points")
    axis.legend()

    if standalone_plot:
        plt.show()


def nth_closest_pair(dist_matrix, n):
    flat_dist_matrix = dist_matrix.flatten()

    # Find the n-th smallest distance's index in the flattened array
    n_th_index = np.argpartition(flat_dist_matrix, n - 1)[n - 1]

    # Convert the flattened index back to a pair of indices in the 2D matrix
    n_th_min_indices = np.unravel_index(n_th_index, dist_matrix.shape)

    return n_th_min_indices


def scipy_plot_closest_edges(
    dataset,
    labels,
    axis=None,
    x_lim: List[float] = None,
    y_lim: List[float] = None,
    kth_closest=1,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import distance_matrix

    # Check if a separate plotting axis is provided or if one should be created
    standalone_plot = axis is None

    if standalone_plot:
        fig, axis = plt.subplots()

    # Ensure labels are integers for indexing
    labels_int = labels.astype(np.int32)

    # Separate the dataset based on labels
    queries_samples = dataset[(labels_int == 0) | (labels_int == 2)]
    reference_samples = dataset[(labels_int == 1) | (labels_int == 3)]
    # queries_samples = dataset[labels_int == 0]
    # reference_samples = dataset[labels_int == 1]

    # Compute the distance matrix between query samples and reference samples
    dist_matrix = distance_matrix(queries_samples, reference_samples)

    # Find the indices of the closest pair
    # min_dist_indices = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

    kth_closest_dist_indices = nth_closest_pair(dist_matrix, kth_closest)

    # Extract the closest pair of points
    closest_pair_query_point = queries_samples[kth_closest_dist_indices[0]]
    closest_pair_reference_point = reference_samples[kth_closest_dist_indices[1]]

    # Plotting
    axis.scatter(
        queries_samples[:, 0],
        queries_samples[:, 1],
        color="blue",
        alpha=0.4,
        label="Queries samples",
    )
    axis.scatter(
        reference_samples[:, 0],
        reference_samples[:, 1],
        color="black",
        alpha=1,
        label="Reference samples",
    )
    axis.scatter(
        closest_pair_query_point[0],
        closest_pair_query_point[1],
        color="red",
        label="Closest pair query element",
    )
    distance = np.linalg.norm(closest_pair_query_point - closest_pair_reference_point)
    axis.scatter(
        closest_pair_reference_point[0],
        closest_pair_reference_point[1],
        color="green",
        label=f"Closest pair reference element, distance = {distance}",
    )

    # Draw a line between the closest pair of points
    axis.plot(
        [closest_pair_query_point[0], closest_pair_reference_point[0]],
        [closest_pair_query_point[1], closest_pair_reference_point[1]],
        color="black",
        label="Closest pair of points",
    )

    if x_lim:
        axis.set_xlim(x_lim)

    if y_lim:
        axis.set_xlim(y_lim)

    axis.grid(True)
    # Set plot details
    axis.set_xlabel("X-axis")
    axis.set_ylabel("Y-axis")
    axis.set_title("Scipy Dual Tree Closest Pair of Points")
    axis.legend()

    # Display the plot if created within the function
    if standalone_plot:
        plt.show()


kdtree_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kdtree/")


"""noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
for dataset_name in [
    "noisy_moons",
]:  # MakeClusteringDatasets.datasets_names + ["unbalanced_blobs"]:
    print(dataset_name)
    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + dataset_name + ".txt")

    data = IO.auto_decode(
        datapath + "inputs/" + dataset_name + ".txt",
        dtype=np.float32,
        n_features=n_features,
    ).astype(np.float32)

    labels = IO.auto_decode(
        datapath + "predictions/" + dataset_name + ".txt",
        dtype=np.int32,
        n_features=1,
    )

    x_lim: List[float] = data.min(axis=1).min(), data.max(axis=1).max()
    y_lim: List[float] = data.min(axis=0).min(), data.max(axis=0).max()

    fig = plt.figure(num=4, figsize=(24, 12))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    # plot from the c++ generated data
    plot_closest_edge(dataset=data, labels=labels, axis=ax1, x_lim=x_lim, y_lim=y_lim)
    scipy_plot_closest_edges(
        dataset=data, labels=labels, axis=ax2, x_lim=x_lim, y_lim=y_lim, kth_closest=10
    )

    # ---
    ax3 = plt.subplot(gs[1, 0])
    kdtree = KDTree(os.path.join(kdtree_folder, dataset_name + "_query.json"))
    plot_bbox(kdtree.root.kd_bounding_box, ax3, color="blue")
    plot_2dtree(kdtree.root, ax=ax3, x_lim=x_lim, y_lim=y_lim)

    # ---
    ax4 = plt.subplot(gs[1, 1])
    kdtree = KDTree(os.path.join(kdtree_folder, dataset_name + "_reference.json"))
    plot_bbox(kdtree.root.kd_bounding_box, ax4, color="black")
    plot_2dtree(kdtree.root, ax=ax4, x_lim=x_lim, y_lim=y_lim)

    plt.tight_layout()
    plt.show()
