import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import MakeClusteringDatasets
from py_helpers import IO

from matplotlib.colors import Normalize

try:
    import hdbscan
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan"])
    import hdbscan

datapath = r"./clustering/"

def plot_mst(dataset, mst, axis=None):
    cmap = plt.get_cmap('viridis')

    mst_data = mst[:, :2]
    mst_distances = mst[:, 2]
    # Normalize the distances to fit the colormap
    norm = Normalize(vmin=min(mst_distances), vmax=max(mst_distances))

    if not axis:
        fig, axis = plt.subplots()

    axis.scatter(dataset[:, 0], dataset[:, 1], color='black', alpha=0.4, label='Dataset Points')

    # Plot the MST edges with color gradient based on distances
    for index, edge in enumerate(mst_data):
        sample_index_1, sample_index_2 = edge
        sample_1 = dataset[int(sample_index_1)]
        sample_2 = dataset[int(sample_index_2)]
        distance = mst_distances[index]
        # Assign color based on distance
        color = cmap(norm(distance))
        # axis.plot([sample_1[0], sample_2[0]], [sample_1[1], sample_2[1]], color=color)
        arrowprops = dict(arrowstyle='<|-|>', mutation_scale=20, color=color, shrinkA=0)
        arrow = FancyArrowPatch(sample_1, sample_2, **arrowprops)
        axis.add_patch(arrow)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # We don't need data for the colorbar
    cbar = plt.colorbar(sm, label='Edge Distance')
    # Add labels and a legend
    axis.set_xlabel('X-axis')
    axis.set_ylabel('Y-axis')
    axis.set_title('Minimum Spanning Tree with Edge Distance Gradient')
    axis.legend()
    if not axis:
        plt.show()


def plot_hdbscan_mst(dataset, axis=None):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, gen_min_span_tree=True)
    clusterer.fit(dataset)

    if not axis:
        fig, axis = plt.subplots()

    axis = clusterer.minimum_spanning_tree_.plot(
        axis=axis,
        edge_cmap='viridis',
        edge_alpha=0.6,
        node_size=80,
        edge_linewidth=2)
    
    if not axis:
        plt.show()

def plot_hdbscan_single_linkage_tree(dataset, axis=None):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, gen_min_span_tree=True)
    clusterer.fit(dataset)

    if not axis:
        fig, axis = plt.subplots()

    axis = clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)

    
    if not axis:
        plt.show()

for dataset_name in (
        MakeClusteringDatasets.datasets_names + ["unbalanced_blobs"]
    ):
    print(dataset_name)
    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + dataset_name + ".txt")

    data = IO.auto_decode(datapath + "inputs/" + dataset_name + ".txt",
                          dtype=np.float32, 
                          n_features=n_features)

    predictions = IO.auto_decode(datapath + "predictions/" + dataset_name + ".txt", 
                                 dtype=np.float32, 
                                 n_features=3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    plot_mst(dataset=data, mst=predictions, axis=ax1)
    # plot_hdbscan_mst(dataset=data, axis=ax1)
    plot_hdbscan_single_linkage_tree(dataset=data, axis=ax2)

    plt.show()
