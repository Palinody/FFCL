import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import MakeClusteringDatasets
from py_helpers import IO

from matplotlib.colors import Normalize


datapath = r"./clustering/"

def plot_mst(dataset, mst):
    cmap = plt.get_cmap('viridis')

    mst_data = mst[:, :2]
    mst_distances = mst[:, 2]
    # Normalize the distances to fit the colormap
    norm = Normalize(vmin=min(mst_distances), vmax=max(mst_distances))

    fig, ax = plt.subplots()

    ax.scatter(dataset[:, 0], dataset[:, 1], color='black', alpha=0.4, label='Dataset Points')

    # Plot the MST edges with color gradient based on distances
    for index, edge in enumerate(mst_data):
        sample_index_1, sample_index_2 = edge
        sample_1 = dataset[int(sample_index_1)]
        sample_2 = dataset[int(sample_index_2)]
        distance = mst_distances[index]
        # Assign color based on distance
        color = cmap(norm(distance))
        # ax.plot([sample_1[0], sample_2[0]], [sample_1[1], sample_2[1]], color=color)
        arrowprops = dict(arrowstyle='<|-|>', mutation_scale=20, color=color, shrinkA=0)
        arrow = FancyArrowPatch(sample_1, sample_2, **arrowprops)
        ax.add_patch(arrow)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # We don't need data for the colorbar
    cbar = plt.colorbar(sm, label='Edge Distance')
    # Add labels and a legend
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Minimum Spanning Tree with Edge Distance Gradient')
    ax.legend()
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
    
    plot_mst(dataset=data, mst=predictions)