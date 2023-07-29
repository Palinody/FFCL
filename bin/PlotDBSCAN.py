import numpy as np
import matplotlib.pyplot as plt
import MakeClusteringDatasets
from py_helpers import IO

datapath = r"./clustering/"

for dataset_name in (
        MakeClusteringDatasets.datasets_names + ["unbalanced_blobs"]
    ):
    print(dataset_name)
    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + dataset_name + ".txt")

    data = IO.auto_decode(datapath + "inputs/" + dataset_name + ".txt",
                          dtype=np.float32, 
                          n_features=n_features)

    predictions = IO.auto_decode(datapath + "predictions/" + dataset_name + ".txt", 
                                 dtype=np.int32, 
                                 n_features=1)

    # Separate the points based on their predictions
    noise_points = data[predictions == 0]

    plt.scatter(noise_points[:, 0], noise_points[:, 1], color='black', marker='x', label='Noise')

    # Assign a unique color to each cluster index
    unique_predictions = np.unique(predictions)

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_predictions)))

    for label, color in zip(unique_predictions, colors):
        if label > 0:
            cluster_points_label = data[predictions == label]
            plt.scatter(cluster_points_label[:, 0], 
                        cluster_points_label[:, 1], 
                        color=color, 
                        label=f'Cluster {label}')

    plt.title(dataset_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
