import numpy as np
import matplotlib.pyplot as plt

import MakeClusteringDatasets

datapath = r"./clustering/"

def read_txt(filepath: str, n_features: int, delimiter: str = " ") -> list:
    data = []
    with open(filepath, mode="r") as file:
        for line in file.readlines():
            data += [float(elem) for elem in line.split(delimiter)]
    return data

def num_features_in_file(filename: str) -> int:
    with open(filename, "r") as file:
        first_line = file.readline()
        return len(first_line.split(" "))
"""["no_structure"]"""
"""MakeClusteringDatasets.datasets_names + ["unbalanced_blobs"]"""
for dataset_name in (
        MakeClusteringDatasets.datasets_names + ["unbalanced_blobs"]
    ):
    print(dataset_name)
    n_features = num_features_in_file(datapath + "inputs/" + dataset_name + ".txt")

    data = read_txt(datapath + "inputs/" + dataset_name + ".txt", n_features)
    predictions = read_txt(datapath + "predictions/" + dataset_name + ".txt", 1)

    np_data = np.array(data).reshape([-1, n_features])
    np_labels = np.array(predictions, dtype=int)

    # Separate the points based on their np_labels
    noise_points = np_data[np_labels == 0]
    # unknown_points = np_data[np_labels == 0]
    # cluster_points = np_data[np_labels > 0]

    # Plotting
    plt.scatter(noise_points[:, 0], noise_points[:, 1], color='black', marker='x', label='Noise')
    # plt.scatter(unknown_points[:, 0], unknown_points[:, 1], facecolors='none', edgecolors='black', label='Unknown')

    # Assign a unique color to each cluster index
    unique_predictions = np.unique(np_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_predictions)))
    for label, color in zip(unique_predictions, colors):
        if label > 0:
            cluster_points_label = np_data[np_labels == label]
            plt.scatter(cluster_points_label[:, 0], cluster_points_label[:, 1], color=color, label=f'Cluster {label}')

    # Set plot title and predictions
    plt.title(dataset_name)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
