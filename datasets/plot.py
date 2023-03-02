import matplotlib.pyplot as plt
import math
import csv
import numpy as np
import random
import colorsys

from sklearn import cluster, datasets, mixture

from typing import Tuple, List, Any

datapath = r"./clustering/"


def generate_colors_list(n_colors: int) -> List[int]:
    colors = [0] * n_colors
    # generate a buffer of index candidates
    indices_buff = list(range(n_colors))
    for i in range(n_colors):
        # choose an index
        buff_idx = random.randint(0, len(indices_buff) - 1)
        # remove the element in the list and save the value (current index)
        current_idx = indices_buff.pop(buff_idx)
        # convert the chosen index to a hex color and spread the values
        colors[i] = "#%06x" % int(current_idx * 0xFFFFFF / n_colors)
    return colors
    # return ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n_colors)]


def generate_color_spectrum(n_colors):
    colors = []
    for i in range(n_colors):
        r = random.random()
        g = random.random()
        b = random.random()
        h, s, l = colorsys.rgb_to_hls(r, g, b)
        colors.append((h, (r, g, b)))

    colors.sort()
    return [color[1] for color in colors]


def read_csv(filepath: str) -> list:
    data = []
    with open(filepath, mode="r") as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            data += line
    return data


def read_txt(filepath: str, n_features: int, delimiter: str = " ") -> list:
    data = []
    with open(filepath, mode="r") as file:
        for line in file.readlines():
            data += [float(elem) for elem in line.split(delimiter)]
    return data


def split_data_by_classes(
    np_data: np.ndarray, np_labels: np.ndarray, target_label: int
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = []
    labels = []
    for np_sample, np_label in zip(np_data, np_labels):
        if np_label == target_label:
            dataset.append(np_sample)
            labels.append(np_label)
    np_dataset = np.array(dataset, ndmin=2)
    np_labels = np.array(dataset)
    return np_dataset, np_labels


def num_features_in_file(filename: str) -> int:
    with open(filename, "r") as file:
        first_line = file.readline()
        return len(first_line.split(" "))


import datasets_maker

if __name__ == "__main__":
    DIM_X = 0
    DIM_Y = 1
    #  datasets_maker.datasets_names + ["iris"] + ["unbalanced_blobs"] + ["mnist_train"]
    # for dataset_name in ["mnist_train"]:
    for dataset_name in datasets_maker.datasets_names + ["iris"] + ["unbalanced_blobs"]:
        print(dataset_name)
        n_features = num_features_in_file(datapath + "inputs/" + dataset_name + ".txt")

        data = read_txt(datapath + "inputs/" + dataset_name + ".txt", n_features)
        predictions = read_txt(datapath + "predictions/" + dataset_name + ".txt", 1)
        targets = read_txt(datapath + "targets/" + dataset_name + ".txt", 1)

        try:
            centroids = read_txt(
                datapath + "centroids/" + dataset_name + ".txt", n_features
            )
        except:
            centroids = None

        np_data = np.array(data).reshape([-1, n_features])
        np_labels = np.array(predictions, dtype=int)
        np_targets = np.array(targets, dtype=int)
        try:
            np_centroids = np.array(centroids).reshape([-1, n_features])
        except:
            np_centroids = None

        n_labels = int(np.max(predictions) + 1)
        # or use ["r", "g", "b", "k"]
        colors = [
            "k",
            "g",
            "b",
            "y",
            "purple",
            "cyan",
            "orange",
            "brown",
            "azure",
            "gray",
            "lime",
            "burlywood",
            "indigo",
            "darkviolet",
            "pink",
        ]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7.5))

        # first plot with predictions
        for label in sorted(np.unique(np_labels)):
            np_data_filtered, np_labels_filtered = split_data_by_classes(
                np_data, np_labels, label
            )
            # 2 & 3
            ax1.scatter(
                np_data_filtered[:, DIM_X],
                np_data_filtered[:, DIM_Y],
                color=colors[label],
                alpha=0.4,
            )
            ax1.set_title("predictions")
        try:
            ax1.scatter(
                np_centroids[:, DIM_X], np_centroids[:, DIM_Y], color="r", alpha=1, s=50
            )
        except:
            pass

        # second plot with targets
        for target in sorted(np.unique(np_targets)):
            np_data_filtered, np_labels_filtered = split_data_by_classes(
                np_data, np_targets, target
            )
            # 2 & 3
            ax2.scatter(
                np_data_filtered[:, DIM_X],
                np_data_filtered[:, DIM_Y],
                color=colors[target],
                alpha=0.4,
            )
            ax2.set_title("targets")
        fig.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()
