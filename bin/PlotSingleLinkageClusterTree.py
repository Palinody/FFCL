import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage, single

from sklearn.cluster import AgglomerativeClustering

import os

from typing import List, Tuple, Dict, Any

import py_helpers.IO as IO


class SingleLinkageClusterNode:
    def __init__(self, representative: int, level: float) -> None:
        self.representative = representative
        self.level = level
        self.left: SingleLinkageClusterNode = None
        self.right: SingleLinkageClusterNode = None

    def is_leaf(self) -> bool:
        return self.left == None and self.right == None


class SLINKTree:
    def __init__(self, input_path: str) -> None:
        self.children = np.ndarray
        self.distances = np.ndarray
        self.cluster_sizes = np.ndarray

        self.cluster_count = 0

        self.root = self.deserialize_slink_tree(input_path)

    def deserialize_slink_tree(self, input_path: str) -> SingleLinkageClusterNode:
        data = IO.load_json(input_path)

        self.n_samples = data["root"]["cluster_size"]

        # the index at which the data will be written in the children, distances, cluster_sizes buffers
        # this ensures that the higher level clusters in the tree are placed at the end to comply with
        # sklearn's dendrogram plotting input encoding
        self.write_at_index = self.n_samples - 2

        self.children = np.zeros((self.n_samples - 1, 2))
        self.distances = np.zeros(self.n_samples - 1)
        self.cluster_sizes = np.zeros(self.n_samples - 1)

        return self.deserialize_slink_node(data["root"])

    def deserialize_slink_node(
        self, kdnode_json: SingleLinkageClusterNode
    ) -> SingleLinkageClusterNode:
        slink_node = SingleLinkageClusterNode(
            kdnode_json["representative"], kdnode_json["level"]
        )

        if "left" in kdnode_json and "right" in kdnode_json:
            children = np.array(
                [
                    kdnode_json["left"]["representative"],
                    kdnode_json["right"]["representative"],
                ]
            )

            # check if the left child node is a cluster node before assigning a unique cluster label
            if "left" in kdnode_json and "right" in kdnode_json["left"]:
                children[0] = self.n_samples + self.cluster_count
                self.cluster_count += 1

            # check if the right child node is a cluster node before assigning a unique cluster label
            if "left" in kdnode_json and "right" in kdnode_json["right"]:
                children[1] = self.n_samples + self.cluster_count
                self.cluster_count += 1

            self.children[self.write_at_index] = children
            self.distances[self.write_at_index] = kdnode_json["level"]
            self.cluster_sizes[self.write_at_index] = kdnode_json["cluster_size"]

            self.write_at_index -= 1

            if "left" in kdnode_json.keys():
                slink_node.left = self.deserialize_slink_node(kdnode_json["left"])

            if "right" in kdnode_json.keys():
                slink_node.right = self.deserialize_slink_node(kdnode_json["right"])

        return slink_node

    def make_linkage_matrix(self) -> np.ndarray:
        linkage_matrix = np.column_stack(
            [self.children, self.distances, self.cluster_sizes]
        ).astype(np.float64)

        sorted_indices = np.argsort(linkage_matrix[:, 2])
        sorted_linkage_matrix = linkage_matrix[sorted_indices]

        self.cluster_count = 0
        for row_index, (cluster_node_1, cluster_node_2) in enumerate(
            sorted_linkage_matrix[:, :2]
        ):
            if cluster_node_1 >= self.n_samples:
                sorted_linkage_matrix[row_index, 0] = (
                    self.n_samples + self.cluster_count
                )
                self.cluster_count += 1

            if cluster_node_2 >= self.n_samples:
                sorted_linkage_matrix[row_index, 1] = (
                    self.n_samples + self.cluster_count
                )
                self.cluster_count += 1

        return sorted_linkage_matrix

    def show_dendrogram(
        self, title=None, x_axis_name=None, y_axis_name=None, axis=None, **kwargs
    ):
        if not axis:
            fig, axis = plt.subplots(figsize=(12, 8))

        linkage_matrix = self.make_linkage_matrix()

        dendrogram(
            linkage_matrix,
            p=1.4,
            orientation="top",
            truncate_mode=None,# "level",
            ax=axis,
            **kwargs,
        )

        if title:
            axis.set_title(title)

        if x_axis_name:
            axis.set_xlabel(x_axis_name)

        if y_axis_name:
            axis.set_ylabel(y_axis_name)

        plt.show()


def plot_predictions(datapath, filename):
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

    plt.scatter(
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
            plt.scatter(
                cluster_points_label[:, 0],
                cluster_points_label[:, 1],
                color=color,
                label=f"Cluster {label}",
            )

    plt.title(filename)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def main():
    """noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "single_linkage_cluster_tree/"
    )

    file_names = os.listdir(root_folder)

    datapath = r"./clustering/"

    for filename in file_names:
        input_path = root_folder + filename

        slink_tree = SLINKTree(input_path)

        stem = os.path.splitext(filename)[0]

        print(stem)

        slink_tree.show_dendrogram(
            title=stem,
            x_axis_name="sample index",
            y_axis_name="k mutual reachability distance",
        )

        plot_predictions(datapath, stem)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    main()
