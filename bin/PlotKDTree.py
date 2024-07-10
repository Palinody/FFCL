import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import json
import os
import copy

import py_helpers.IO as io

from typing import List, Tuple, Dict, Any, Union

KDBoundingBoxType = List[List[Any]]
FeatureVectorType = List[Any]
DatasetType = List[FeatureVectorType]

VERBOSE = False


class KDNode:
    def __init__(self) -> None:
        self.points: List[Tuple[float]]
        self.cut_feature_index: int
        self.kd_bounding_box: KDBoundingBoxType
        self.left: KDNode = None
        self.right: KDNode = None

    def is_leaf(self) -> bool:
        return self.cut_feature_index == -1


class KDTree:
    def __init__(self, input_path: str, keep_sequence: bool = False) -> None:
        # whether we should keep a sequence version of the data (n_samples x n_features list of lists)
        self.keep_sequence: bool = keep_sequence
        # the sequence of samples itself
        self.points_sequence: DatasetType = []

        self.kd_bounding_box: KDBoundingBoxType

        self.root = self.deserialize_kdtree(input_path)

    def deserialize_kdtree(self, input_path: str) -> KDNode:
        data = io.load_json(input_path)

        n_features: int = data["n_features"]
        options: Dict[str, Any] = data["options"]
        self.kd_bounding_box: KDBoundingBoxType = data["bounding_box"]

        if VERBOSE:
            print(f"n_features: {n_features}")
            print(f"options: {options}")
            print(f"bounding_box: {self.kd_bounding_box}")

        return self.deserialize_kdnode(data["root"], self.kd_bounding_box)

    def deserialize_kdnode(
        self, kdnode_json: KDNode, kd_bounding_box: KDBoundingBoxType
    ) -> KDNode:
        kdnode = KDNode()

        kdnode.points = kdnode_json["points"]
        kdnode.cut_feature_index = kdnode_json["axis"]
        kdnode.kd_bounding_box = kd_bounding_box

        if self.keep_sequence:
            self.points_sequence += kdnode_json["points"]

        if not kdnode.is_leaf():
            cut_axis: int = kdnode_json["axis"]
            cut_value: float = kdnode_json["points"][0][cut_axis]

            if "left" in kdnode_json.keys():
                kd_bounding_box_left_copy = copy.deepcopy(kd_bounding_box)

                # set the max bounding value at the cut axis equal to the value at the pivot cut axis
                kd_bounding_box_left_copy[cut_axis][1] = cut_value

                kdnode.left = self.deserialize_kdnode(
                    kdnode_json["left"], kd_bounding_box_left_copy
                )
            if "right" in kdnode_json.keys():
                kd_bounding_box_right_copy = copy.deepcopy(kd_bounding_box)

                # set the min bounding value at the cut axis equal to the value at the pivot cut axis
                kd_bounding_box_right_copy[cut_axis][0] = cut_value

                kdnode.right = self.deserialize_kdnode(
                    kdnode_json["right"], kd_bounding_box_right_copy
                )
        return kdnode

    def get_sequence(self):
        return self.points_sequence


def plot_2dtree(
    kdnode: KDNode,
    ax: Axes,
    x_lim: List[float] = None,
    y_lim: List[float] = None,
    color: Union[None, str] = None,
    splitting_lane_color: Union[None, str] = None,
) -> None:
    if kdnode is None:
        return

    is_leaf: bool = kdnode.is_leaf()

    x, y = zip(*kdnode.points) if kdnode.points else [[], []]
    # set the leaf points to green and the pivot ones to blue
    # points_color = "green" if is_leaf else "blue"
    points_color = color if color is not None else ("green" if is_leaf else "blue")
    # put the pivot points in the foreground so that they can be seen
    zorder = -1 if is_leaf else 0
    ax.scatter(x, y, color=points_color, zorder=zorder)

    if x_lim:
        ax.set_xlim(x_lim)

    if y_lim:
        ax.set_xlim(y_lim)

    ax.grid(True)

    if not is_leaf:
        draw_orthogonal_line(
            kdnode.points[0],
            kdnode.cut_feature_index,
            kdnode.kd_bounding_box,
            ax,
            color=splitting_lane_color,
        )

    if kdnode.left is not None:
        plot_2dtree(kdnode.left, ax, x_lim, y_lim, color, splitting_lane_color)

    if kdnode.right is not None:
        plot_2dtree(kdnode.right, ax, x_lim, y_lim, color, splitting_lane_color)


def plot_bbox(bbox, ax, color="red", linestyle="-", alpha=1) -> None:
    point1, point2 = zip(*bbox)
    x = [point1[0], point2[0], point2[0], point1[0], point1[0]]
    y = [point1[1], point1[1], point2[1], point2[1], point1[1]]
    ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha)


def draw_orthogonal_line(
    pivot_point, cut_index, bounding_box, ax, color="red", linestyle="--", alpha=1
):
    # bound the segment_representation with the min max values of the non pivot axis dimension
    other_axis_min, other_axis = bounding_box[1 - cut_index]
    pivot_value = pivot_point[cut_index]
    if cut_index == 0:
        ax.plot(
            [pivot_value, pivot_value],
            [other_axis_min, other_axis],
            color=color,
            linestyle=linestyle,
            alpha=alpha,
        )
    elif cut_index == 1:
        ax.plot(
            [other_axis_min, other_axis],
            [pivot_value, pivot_value],
            color=color,
            linestyle=linestyle,
        )


def main():
    """noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kdtree/")

    file_names = [
        f
        for f in os.listdir(root_folder)
        if os.path.isfile(os.path.join(root_folder, f))
    ]

    for filename in file_names:
        input_path = root_folder + filename

        kdtree = KDTree(input_path, keep_sequence=False)

        fig, ax = plt.subplots()

        plot_bbox(kdtree.root.kd_bounding_box, ax, color="black")

        plot_2dtree(kdtree.root, ax=ax)

        stem = os.path.splitext(filename)[0]
        plt.title(stem)
        plt.grid(False)
        plt.tight_layout()
        plt.show()


def plot_query_reference():
    """noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kdtree/")

    file_names = [
        f
        for f in os.listdir(root_folder)
        if os.path.isfile(os.path.join(root_folder, f))
    ]

    for filename in ["noisy_moons"]:
        reference_path = root_folder + filename + "_reference.json"
        query_path = root_folder + filename + "_query.json"

        kdtree = KDTree(reference_path, keep_sequence=False)

        fig, ax = plt.subplots(figsize=(12, 6))  # Create a single plot

        # Plot reference
        plot_bbox(kdtree.root.kd_bounding_box, ax, color="orange")
        plot_2dtree(kdtree.root, ax=ax, color="orange", splitting_lane_color="orange")

        # Load query data and plot (assuming query data is loaded in a similar way to KDTree)
        kdtree_query = KDTree(query_path, keep_sequence=False)
        plot_bbox(kdtree_query.root.kd_bounding_box, ax, color="blue")
        plot_2dtree(kdtree_query.root, ax=ax, color="blue", splitting_lane_color="blue")

        # General title and layout
        ax.set_title(f"{os.path.splitext(filename)[0]}: Reference and Query")
        ax.grid(False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    plot_query_reference()
