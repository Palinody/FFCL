import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

import numpy as np
import json
import os
import copy


from typing import List, Tuple, Dict, Any

KDBoundingBoxType = List[List[Any]]


def load_json(input_path):
    with open(input_path, "r") as f:
        try:
            return json.load(f)

        except json.JSONDecodeError as e:
            print(f"Error loading JSON file {input_path}: {e}")
            return None


class KDNode:
    def __init__(self) -> None:
        self.points: List[Tuple[float]]
        self.cut_feature_index: int
        self.kd_bounding_box: KDBoundingBoxType
        self.left: KDNode = None
        self.right: KDNode = None

    def is_leaf(self) -> bool:
        return self.cut_feature_index == -1

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class KDTree:
    def __init__(self, input_path: str, keep_sequence: bool = False) -> None:
        self.keep_sequence = keep_sequence
        self.points_sequence: List[Any] = []

        self.kd_bounding_box: KDBoundingBoxType = None

        self.root = self.deserialize_kdtree(input_path)

    def deserialize_kdtree(self, input_path: str) -> KDNode:
        data = load_json(input_path)

        n_samples: int = data["n_samples"]
        n_features: int = data["n_features"]
        options: Dict[str, Any] = data["options"]
        self.kd_bounding_box: KDBoundingBoxType = data["bounding_box"]

        print(f"n_samples: {n_samples}")
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

        print(kd_bounding_box)

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
                # restore the original bounding box
                kd_bounding_box_left_copy[cut_axis][1] = kdnode.kd_bounding_box[
                    cut_axis
                ][1]

            if "right" in kdnode_json.keys():
                kd_bounding_box_right_copy = copy.deepcopy(kd_bounding_box)

                # set the min bounding value at the cut axis equal to the value at the pivot cut axis
                kd_bounding_box_right_copy[cut_axis][0] = cut_value

                kdnode.right = self.deserialize_kdnode(
                    kdnode_json["right"], kd_bounding_box_right_copy
                )
                # restore the original bounding box
                kd_bounding_box_right_copy[cut_axis][0] = kdnode.kd_bounding_box[
                    cut_axis
                ][0]

        return kdnode

    def get_sequence(self):
        return self.points_sequence

    def scatter_plot_2d(self, kdnode: KDNode, ax: Axes) -> None:
        if kdnode is None:
            return

        x, y = zip(*kdnode.points)
        color = "green" if kdnode.is_leaf else "blue"
        ax.scatter(x, y, color=color)

        plot_bbox(kdnode.kd_bounding_box, ax)

        if kdnode.left is not None:
            self.scatter_plot_2d(kdnode.left, ax)

        if kdnode.right is not None:
            self.scatter_plot_2d(kdnode.right, ax)


def plot_bbox(bbox, ax):
    point1, point2 = zip(*bbox)
    x = [point1[0], point2[0], point2[0], point1[0], point1[0]]
    y = [point1[1], point1[1], point2[1], point2[1], point1[1]]
    ax.plot(x, y, color="red", linestyle="-")


def main():
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kdtree/")

    filename: str = "noisy_circles.json"

    input_path = root_folder + filename

    # deserialize_kdtree(input_path)

    kdtree = KDTree(input_path, True)

    """
    points_sequence = kdtree.get_sequence()
    x = [row[0] for row in points_sequence]
    y = [row[1] for row in points_sequence]

    plt.scatter(x, y, color="blue")
    plt.show()"""
    fig, ax = plt.subplots()

    kdtree.scatter_plot_2d(kdtree.root, ax=ax)

    plot_bbox(
        kdtree.kd_bounding_box,
        ax,
    )

    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
