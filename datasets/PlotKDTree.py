import matplotlib.pyplot as plt
import json
import os

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
        self.left: KDNode
        self.right: KDNode

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


class KDTree:
    def __init__(self, input_path: str, keep_sequence: bool = False) -> None:
        self.keep_sequence = keep_sequence
        self.points_sequence: List[Any] = []

        self.root = self.deserialize_kdtree(input_path)

    def deserialize_kdtree(self, input_path: str) -> KDNode:
        data = load_json(input_path)

        n_samples: int = data["n_samples"]
        n_features: int = data["n_features"]
        options: Dict[str, Any] = data["options"]

        print(f"n_samples: {n_samples}")
        print(f"n_features: {n_features}")
        print(f"options: {options}")

        self.deserialize_kdnode(data["root"], data["bounding_box"])

    def deserialize_kdnode(
        self, kdnode_json: KDNode, kd_bounding_box: KDBoundingBoxType
    ) -> KDNode:
        kdnode = KDNode()

        kdnode.points = kdnode_json["points"]
        kdnode.cut_feature_index = kdnode_json["axis"]
        kdnode.kd_bounding_box = kd_bounding_box

        if self.keep_sequence:
            self.points_sequence += kdnode_json["points"]

        cut_axis: int = kdnode_json["axis"]
        cut_value: float = kdnode_json["points"][0][cut_axis]

        # print(f"cut_axis: {cut_axis}, cut_value: {cut_value}:.2f")

        if "left" in kdnode_json.keys():
            # set the max bounding value at the cut axis equal to the value at the pivot cut axis
            kd_bounding_box[cut_axis][1] = cut_value

            self.deserialize_kdnode(kdnode_json["left"], kd_bounding_box)
            # restore the original bounding box
            kd_bounding_box[cut_axis][1] = kdnode.kd_bounding_box[cut_axis][1]

        if "right" in kdnode_json.keys():
            # set the min bounding value at the cut axis equal to the value at the pivot cut axis
            kd_bounding_box[cut_axis][0] = cut_value

            self.deserialize_kdnode(kdnode_json["right"], kd_bounding_box)
            # restore the original bounding box
            kd_bounding_box[cut_axis][0] = kdnode.kd_bounding_box[cut_axis][0]

        return kdnode

    def get_sequence(self):
        return self.points_sequence


def main():
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kdtree/")

    filename: str = "noisy_circles.json"

    input_path = root_folder + filename

    # deserialize_kdtree(input_path)

    kdtree = KDTree(input_path, True)

    points_sequence = kdtree.get_sequence()

    # Extract x and y coordinates
    x = [row[0] for row in points_sequence]
    y = [row[1] for row in points_sequence]

    plt.scatter(x, y, color="blue")
    plt.show()


if __name__ == "__main__":
    main()
