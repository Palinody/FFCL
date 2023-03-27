import matplotlib.pyplot as plt
import json
import os


def load_json(input_path):
    with open(input_path, "r") as f:
        try:
            return json.load(f)

        except json.JSONDecodeError as e:
            print(f"Error loading JSON file {input_path}: {e}")
            return None


def main():
    root_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kdtree/")

    filename: str = "noisy_circles.json"

    input_path = root_folder + filename

    data = load_json(input_path)

    print(data)


if __name__ == "__main__":
    main()
