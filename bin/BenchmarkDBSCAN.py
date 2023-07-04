import numpy as np
import time, os, math

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "sklearn"])
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree

PLOT_RESULTS = True

def read_dataset(filepath: str):
    return np.loadtxt(filepath, dtype=np.float32, delimiter=" ")

def TestSkLearnDBSCAN(points: np.ndarray, epsilon, min_samples):
    np.random.shuffle(points)

    clustering = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='kd_tree')
    
    start_time = time.process_time()
    clustering.kdtree_ = KDTree(points, leaf_size=math.sqrt(points.shape[0]))
    end_time = time.process_time()
    # print the elapsed time
    print(
        "Elapsed time for KDTREE (sklearn):",
        end_time - start_time,
        "seconds",
    )

    start_time = time.process_time()
    clustering.fit(points)
    end_time = time.process_time()
    # print the elapsed time
    print(
        "Elapsed time for DBSCAN (sklearn):",
        end_time - start_time,
        "seconds",
    )

    if PLOT_RESULTS:
        pass

def run_all():
    """noisy_circles.txt, noisy_moons.txt, varied.txt, aniso.txt, blobs.txt, no_structure.txt, unbalanced_blobs.txt"""
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "clustering/inputs/"
    )

    file_names = [
        "noisy_circles.txt",
        "noisy_moons.txt",
        "varied.txt",
        "aniso.txt",
        "blobs.txt",
        "no_structure.txt",
        "unbalanced_blobs.txt",
    ]  # os.listdir(root_folder)

    datasets_parameters = [
        (2, 5), 
        (1, 5), 
        (1, 3),
        (1.2,10),
        (1, 10),
        (1, 5),
        (2, 5)
    ]

    for filename, dataset_parameters in zip(file_names, datasets_parameters):
        input_path = root_folder + filename
        dataset = read_dataset(input_path)
        epsilon, min_samples = dataset_parameters

        print("---")
        print(filename)
        print(dataset.shape)
        print(f"Epsilon: {epsilon}, MinSamples: {min_samples}")

        TestSkLearnDBSCAN(dataset, epsilon=epsilon, min_samples=min_samples)

if __name__ == "__main__":
    run_all()
