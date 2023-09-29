import numpy as np
import time, os, math

try:
    import hdbscan
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan"])
    import hdbscan

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "sklearn"])
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree


def read_dataset(filepath: str):
    return np.loadtxt(filepath, dtype=np.float32, delimiter=" ")

def TestHDBSCAN(points: np.ndarray, min_samples):
    # np.random.shuffle(points)
    
    clustering = hdbscan.HDBSCAN(min_cluster_size=10, 
                                 min_samples=min_samples, 
                                 gen_min_span_tree=True, 
                                 approx_min_span_tree=True, 
                                 core_dist_n_jobs=6)

    start_time = time.process_time()
    labels = clustering.fit_predict(points)
    end_time = time.process_time()
    # print the elapsed time
    print(
        "Elapsed time for HDBSCAN (sklearn):",
        end_time - start_time,
        "seconds",
    )

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
    ]

    datasets_parameters = [
        (5),
        (5),
        (5),
        (5),
        (5),
        (5),
        (5)
    ]

    for filename, dataset_parameters in zip(file_names, datasets_parameters):
        input_path = root_folder + filename
        dataset = read_dataset(input_path)
        min_samples = dataset_parameters

        print("---")
        print(f"Dataset name: '{filename}'")
        print(f"Dataset shape: {dataset.shape}")
        print(f"HDBSCAN parameters: MinSamples: {min_samples}")

        TestHDBSCAN(dataset, min_samples=min_samples)

if __name__ == "__main__":
    run_all()
