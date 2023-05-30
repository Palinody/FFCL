import os

try:
    import numpy as np
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

try:
    from sklearn import datasets
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn import datasets

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# datasets.fetch_openml needs pandas. Check if it exists and install if it doesnt
try:
    import pandas
except:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])


VERBOSE = True

# processing on input data
SCALE_MULTIPLIER = 10
SHIFT = 0
TYPE = np.float32

# n_samples for all the toy datasets except mnist
n_samples = 20000
# None to load everything else specify
n_samples_mnist = None
n_samples_mnist = n_samples

# np.random.seed(1)
random_state = 1

noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state
)
noisy_moons = datasets.make_moons(
    n_samples=n_samples, noise=0.05, random_state=random_state
)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
no_structure = np.random.rand(n_samples, 2), np.zeros((n_samples))

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

datasets_names = [
    "noisy_circles",
    "noisy_moons",
    "varied",
    "aniso",
    "blobs",
    "no_structure",
]

datasets_params = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {}),
]


def normalize_dataset(inputs, labels, scale=1, shift=0):
    # normalize dataset for easier parameter selection
    inputs = (StandardScaler().fit_transform(inputs) * scale + shift).astype(TYPE)
    labels = labels.reshape(-1, 1).astype(np.int64)
    return inputs, labels


def normalize_mnist(inputs, labels):
    # normalize dataset for easier parameter selection
    inputs = MinMaxScaler(feature_range=(0, 1)).fit_transform(inputs).astype(TYPE)
    labels = labels.reshape(-1, 1).astype(np.int64)
    return inputs, labels


def normalize_dataset_with_imbalances(inputs):
    ratio1 = 0.3333
    ratio2 = 0.0666
    ratio3 = 0.01
    X = (StandardScaler().fit_transform(inputs) * SCALE_MULTIPLIER + SHIFT).astype(TYPE)
    X_filtered = np.vstack(
        (
            X[y == 0][: int(n_samples * ratio1), :],
            X[y == 1][: int(n_samples * ratio2), :],
            X[y == 2][: int(n_samples * ratio3), :],
        )
    )
    y_filtered = np.array(
        [0] * int(n_samples * ratio1)
        + [1] * int(n_samples * ratio2)
        + [2] * int(n_samples * ratio3),
        ndmin=2,
    ).T
    return X_filtered, y_filtered


def save_dataset(inputs, labels, root_folder, dataset_name):
    if VERBOSE:
        print("\t---")
        print(f"Making dataset: {dataset_name}")
        print(f"Input shape: {inputs.shape} | Labels shape: {labels.shape}")
        print(f"Mean: {np.mean(inputs)} | std: {np.std(inputs)}")

    inputs_folder = root_folder + "inputs/"
    targets_folder = root_folder + "targets/"

    if not os.path.exists(inputs_folder):
        os.makedirs(inputs_folder)

    if not os.path.exists(targets_folder):
        os.makedirs(targets_folder)

    np.savetxt(inputs_folder + dataset_name + ".txt", inputs, delimiter=" ")
    np.savetxt(
        targets_folder + dataset_name + ".txt",
        labels,
        delimiter=" ",
        fmt="%d",
    )


def write_datasets(root_folder):
    for dataset_index, (dataset_name, (dataset, algo_params)) in enumerate(
        zip(datasets_names, datasets_params)
    ):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        X, y = dataset
        X, y = normalize_dataset(
            X,
            y,
            scale=SCALE_MULTIPLIER,
            shift=SHIFT,
        )
        save_dataset(
            X,
            y,
            root_folder=root_folder,
            dataset_name=dataset_name,
        )

    # make a last dataset with unbalanced clusters densities
    X, _ = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    X, y = normalize_dataset_with_imbalances(X)
    save_dataset(
        X,
        y,
        root_folder=root_folder,
        dataset_name="unbalanced_blobs",
    )

    X, y = datasets.fetch_openml(
        "mnist_784", return_X_y=True, parser="auto", as_frame=False
    )
    X, y = normalize_mnist(X[:n_samples_mnist, :], y[:n_samples_mnist])

    save_dataset(
        X,
        y,
        root_folder=root_folder,
        dataset_name="mnist",
    )

    X, y = datasets.load_iris(return_X_y=True, as_frame=False)
    X, y = normalize_dataset(X, y, scale=SCALE_MULTIPLIER, shift=SHIFT)
    save_dataset(
        X,
        y,
        root_folder=root_folder,
        dataset_name="iris",
    )


if __name__ == "__main__":
    # ref lib: https://scikit-learn.org/stable/modules/classes.html
    # load datasets: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py

    # where to place the datasets
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "clustering/"
    )
    write_datasets(root_folder=root_folder)
