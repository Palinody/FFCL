# cpp_clustering
Fast clustering algorithms written in c++

- [Current algorithms](#current-algorithms)
- [Performance](#performance)
- [Installation](#installation)
- [Compiling](#compiling)
- [How to use](#how-to-use)

## Current features

- ### KMedoids:
  - FasterMSC (with(out) pairwise distance matrix) [paper](https://arxiv.org/pdf/2209.12553.pdf)
  - FasterPAM (with(out) pairwise distance matrix) [paper](https://arxiv.org/pdf/2008.05171.pdf)

The implementation of the papers was helped by one of the [author's repo](https://github.com/kno10/rust-kmedoids)

- ### KMeans:
  - Lloyd

- ### Distance functions
  - euclidean
  - manhattan
  - cosine similarity

- ### initialization:
  - random uniform
  - spatial uniform
  - pam::build (with(out) pairwise distance matrix)
  - kmeans++

## Performance

`<...>`

## Installation
```sh
cd <your_repo>
git clone <this_repo>
mkdir build
cd build
```

## Compiling
### 1) As a library
```sh
cmake ..
build
```
This will create a `libcpp_clustering.a` file in the `build/` firectory

### 2) To run the benchmarks

Generate the datasets automatically:
```
python3 ../datasets/datasets_maker.py
```
This will create the following files in `../datasets/clustering/`
```sh
.
├── inputs
│   ├── aniso.txt
│   ├── blobs.txt
│   ├── iris.txt
│   ├── mnist.txt
│   ├── noisy_circles.txt
│   ├── noisy_moons.txt
│   ├── no_structure.txt
│   ├── unbalanced_blobs.txt
│   └── varied.txt
└── targets
    ├── aniso.txt
    ├── blobs.txt
    ├── iris.txt
    ├── mnist.txt
    ├── noisy_circles.txt
    ├── noisy_moons.txt
    ├── no_structure.txt
    ├── unbalanced_blobs.txt
    └── varied.txt
```

The average default number of samples for the datasets is ~2000 samples. This can be changed in the `datasets_maker.py` script. MNIST can have up to 70000 samples and 784 features. Setting `n_samples_mnist=None` results in loading mnist with the maximum number of samples.

Compile (this can also be done before generating the datasets except running the benchmarks)

```sh
cmake .. -DMODE="benchmark"
build
```
This will create a `cpp_clustering` executable in the `build` folder. To run the benchmarks, simply:
```
./cpp_clustering
```
The results will be written in `../datasets/clustering/` and the folder structure might now look like this:
```sh
├── centroids
│   ├── aniso.txt
<...>
│   └── varied.txt
├── inputs
│   ├── aniso.txt
<...>
│   └── varied.txt
├── predictions
│   ├── aniso.txt
<...>
│   └── varied.txt
└── targets
    ├── aniso.txt
<...>
    └── varied.txt
```
The clusters can be visualised:
```sh
python3 ../datasets/plot.py
```

### 3) To run the unit tests
```sh
cmake .. -DMODE="gtest"
build
```
Run the gtests:
```sh
ctest -V
```

### \*) All cmake options (the ones with a (\*) can be accumulated)
- `-DMODE="gtest"`
- `-DMODE="benchmark"`
- `-DVERBOSE=true` (\*)
- `-DTHREADS_ENABLED=true` (\*)

## How to use

`<...>`
