# cpp_clustering
Fast iterator compatible clustering algorithms written in c++.

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
  - pam::build
  - kmeans++

## Performance (only approx. for now)
- Cpu: `Intel® Core™ i5-9600KF CPU @ 3.70GHz × 6`
- Dataset: MNIST
- initialization: random uniform

|                         | **type**   | **n_samples** | **n_features**|**n_centroids**|**n_iter (until convergence)**| **time (s)**|**n_threads**|
------                    | -----   | ----- |--- |------|--- |--- |--- |
**FasterPAM**             | float       | 15000     |784|10|3|2.5|1|
**FasterPAM**             | float       | 15000     |784|100|3|3.8|1|
**FasterPAM**             | float       | 30000     |784|10|3|11.1|1|
**FasterPAM**             | float       | 30000     |784|100|3|13.2|1|
**FasterMSC**             | float       | 15000     |784|10|3|4.2|1|
**FasterMSC**             | float       | 15000     |784|100|3|5.8|1|
**FasterMSC**             | float       | 30000     |784|10|3|18.6|1|
**FasterMSC**             | float       | 30000     |784|100|3|27.2|1|
**PairwiseDistanceMatrix**| float       |   15000    |784|10||40|1|
**PairwiseDistanceMatrix**| float       |   30000    |784|10||162|1|
**PairwiseDistanceMatrix**| float       |   15000    |784|10||6.3|6|
**PairwiseDistanceMatrix**| float       |   30000    |784|10||26|6|

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
