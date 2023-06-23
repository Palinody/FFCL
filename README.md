# FFCL

**FFCL**: **F**lexible and (probably not the) **f**ast(est) c++ **c**lustering **l**ibrary. 

- [Current features](#current-features)
- [Performance](#performance)
- [Installation](#installation)
- [Compiling](#compiling)
- [How to use](#how-to-use)

## TODO \& Updates

- `ffcl::containers::KDTree`
  **update**: the build strategy seems to be faster than FLANN on mnist with variance build (10% sampling) and also with 2D datasets with the bounding box axis selection policy. The nearest neighbor search seems to be faster than FLANN on all the datasets. Nearest neighbor search is much more effective with kdtrees when n_features is relatively small and n_samples large. The speed is slightly worse with kdtree on MNIST (curiously not bad at all). The tests were made with a `bucket_size=40`.
    - kdtree build only with custom options: depth, bucket size, axis selection and split policies
    - multiple axis selection policies (client can make its own policies)
    - quickselect (handles duplicate data) splitting rule policy (client can make its own rules)
    - (single) nearest neighbor index with `KDTreeIndexed` (bug fixed) and `KDTree`. The nearest neighbor is an already existing point in the dataset that is designated by index. The function returns a remapped index that can be passed to the original unswapped dataset (for `KDTreeIndexed`) to retreive the neighbor sample. The same function has been implemented for `KDTree`. The nearest neighbor retrieval seems much faster in practice (both kdtrees) than FLANN but no in depth study has been made to guarantee that this is true, only quick tests by choosing various realistic hyperparameters. I also made the tests with FLANN using `result = flann.nn_index(query_point, k=1)`. I removed the queried sample from the original dataset to not call it since the nearest neighbor finds itself when querying an already existing point (see `BenchmarkKDTree.py`).
    - K-nearest neighbors indices for KDTreeIndexed and KDTree. The function outputs a pair of vectors containing the indices of the nearest neighbors according to the general index as well as the corresponding distances to the sample query index. The data is stored in reverse order (furthest to closest) because the nearest neighbors were stored in a priority queue with the furthest neighbor at the top of the queue. The data was "unrolled" from the top to the bottom of the queue sequentially at the very end of the k_nearest_neighbors function call. **The function ignores the query itself as a nearest neighbor. Meaning that a distance to the query cannot be zero, UNLESS another sample is at the exact same position as the query.** K-nearest neighbors has an overhead compared to the single neighbor version so the single version should be prefered if performance is important and a single nearest neighbor is needed.
    - `radius_count_around_query_index`. Counts the number of neighbors that are withing a given radius from a query sample index. **(Not tested enough)**: needs at least to be compared to the sequential algorithm.
    - `radius_search_around_query_index`. Finds the nearest neighbors inside of a radius centered at a query index. **(Not tested enough)**: needs at least to be compared to the sequential algorithm.
    - `nearest_neighbor_around_query_sample`, `k_nearest_neighbors_around_query_sample`, `radius_count_around_query_sample` and `radius_search_around_query_sample` are the same functions as the ones previously mentioned that work with an index query. They run faster since they dont have the overhead of checking whether the current iteration sample is the query but keep in mind that they will return the query itself as a first neighbor if its part of the dataset. So in this case for example `nearest_neighbor_around_query_sample` would be useless and `k_nearest_neighbors_around_query_sample` would also return the query. `radius_count_around_query_sample` would need to be subtracted by `1` and `radius_search_around_query_sample` would be subject to the same "issue" as `k_nearest_neighbors_around_query_sample`.
    - **NEW** All the core functions have been implemented for `KDTreeIndexed`. I might consider dropping `KDTree` support because there doesnt seem to be any performance gain and it would simplify the library a bit. Here's an exhaustive list of the current core functions that `KDTreeIndexed` supports: (using an index pointing to one of the samples of the input dataset) `nearest_neighbor_around_query_index`, `k_nearest_neighbors_around_query_index`, `radius_count_around_query_index`, `radius_search_around_query_index`, `range_count_around_query_index`, `range_search_around_query_index`, (using a range pointing to the beginning and end of a new sample query that may not exist in input dataset) `nearest_neighbor_around_query_sample`, `k_nearest_neighbors_around_query_sample`, `radius_count_around_query_sample`, `radius_search_around_query_sample`, `range_count_around_query_sample`, `range_search_around_query_sample`
    - **Next**: FEATURE MASK for kdtree construction. The kdtree will be built around target feature indices if an array of indices is specified.
- Proper unit testing (**update**: all the generic code is now unit tested)
- DBSCAN
- OPTICS
- DENCLUE

## Current features

- ### KMedoids

  - FasterMSC [paper](https://arxiv.org/pdf/2209.12553.pdf) | [author's repo](https://github.com/kno10/rust-kmedoids)
  - FasterPAM [paper](https://arxiv.org/pdf/2008.05171.pdf) | [author's repo](https://github.com/kno10/rust-kmedoids)

- ### KMeans

  - Lloyd
  - Hamerly [paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12) | [authors' repo](https://github.com/ghamerly/fast-kmeans)

- ### Distance functions

  - euclidean
  - manhattan
  - cosine similarity

- ### initialization

  - random uniform
  - spatial uniform
  - pam::build
  - kmeans++

- ### Selecting the number of clusters

  - silhouette method

## Performance

- Cpu: `Intel® Core™ i5-9600KF CPU @ 3.70GHz × 6`

### KMedoids algorithms

- Dataset: MNIST
- initialization: random uniform

The following table summarizes the results of single runs made with different parameters. It provides a rough estimate about what one could expect from the present library. For a more in depth analysis of the performance of the algorithms, refer to the articles [(1)](https://arxiv.org/pdf/2209.12553.pdf) and [(2)](https://arxiv.org/pdf/2008.05171.pdf)

|                         | **type**    |**n_samples**|**n_features**|**n_centroids**|**n_iter (converged)**| **n_threads**|**computation time (s)**|
------                    | -----       | -----       |---           |------         |---                   | ---          |-                       |
**FasterPAM**             | float       | 15,000      |784           | 10            |2                     |1             | 1.7                    |
**FasterPAM**             | float       | 15,000      |784           |100            |4                     |1             | 4.5                    |
**FasterPAM**             | float       | 30,000      |784           | 10            |3                     |1             |11.1                    |
**FasterPAM**             | float       | 30,000      |784           |100            |4                     |1             |  17                    |
**FasterMSC**             | float       | 15,000      |784           | 10            |2                     |1             | 3.3                    |
**FasterMSC**             | float       | 15,000      |784           |100            |2                     |1             | 6.4                    |
**FasterMSC**             | float       | 30,000      |784           | 10            |2                     |1             |12.8                    |
**FasterMSC**             | float       | 30,000      |784           |100            |3                     |1             |27.1                    |
**PairwiseDistanceMatrix**| float       | 15,000      |784           |               |                      |1             |  40                    |
**PairwiseDistanceMatrix**| float       | 30,000      |784           |               |                      |1             | 162                    |
**PairwiseDistanceMatrix**| float       | 15,000      |784           |               |                      |6             |6.27                    |
**PairwiseDistanceMatrix**| float       | 30,000      |784           |               |                      |6             |25.6                    |

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
make
```

This will create a `libcpp_clustering.a` file in the `build/` firectory

### 2) To run the benchmarks

Generate the datasets automatically:

```sh
python3 ../bin/MakeClusteringDatasets.py
```

This will create the following files in `../bin/clustering/`

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

The average default number of samples for the datasets is ~2000 samples. This can be changed in the `MakeClusteringDatasets.py` script. MNIST can have up to 70000 samples and 784 features. Setting `n_samples_mnist=None` results in loading mnist with the maximum number of samples.

Compile (this can also be done before generating the datasets except running the benchmarks)

```sh
cmake .. -DMODE="benchmark"
make
```

This will create a `ffcl` executable in the `build` folder. To run the benchmarks, simply:

```sh
./ffcl
```

The results will be written in `../bin/clustering/` and the folder structure might now look like this:

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
python3 ../bin/plot.py
```

### 3) To run the unit tests

```sh
cmake .. -DMODE="gtest"
make
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

### Silhouette method

#### Example: select the best number of centroids/medoids k in a range

```c
std::size_t k_min = 2;
std::size_t k_max = 10;

std::vector<float> scores(k_max - k_min);

// range n_centroids/n_medoids in [2, 10[
for (std::size_t k = k_min; k < k_max; ++k) {
    // use any clustering algorithm that better suits your use case
    common::clustering::KMeans<dType> kmeans(k, n_features);
    // fit the centroids (or medoids if it was KMedoids)
    kmeans.fit(data.begin(), data.end());
    // map the samples to their closest centroid/medoid
    const auto predictions = kmeans.predict(data.begin(), data.end());
    // compute the silhouette scores for each sample
    const auto samples_silhouette_values =
        math::heuristics::silhouette(data.begin(), data.end(), predictions.begin(), predictions.end(), n_features);

    // get the average score
    const auto mean_silhouette_coefficient = math::heuristics::get_mean_silhouette_coefficient(
        samples_silhouette_values.begin(), samples_silhouette_values.end());
    // accumulate the current scores
    scores[k - k_min] = mean_silhouette_coefficient;
}
// find the k corresponding to the number of centroids/medoids k with the best average silhouette score
const auto best_k = k_min + math::statistics::argmax(scores.begin(), scores.end())
```

### KMeans

#### Example with kmeans++ initialization

```c
using KMeans = ffcl::KMeans<float>;

const std::size_t n_features = 3;
// input_data.size() / n_features -> n_samples (4 in this case)
std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
// must be less than n_samples
const std::size_t n_centroids = 2;

const auto centroids_init = ffcl::kmeansplusplus::make_centroids(input_data.begin(), input_data.end(), n_centroids, n_features))
// initializing the centroids manually is optional
auto kmeans = KMeans(n_centroids, n_features, centroids_init);

const auto centroids = kmeans.fit(input_data.begin(), input_data.end());
```

### KMedoids

#### Simple example

```c
using KMedoids = ffcl::KMedoids<float>;

const std::size_t n_features = 3;
// input_data.size() / n_features -> n_samples (4 in this case)
std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
// must be less than n_samples
const std::size_t n_medoids = 2;

auto kmedoids = KMedoids(n_medoids, n_features);
// swap medoids and returns the centroids
const auto centroids = kmedoids.fit(input_data.begin(), input_data.end());
```

#### Complete example

```c
#include "ffcl/kmedoids/KMedoids.hpp"
// dType: type of the training samples
// true: PrecomputePairwiseDistanceMatrix (set to true by default)
using KMedoids = ffcl::KMedoids<SomeDataType, true>;

const std::size_t n_features = 3;
// input_data.size() / n_features -> n_samples (4 in this case)
std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
// each elements of the labels vector maps to a sample from input_data
std::vector<std::size_t> labels = {0, 0, 2, 1};
// 3 for class 0, 1 and 2 from the labels vector
const auto n_classes = 3;
// must be less than n_samples
const std::size_t n_medoids = 2;

auto kmedoids = KMedoids(n_medoids, n_features);
// set the options. The ones presented in this example are the same as the ones by default and provide no change
kmedoids.set_options(KMedoids::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

// Fit the data (inputs_first, inputs_last) with a PAM algorithm. Default: ffcl::FasterPAM
const auto centroids = kmedoids.fit<ffcl::FasterMSC>(input_data.begin(), input_data.end());

// map each data sample to its medoid
const std::vector<std::size_t> predictions = kmedoids.predict(input_data.begin(), input_data.end());
```
