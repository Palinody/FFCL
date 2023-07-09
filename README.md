# FFCL

**FFCL**: **F**lexible and (probably not the) **f**ast(est) c++ **c**lustering **l**ibrary. 

This can be used as a header only library by dumping `include/*` in your project.
`KDTree`s use `rapidjson` to save data and its the only dependency. So I just dumped the `rapidjson` headers in `include/rapidjson`.
It'll stay that way for now for practicality.

- [Current features](#current-features)
- [Performance](#performance)
- [Installation](#installation)
- [Compiling](#compiling)
- [How to use](#how-to-use)

## TODO | New features | Updates

- **TODO**
  - [Faster DBSCAN](https://alexandria.tue.nl/extra1/afstversl/wsk-i/gunawan2013.pdf) | [DBSCAN++](https://arxiv.org/pdf/1810.13105.pdf)
  - performance tests comaring `ffcl` to the most popular libraries
  - Some optimization for the nearest neighbors computations of KMeans and KMedoids. I might enable KDTree acceleration but its not beneficial unless a large number of centroids/medoids is needed. Its not a priority for now.
  - Proper unit testing (**update**: all the generic code is now unit tested)
  - OPTICS
  - DENCLUE

- **New features**
  - DBSCAN

## Current features

- ### DBSCAN (kdtree)

  - DBSCAN [original paper](https://www2.cs.uh.edu/~ceick/7363/Papers/dbscan.pdf) | [used algorithm (sequential only)](https://arxiv.org/pdf/2103.05162.pdf)

- ### KMedoids

  - FasterMSC [paper](https://arxiv.org/pdf/2209.12553.pdf) | [author's repo](https://github.com/kno10/rust-kmedoids)
  - FasterPAM [paper](https://arxiv.org/pdf/2008.05171.pdf) | [author's repo](https://github.com/kno10/rust-kmedoids)

- ### KMeans

  - Lloyd
  - Hamerly [paper](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12) | [authors' repo](https://github.com/ghamerly/fast-kmeans)


- ### Containers
  - `KDTreeIndexed` is currently used for `DBSCAN`
    - methods using an index pointing to one of the samples of the input dataset
      - nearest_neighbor_around_query_index
      - k_nearest_neighbors_around_query_index
      - radius_count_around_query_index
      - radius_search_around_query_index
      - range_count_around_query_index
      - range_search_around_query_index
    - methods using a range pointing to the beginning and end of a new sample query that may not exist in the input dataset
      - nearest_neighbor_around_query_sample
      - k_nearest_neighbors_around_query_sample
      - radius_count_around_query_sample
      - radius_search_around_query_sample
      - range_count_around_query_sample
      - range_search_around_query_sample
    - Options
      - `bucket_size(ssize_t)`: the maximum number of samples that a leaf node can contain
      - `max_depth(ssize_t)`: the maximum recursion depth of the tree. The `max_depth` takes priority over `bucket_size`, meaning that if the maximum recursion depth cannot be satisfied without violating the bucket size condition, the latter may not be fulfilled.
      - `axis_selection_policy`: `IndexedHighestVarianceBuild` (default) or `IndexedMaximumSpreadBuild`. Custom policies can be implmented and used.
      - `feature_mask({feature_index_0, feature_index_1, ..., feature_index_n})`: dynamic feature mask used the select the features of interest for the axis selection procedure during the build phase. The feature indices can be specified in any order, as long as `0 <= feature_index_j < n_features`. Duplicates feature indices should result in valid results but it would bias the axis selection procedure and thus defeat its purpose.
      - `splitting_rule_policy`: `IndexedQuickselectMedianRange` (default): quickselect median selection strategy for partitioning the children leaves around the pivot median value of the binary tree.
  - `KDTree`: kd-tree without index but with much less features than `KDTreeIndexed`. It was just implemented for experiments purposes against `KDTreeIndexed` but it doesnt seem to have anything better other than no index creation overhead.
  - `LowerTriangleMatrix` currently used for the pairwise distance values that can be buffered for `KMedoids`.

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

### DBSCAN

#### Practical example using one of the autogenerated datasets

```c
fs::path filename = "unbalanced_blobs.txt";

// flattened vector: std::vector<dType>
std::vector<dType> data = load_data<dType>(inputs_folder_ / filename, ' ');
// the number of features is needed to use the flattened vector
const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);
// the vector of indices that will be rearranged by the indexer: {0, 1, ..., n_samples-1}
auto indices = generate_indices(n_samples);

/*
unfortunately those aliases will be needed if you want to explicitly specify axis selection or splitting rule algorithms as options.
The default ones are currently based on a quickselect median selection strategy for the split and the highest variance with a sampling rate of 0.1 for the axis selection.
You can also implement your own custom policies.

axis_selection_policy: IndexedHighestVarianceBuild (default), IndexedMaximumSpreadBuild, IndexedCycleThroughAxesBuild
splitting_rule_policy: IndexedQuickselectMedianRange (default)
*/
using IndicesIterator         = decltype(indices)::iterator;
using SamplesIterator         = decltype(data)::iterator;
using OptionsType             = ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>::Options;
using AxisSelectionPolicyType = kdtree::policy::IndexedMaximumSpreadBuild<IndicesIterator, SamplesIterator>;
using SplittingRulePolicyType = kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator SamplesIterator>;


// KDTreeIndexed will only rearrange the 'indices' container. The 'data' container remains unchanged.
auto kdtree = ffcl::containers::KDTreeIndexed(indices.begin(),
                                              indices.end(),
                                              data.begin(),
                                              data.end(),
                                              n_features,
                                              OptionsType()
                                                  .bucket_size(std::sqrt(n_samples))
                                                  .max_depth(std::log2(n_samples))
                                                  .axis_selection_policy(AxisSelectionPolicyType().feature_mask({0, 1}))
                                                  .splitting_rule_policy(SplittingRulePolicyType())


auto dbscan = ffcl::DBSCAN<dType>();
// radius: 'epsilon' in DBSCAN paper
// min_samples_in_radius: 'minPts' in DBSCAN paper
dbscan.set_options(ffcl::DBSCAN<dType>::Options().radius(2).min_samples_in_radius(5));
// returns a vector of labels that describe each sample of the data described by the following vector
// label[sample_index] <-> std::vector(data.begin() + sample_index * n_features, data.begin() + sample_index * n_features + n_features]
const auto predictions = dbscan.predict(kdtree);
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