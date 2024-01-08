#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/random/Distributions.hpp"

#include <algorithm>
#include <vector>

namespace ffcl::common::math::random {

// https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
inline std::vector<std::size_t> select_from_range(std::size_t                                n_choices,
                                                  const std::pair<std::size_t, std::size_t>& indices_range) {
    const std::size_t range_size = indices_range.second - indices_range.first;

    assert(indices_range.first <= indices_range.second &&
           "The first value of the indices range should be less or equal than the second.");

    assert(range_size >= n_choices &&
           "The number of random choice indices should be less or equal than the indices candidates.");

    std::vector<std::size_t> indices(range_size);
    // Initialize indices with sequential values
    std::iota(indices.begin(), indices.end(), indices_range.first);

    // Perform a partial shuffle of the first n_choices elements
    for (std::size_t index = 0; index < n_choices; ++index) {
        // range: [i, n_indices-1], upper bound included
        std::swap(indices[index], indices[uniform_distribution<std::size_t>{index, n_choices - 1}()]);
    }
    // Resize the vector to keep only the first n_choices elements
    indices.resize(n_choices);

    return indices;
}

template <typename SamplesIterator>
auto select_random_sample(const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_features)
    -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);
    // selects an index w.r.t. an uniform random distribution [0, n_samples)
    auto index_select = uniform_distribution<std::size_t>(0, n_samples - 1);
    // pick the initial index that represents the first cluster
    const std::size_t random_index = index_select();

    return std::vector<DataType>(samples_range_first + random_index * n_features,
                                 samples_range_first + random_index * n_features + n_features);
}

template <typename SamplesIterator>
auto select_n_random_samples(const SamplesIterator& samples_range_first,
                             const SamplesIterator& samples_range_last,
                             std::size_t            n_features,
                             std::size_t            n_choices)
    -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const auto n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);
    // clip n_choices to prevent overflows
    n_choices = std::min(n_choices, n_samples);
    // return n_choices distinctive indices from the pool of indices defined by the desired indices range
    const auto random_distinct_indices = select_from_range(n_choices, {0, n_samples});

    auto random_samples = std::vector<DataType>(n_choices * n_features);

    for (std::size_t sample_index = 0; sample_index < n_choices; ++sample_index) {
        std::copy(samples_range_first + random_distinct_indices[sample_index] * n_features,
                  samples_range_first + random_distinct_indices[sample_index] * n_features + n_features,
                  random_samples.begin() + sample_index * n_features);
    }
    return random_samples;
}

template <typename IndicesIterator, typename SamplesIterator>
auto select_n_random_samples_from_indices(const IndicesIterator& index_first,
                                          const IndicesIterator& index_last,
                                          const SamplesIterator& samples_range_first,
                                          const SamplesIterator& samples_range_last,
                                          std::size_t            n_features,
                                          std::size_t            n_choices)
    -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(index_first, index_last);
    // clip n_choices to prevent overflows
    n_choices = std::min(n_choices, n_samples);
    // return n_choices distinctive indices from the pool of indices defined by the desired indices range
    const auto random_distinct_indices = select_from_range(n_choices, {0, n_samples});

    auto random_samples = std::vector<DataType>(n_choices * n_features);

    for (std::size_t sample_index = 0; sample_index < n_choices; ++sample_index) {
        std::copy(samples_range_first + index_first[random_distinct_indices[sample_index]] * n_features,
                  samples_range_first + index_first[random_distinct_indices[sample_index]] * n_features + n_features,
                  random_samples.begin() + sample_index * n_features);
    }
    return random_samples;
}

template <typename IndicesIterator>
auto select_n_random_indices_from_indices(const IndicesIterator& index_first,
                                          const IndicesIterator& index_last,
                                          std::size_t            n_choices) {
    using IndexType = typename std::iterator_traits<IndicesIterator>::value_type;

    const std::size_t n_samples = std::distance(index_first, index_last);
    // clip n_choices to prevent overflows
    n_choices = std::min(n_choices, n_samples);
    // return n_choices distinctive indices from the pool of indices defined by the desired indices range
    const auto random_distinct_indices = select_from_range(n_choices, {0, n_samples});

    auto random_indices = std::vector<IndexType>(n_choices);

    for (std::size_t index_index = 0; index_index < n_choices; ++index_index) {
        random_indices[index_index] = index_first[random_distinct_indices[index_index]];
    }
    return random_indices;
}

template <typename SamplesIterator>
auto init_spatial_uniform(const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_centroids,
                          std::size_t            n_features)
    -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    auto centroids = std::vector<DataType>(n_centroids * n_features);

    // row vector buffers to keep track of min-max values
    auto min_buffer = std::vector<DataType>(n_features, std::numeric_limits<DataType>::max());
    auto max_buffer = std::vector<DataType>(n_features, std::numeric_limits<DataType>::min());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto curr_elem      = samples_range_first[feature_index + sample_index * n_features];
            min_buffer[feature_index] = std::min(curr_elem, min_buffer[feature_index]);
            max_buffer[feature_index] = std::max(curr_elem, max_buffer[feature_index]);
        }
    }
    using UniformDistributionPtr = typename std::unique_ptr<uniform_distribution<DataType>>;
    // initialize a uniform random generatore w.r.t. each feature
    auto random_buffer = std::vector<UniformDistributionPtr>(n_features);
    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        random_buffer[feature_index] =
            std::make_unique<uniform_distribution<DataType>>(min_buffer[feature_index], max_buffer[feature_index]);
    }
    for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // generate a centroid position that lies in the [min, max] range
            centroids[feature_index + centroid_index * n_features] = (*random_buffer[feature_index])();
        }
    }
    return centroids;
}

template <typename SamplesIterator>
auto init_uniform(const SamplesIterator& samples_range_first,
                  const SamplesIterator& samples_range_last,
                  std::size_t            n_centroids,
                  std::size_t n_features) -> std::vector<typename std::iterator_traits<SamplesIterator>::value_type> {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    auto centroids = std::vector<DataType>(n_centroids * n_features);

    const auto indices = select_from_range(n_centroids, {0, n_samples});

    for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
        const auto index = indices[centroid_index];
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            centroids[centroid_index * n_features + feature_index] =
                *(samples_range_first + index * n_features + feature_index);
        }
    }
    return centroids;
}

}  // namespace ffcl::common::math::random