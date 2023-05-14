#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/random/Distributions.hpp"

#include <algorithm>
#include <vector>

namespace math::random {

/**
 * @brief might have overlaps when n_choices is close to the range size
 *
 * @param n_choices the number of indices that will be selected
 * @param indices_range the pool of indices to choose from
 * @return std::vector<std::size_t>
 */
inline std::vector<std::size_t> select_from_range(std::size_t                                n_choices,
                                                  const std::pair<std::size_t, std::size_t>& indices_range) {
    assert(indices_range.second - indices_range.first >= n_choices &&
           "The number of random choice indices should be less or equal than the indices candidates.");
    // the unique indices
    std::vector<std::size_t> random_distinct_indices;
    // keeps track of the indices that have already been generated as unique objects
    std::unordered_set<std::size_t> generated_indices;
    // range: [0, n_indices-1], upper bound included
    math::random::uniform_distribution<std::size_t> random_number_generator(indices_range.first,
                                                                            indices_range.second - 1);

    while (random_distinct_indices.size() < n_choices) {
        const auto index_candidate = random_number_generator();
        // check if the index candidate is already in the set and adds it to both containers if not
        if (generated_indices.find(index_candidate) == generated_indices.end()) {
            random_distinct_indices.emplace_back(index_candidate);
            generated_indices.insert(index_candidate);
        }
    }
    return random_distinct_indices;
}

/**
 * @brief not recommended for very large ranges since it will create a buffer for it that might take this amount of
 * memory but it wont have overlaps for n_choices ~= n_indices_candidates
 *
 * @param n_choices the number of indices that will be selected
 * @param indices_range the pool of indices to choose from
 * @return std::vector<std::size_t>
 */
inline std::vector<std::size_t> select_from_range_buffered(std::size_t                                n_choices,
                                                           const std::pair<std::size_t, std::size_t>& indices_range) {
    // indices_range upper bound excluded
    std::size_t n_indices_candidates = indices_range.second - indices_range.first;

    assert(n_indices_candidates >= n_choices &&
           "The number of random choice indices should be less or equal than the indices candidates.");

    // the unique indices
    std::vector<std::size_t> random_distinct_indices(n_choices);
    // generate the initial indices sequence which elements will be drawn from
    std::vector<std::size_t> initial_indices_candidates(n_indices_candidates);
    std::iota(initial_indices_candidates.begin(), initial_indices_candidates.end(), indices_range.first);

    for (auto& selected_index : random_distinct_indices) {
        // range: [0, N-1], upper bound is included
        math::random::uniform_distribution<std::size_t> random_number_generator(0,
                                                                                initial_indices_candidates.size() - 1);
        // generate the index of the indices vector
        const auto index_index = random_number_generator();
        // get the actual value
        const auto index_value = initial_indices_candidates[index_index];
        // save the index value
        selected_index = index_value;
        // and remove it from the candidates to make it unavalable for ther next iteration
        initial_indices_candidates.erase(initial_indices_candidates.begin() + index_index);
    }
    return random_distinct_indices;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> select_random_sample(const Iterator& data_first,
                                                                const Iterator& data_last,
                                                                std::size_t     n_features) {
    using DataType = typename Iterator::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);
    // selects an index w.r.t. an uniform random distribution [0, n_samples)
    auto index_select = math::random::uniform_distribution<std::size_t>(0, n_samples - 1);
    // pick the initial index that represents the first cluster
    const std::size_t random_index = index_select();

    return std::vector<DataType>(data_first + random_index * n_features,
                                 data_first + random_index * n_features + n_features);
}

template <typename Iterator>
std::vector<typename Iterator::value_type> select_n_random_samples(const Iterator& data_first,
                                                                   const Iterator& data_last,
                                                                   std::size_t     n_features,
                                                                   std::size_t     n_choices) {
    using DataType = typename Iterator::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);
    // clip n_choices to prevent overflows
    n_choices = std::min(n_choices, n_samples);
    // return n_choices distinctive indices from the pool of indices defined by the desired indices range
    const auto random_distinct_indices = select_from_range(n_choices, {0, n_samples});

    auto random_samples = std::vector<DataType>(n_choices * n_features);

    for (std::size_t sample_index = 0; sample_index < n_choices; ++sample_index) {
        std::copy(data_first + random_distinct_indices[sample_index] * n_features,
                  data_first + random_distinct_indices[sample_index] * n_features + n_features,
                  random_samples.begin() + sample_index * n_features);
    }
    return random_samples;
}

template <typename RandomAccessIntIterator, typename SamplesIterator>
std::vector<typename SamplesIterator::value_type> select_n_random_samples_from_indices(
    const RandomAccessIntIterator& index_first,
    const RandomAccessIntIterator& index_last,
    const SamplesIterator&         data_first,
    const SamplesIterator&         data_last,
    std::size_t                    n_features,
    std::size_t                    n_choices) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(data_last);

    const std::size_t n_samples = std::distance(index_first, index_last);
    // clip n_choices to prevent overflows
    n_choices = std::min(n_choices, n_samples);
    // return n_choices distinctive indices from the pool of indices defined by the desired indices range
    const auto random_distinct_indices = select_from_range(n_choices, {0, n_samples});

    auto random_samples = std::vector<DataType>(n_choices * n_features);

    for (std::size_t sample_index = 0; sample_index < n_choices; ++sample_index) {
        std::copy(data_first + index_first[random_distinct_indices[sample_index]] * n_features,
                  data_first + index_first[random_distinct_indices[sample_index]] * n_features + n_features,
                  random_samples.begin() + sample_index * n_features);
    }
    return random_samples;
}

template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> init_spatial_uniform(const IteratorFloat& data_first,
                                                                     const IteratorFloat& data_last,
                                                                     std::size_t          n_centroids,
                                                                     std::size_t          n_features) {
    using FloatType = typename IteratorFloat::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    auto centroids = std::vector<FloatType>(n_centroids * n_features);

    // row vector buffers to keep track of min-max values
    auto min_buffer = std::vector<FloatType>(n_features, std::numeric_limits<FloatType>::max());
    auto max_buffer = std::vector<FloatType>(n_features, std::numeric_limits<FloatType>::min());
    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            const FloatType curr_elem = *(data_first + j + i * n_features);
            min_buffer[j]             = std::min(curr_elem, min_buffer[j]);
            max_buffer[j]             = std::max(curr_elem, max_buffer[j]);
        }
    }
    using uniform_distr_ptr = typename std::unique_ptr<math::random::uniform_distribution<FloatType>>;
    // initialize a uniform random generatore w.r.t. each feature
    auto random_buffer = std::vector<uniform_distr_ptr>(n_features);
    for (std::size_t f = 0; f < n_features; ++f) {
        random_buffer[f] =
            std::make_unique<math::random::uniform_distribution<FloatType>>(min_buffer[f], max_buffer[f]);
    }
    for (std::size_t k = 0; k < n_centroids; ++k) {
        for (std::size_t f = 0; f < n_features; ++f) {
            // generate a centroid position that lies in the [min, max] range
            centroids[f + k * n_features] = (*random_buffer[f])();
        }
    }
    return centroids;
}

template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> init_uniform(const IteratorFloat& data_first,
                                                             const IteratorFloat& data_last,
                                                             std::size_t          n_centroids,
                                                             std::size_t          n_features) {
    using FloatType = typename IteratorFloat::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    auto centroids = std::vector<FloatType>(n_centroids * n_features);

    const auto indices = select_from_range(n_centroids, {0, n_samples});

    for (std::size_t k = 0; k < n_centroids; ++k) {
        const auto idx = indices[k];
        for (std::size_t f = 0; f < n_features; ++f) {
            centroids[k * n_features + f] = *(data_first + idx * n_features + f);
        }
    }
    return centroids;
}

}  // namespace math::random