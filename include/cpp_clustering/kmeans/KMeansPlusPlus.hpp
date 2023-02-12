#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"
#include "cpp_clustering/math/random/VosesAliasMethod.hpp"

#include <algorithm>  // std::minmax_element
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>  // std::numeric_limits<T>::max()
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace cpp_clustering::kmeansplusplus {

namespace private_ {

template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> compute_euclidean_distances(IteratorFloat data_first,
                                                                            IteratorFloat data_last,
                                                                            IteratorFloat sample_first,
                                                                            std::size_t   n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");

    using FloatType = typename IteratorFloat::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    auto distances = std::vector<FloatType>(n_samples);

    for (std::size_t i = 0; i < n_samples; ++i) {
        distances[i] = cpp_clustering::heuristic::faster_euclidean_distance(
            data_first + i * n_features, data_first + i * n_features + n_features, sample_first);
    }
    return distances;
}

template <typename T = double, typename IteratorFloat>
std::vector<T> make_distances_scores(IteratorFloat distance_first, IteratorFloat distance_last) {
    const auto [min, max] = std::minmax_element(distance_first, distance_last);

    if (*min == *max) {
        const std::size_t n_elements = std::distance(distance_first, distance_last);
        // if all the values are the same distribute the weights equaly
        return std::vector<T>(n_elements, 1.0 / n_elements);
    }
    auto res = std::vector<T>(distance_last - distance_first);
    // closest objects get a higher score. Distance zero -> 1
    std::transform(distance_first, distance_last, res.begin(), [&min, &max](const auto& dist) {
        return static_cast<T>(1) - (dist - *min) / (*max - *min);
    });
    return res;
}

}  // namespace private_

template <typename IteratorFloat>
std::vector<typename IteratorFloat::value_type> make_centroids(IteratorFloat data_first,
                                                               IteratorFloat data_last,
                                                               std::size_t   n_centroids,
                                                               std::size_t   n_features) {
    static_assert(std::is_floating_point<typename IteratorFloat::value_type>::value,
                  "Data should be a floating point type.");

    using FloatType = typename IteratorFloat::value_type;

    const auto n_samples = common::utils::get_n_samples(data_first, data_last, n_features);

    // make a copy of the data as candidates
    auto remaining_candidates = std::vector<FloatType>(n_samples * n_features);
    std::copy(data_first, data_last, remaining_candidates.begin());

    // indices buffer that keeps track of the available indices (1 pops per iteration)
    auto indices_buf = std::vector<std::size_t>(n_samples);
    // initialize indices
    std::iota(indices_buf.begin(), indices_buf.end(), 0);

    // selects an index w.r.t. an uniform random distribution [0, n_samples)
    math::random::uniform_distribution<std::size_t> index_select(0, n_samples - 1);
    // pick the initial index that represents the first cluster
    std::size_t sample_idx = index_select();
    // buffer_idx is the index of the buffer that keeps track of the available indices
    std::size_t buffer_idx = sample_idx;

    // make a placeholder container for the randomly selected centroids
    auto centroids = std::vector<FloatType>(n_centroids * n_features);
    // save the current chosen cluster
    std::copy(/*source begin*/ data_first + sample_idx * n_features,
              /*source end*/ data_first + sample_idx * n_features + n_features,
              /*destination begin*/ centroids.begin());

    for (std::size_t k = 1; k < n_centroids; ++k) {
        // remove the randomly chosen sample from the candidates
        remaining_candidates.erase(remaining_candidates.begin() + buffer_idx * n_features,
                                   remaining_candidates.begin() + buffer_idx * n_features + n_features);

        // remove the chosen index from the indices buffer
        indices_buf.erase(indices_buf.begin() + buffer_idx);

        // compute the distance of the remaining candidates with the current chosen centroid
        auto distances_buffer = private_::compute_euclidean_distances<IteratorFloat>(
            remaining_candidates.begin(),
            remaining_candidates.end(),
            /*current_centroid=*/data_first + sample_idx * n_features,
            n_features);

        auto weights      = private_::make_distances_scores<double>(distances_buffer.begin(), distances_buffer.end());
        auto alias_method = math::random::VosesAliasMethod(weights);
        // update buffer index which value ranges w.r.t. current number of candidates
        buffer_idx = alias_method();
        // retrieve the global index according to the original data container
        sample_idx = indices_buf[buffer_idx];

        // save the current chosen cluster
        std::copy(/*source begin*/ data_first + sample_idx * n_features,
                  /*source end*/ data_first + sample_idx * n_features + n_features,
                  /*destination begin*/ centroids.begin() + k * n_features);
    }
    return centroids;
}

}  // namespace cpp_clustering::kmeansplusplus