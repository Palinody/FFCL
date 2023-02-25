#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/kmeans/KMeansPlusPlus.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

namespace kmeans::utils {

template <typename Iterator>
std::vector<std::size_t> samples_to_nearest_centroid_indices(
    const Iterator&                                   data_first,
    const Iterator&                                   data_last,
    std::size_t                                       n_features,
    const std::vector<typename Iterator::value_type>& centroids) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples   = common::utils::get_n_samples(data_first, data_last, n_features);
    const std::size_t n_centroids = centroids.size() / n_features;

    // contains the indices from each sample to the nearest centroid
    auto nearest_centroid_indices = std::vector<std::size_t>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // distance buffer for a given data sample to each cluster
        auto        min_distance = common::utils::infinity<DataType>();
        std::size_t min_index    = 0;

        for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
            // sqrt(sum((a_j - b_j)²))
            const DataType sample_to_centroid_distance =
                cpp_clustering::heuristic::heuristic(data_first + sample_index * n_features,
                                                     data_first + sample_index * n_features + n_features,
                                                     centroids.begin() + centroid_index * n_features);

            if (sample_to_centroid_distance < min_distance) {
                min_distance = sample_to_centroid_distance;
                min_index    = centroid_index;
            }
        }
        nearest_centroid_indices[sample_index] = min_index;
    }
    return nearest_centroid_indices;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_nearest_centroid_distances(
    const Iterator&                                   data_first,
    const Iterator&                                   data_last,
    std::size_t                                       n_features,
    const std::vector<typename Iterator::value_type>& centroids) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples   = common::utils::get_n_samples(data_first, data_last, n_features);
    const std::size_t n_centroids = centroids.size() / n_features;

    // contains the distances from each sample to the nearest centroid
    auto nearest_centroid_distances = std::vector<DataType>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto min_distance = common::utils::infinity<DataType>();

        for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
            const auto nearest_candidate =
                cpp_clustering::heuristic::heuristic(data_first + sample_index * n_features,
                                                     data_first + sample_index * n_features + n_features,
                                                     centroids.begin() + centroid_index * n_features);

            if (nearest_candidate < min_distance) {
                min_distance = nearest_candidate;
            }
        }
        nearest_centroid_distances[sample_index] = min_distance;
    }
    return nearest_centroid_distances;
}

/*
template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_nearest_centroid_distances(
    const Iterator&                                   data_first,
    const Iterator&                                   data_last,
    std::size_t                                       n_features,
    const std::vector<typename Iterator::value_type>& centroids) {
    using DataType = typename Iterator::value_type;

    const auto nearest_centroid_indices =
        samples_to_nearest_centroid_indices(data_first, data_last, n_features, centroids);

    std::vector<DataType> nearest_centroid_distances(nearest_centroid_indices.size());

    for (std::size_t sample_index = 0; sample_index < nearest_centroid_indices.size(); ++sample_index) {
        const std::size_t centroid_index = nearest_centroid_indices[sample_index];

        const auto distance = cpp_clustering::heuristic::heuristic(data_first + sample_index * n_features,
                                                                   data_first + sample_index * n_features + n_features,
                                                                   centroids.begin() + centroid_index * n_features);
        nearest_centroid_distances[sample_index] = distance;
    }

    return nearest_centroid_distances;
}
*/

}  // namespace kmeans::utils