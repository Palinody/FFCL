#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/heuristics/Heuristics.hpp"

namespace kmeans::utils {

template <typename Iterator>
std::vector<std::size_t> samples_to_nearest_centroid_indices(
    const Iterator&                                   samples_first,
    const Iterator&                                   samples_last,
    std::size_t                                       n_features,
    const std::vector<typename Iterator::value_type>& centroids) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples   = common::utils::get_n_samples(samples_first, samples_last, n_features);
    const std::size_t n_centroids = centroids.size() / n_features;

    // contains the indices from each sample to the nearest centroid
    auto nearest_centroid_indices = std::vector<std::size_t>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // distance buffer for a given data sample to each cluster
        auto        min_distance = common::utils::infinity<DataType>();
        std::size_t min_index    = 0;

        for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
            const DataType sample_to_centroid_distance =
                ffcl::heuristic::heuristic(samples_first + sample_index * n_features,
                                           samples_first + sample_index * n_features + n_features,
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
    const Iterator&                                   samples_first,
    const Iterator&                                   samples_last,
    std::size_t                                       n_features,
    const std::vector<typename Iterator::value_type>& centroids) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples   = common::utils::get_n_samples(samples_first, samples_last, n_features);
    const std::size_t n_centroids = centroids.size() / n_features;

    // contains the distances from each sample to the nearest centroid
    auto nearest_centroid_distances = std::vector<DataType>(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto min_distance = common::utils::infinity<DataType>();

        for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
            const auto nearest_candidate =
                ffcl::heuristic::heuristic(samples_first + sample_index * n_features,
                                           samples_first + sample_index * n_features + n_features,
                                           centroids.begin() + centroid_index * n_features);

            if (nearest_candidate < min_distance) {
                min_distance = nearest_candidate;
            }
        }
        nearest_centroid_distances[sample_index] = min_distance;
    }
    return nearest_centroid_distances;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> samples_to_second_nearest_centroid_distances(
    const Iterator&                                   samples_first,
    const Iterator&                                   samples_last,
    std::size_t                                       n_features,
    const std::vector<typename Iterator::value_type>& centroids) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_samples   = common::utils::get_n_samples(samples_first, samples_last, n_features);
    const std::size_t n_centroids = centroids.size() / n_features;

    // the vector that will contain the distances from each sample to the nearest centroid
    auto second_nearest_centroid_distances = std::vector<DataType>(n_samples);

    // iterate over all the samples
    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto first_min_distance  = std::numeric_limits<DataType>::max();
        auto second_min_distance = std::numeric_limits<DataType>::max();
        // iterate over the centroids indices
        for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
            const auto second_nearest_candidate =
                ffcl::heuristic::heuristic(samples_first + sample_index * n_features,
                                           samples_first + sample_index * n_features + n_features,
                                           centroids.begin() + centroid_index * n_features);

            if (second_nearest_candidate < first_min_distance) {
                second_min_distance = first_min_distance;
                first_min_distance  = second_nearest_candidate;

            } else if (second_nearest_candidate < second_min_distance &&
                       common::utils::inequality(second_nearest_candidate, first_min_distance)) {
                second_min_distance = second_nearest_candidate;
            }
        }
        // assign the current sample with its nth nearest centroid distance
        second_nearest_centroid_distances[sample_index] = second_min_distance;
    }
    return second_nearest_centroid_distances;
}

template <typename IteratorInt>
std::vector<std::size_t> compute_cluster_sizes(const IteratorInt& samples_to_nearest_centroid_indices_first,
                                               const IteratorInt& samples_to_nearest_centroid_indices_last,
                                               std::size_t        n_centroids) {
    static_assert(std::is_integral_v<typename IteratorInt::value_type>, "Input elements type should be integral.");

    // the number of samples associated to each centroids
    auto cluster_sizes = std::vector<std::size_t>(n_centroids);

    for (auto centroid_index_iter = samples_to_nearest_centroid_indices_first;
         common::utils::inequality(centroid_index_iter, samples_to_nearest_centroid_indices_last);
         ++centroid_index_iter) {
        ++cluster_sizes[*centroid_index_iter];
    }
    return cluster_sizes;
}

template <typename Iterator, typename IteratorInt>
std::vector<typename Iterator::value_type> compute_cluster_positions_sum(
    const Iterator&    samples_first,
    const Iterator&    samples_last,
    const IteratorInt& samples_to_nearest_centroid_indices_first,
    std::size_t        n_centroids,
    std::size_t        n_features) {
    static_assert(std::is_integral_v<typename IteratorInt::value_type>, "Input elements type should be integral.");

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    // accumulate the positions of each sample in each cluster
    auto cluster_positions_sum = std::vector<typename Iterator::value_type>(n_centroids * n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        const auto assigned_centroid_index = *(samples_to_nearest_centroid_indices_first + sample_index);

        std::transform(cluster_positions_sum.begin() + assigned_centroid_index * n_features,
                       cluster_positions_sum.begin() + assigned_centroid_index * n_features + n_features,
                       samples_first + sample_index * n_features,
                       cluster_positions_sum.begin() + assigned_centroid_index * n_features,
                       std::plus<>());
    }
    return cluster_positions_sum;
}

template <typename Iterator>
std::vector<typename Iterator::value_type> nearest_neighbor_distances(const Iterator& data_first,
                                                                      const Iterator& data_last,
                                                                      std::size_t     n_features) {
    using DataType = typename Iterator::value_type;

    const std::size_t n_rows = common::utils::get_n_samples(data_first, data_last, n_features);

    // contains the distances from each data d_i to its nearest data d_j with i != j
    auto neighbor_distances = std::vector<DataType>(n_rows);

    for (std::size_t row_index = 0; row_index < n_rows; ++row_index) {
        auto min_distance = common::utils::infinity<DataType>();

        for (std::size_t other_row_index = 0; other_row_index < n_rows; ++other_row_index) {
            if (row_index != other_row_index) {
                const auto nearest_candidate =
                    ffcl::heuristic::heuristic(data_first + row_index * n_features,
                                               data_first + row_index * n_features + n_features,
                                               data_first + other_row_index * n_features);

                if (nearest_candidate < min_distance) {
                    min_distance = nearest_candidate;
                }
            }
        }
        neighbor_distances[row_index] = min_distance;
    }
    return neighbor_distances;
}

}  // namespace kmeans::utils