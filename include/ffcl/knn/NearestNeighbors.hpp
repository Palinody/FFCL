#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/math/heuristics/Distances.hpp"

#include "ffcl/datastruct/BoundingBox.hpp"

#include "ffcl/knn/Buffer.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace ffcl::knn {

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor(const IndicesIterator&                indices_range_first,
                      const IndicesIterator&                indices_range_last,
                      const SamplesIterator&                samples_range_first,
                      const SamplesIterator&                samples_range_last,
                      std::size_t                           n_features,
                      std::size_t                           sample_index_query,
                      ssize_t&                              current_nearest_neighbor_index,
                      typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
                current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
                current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor(const IndicesIterator&                indices_range_first,
                      const IndicesIterator&                indices_range_last,
                      const SamplesIterator&                samples_range_first,
                      const SamplesIterator&                samples_range_last,
                      std::size_t                           n_features,
                      const SamplesIterator&                feature_query_range_first,
                      const SamplesIterator&                feature_query_range_last,
                      ssize_t&                              current_nearest_neighbor_index,
                      typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(feature_query_range_first,
                                            feature_query_range_last,
                                            samples_range_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
            current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
            current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors(const IndicesIterator&                                            indices_range_first,
                         const IndicesIterator&                                            indices_range_last,
                         const SamplesIterator&                                            samples_range_first,
                         const SamplesIterator&                                            samples_range_last,
                         std::size_t                                                       n_features,
                         std::size_t                                                       sample_index_query,
                         NearestNeighborsBufferBase<typename IndicesIterator::value_type,
                                                    typename SamplesIterator::value_type>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors(const IndicesIterator&                                            indices_range_first,
                         const IndicesIterator&                                            indices_range_last,
                         const SamplesIterator&                                            samples_range_first,
                         const SamplesIterator&                                            samples_range_last,
                         std::size_t                                                       n_features,
                         const SamplesIterator&                                            feature_query_range_first,
                         const SamplesIterator&                                            feature_query_range_last,
                         NearestNeighborsBufferBase<typename IndicesIterator::value_type,
                                                    typename SamplesIterator::value_type>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(feature_query_range_first,
                                            feature_query_range_last,
                                            samples_range_first + candidate_nearest_neighbor_index * n_features);

        nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius(const IndicesIterator&                      indices_range_first,
                                         const IndicesIterator&                      indices_range_last,
                                         const SamplesIterator&                      samples_range_first,
                                         const SamplesIterator&                      samples_range_last,
                                         std::size_t                                 n_features,
                                         std::size_t                                 sample_index_query,
                                         const typename SamplesIterator::value_type& radius,
                                         std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                ++neighbors_count;
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius(const IndicesIterator&                      indices_range_first,
                                         const IndicesIterator&                      indices_range_last,
                                         const SamplesIterator&                      samples_range_first,
                                         const SamplesIterator&                      samples_range_last,
                                         std::size_t                                 n_features,
                                         const SamplesIterator&                      feature_query_range_first,
                                         const SamplesIterator&                      feature_query_range_last,
                                         const typename SamplesIterator::value_type& radius,
                                         std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(feature_query_range_first,
                                            feature_query_range_last,
                                            samples_range_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_hyper_range(const IndicesIterator&                       indices_range_first,
                                              const IndicesIterator&                       indices_range_last,
                                              const SamplesIterator&                       samples_range_first,
                                              const SamplesIterator&                       samples_range_last,
                                              std::size_t                                  n_features,
                                              std::size_t                                  sample_index_query,
                                              const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                              std::size_t&                                 neighbors_count) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            bbox::is_sample_in_kd_bounding_box(
                samples_range_first + candidate_nearest_neighbor_index * n_features,
                samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_hyper_range(const IndicesIterator&                       indices_range_first,
                                              const IndicesIterator&                       indices_range_last,
                                              const SamplesIterator&                       samples_range_first,
                                              const SamplesIterator&                       samples_range_last,
                                              std::size_t                                  n_features,
                                              const SamplesIterator&                       feature_query_range_first,
                                              const SamplesIterator&                       feature_query_range_last,
                                              const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                              std::size_t&                                 neighbors_count) {
    common::utils::ignore_parameters(samples_range_last, feature_query_range_first, feature_query_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (bbox::is_sample_in_kd_bounding_box(
                samples_range_first + candidate_nearest_neighbor_index * n_features,
                samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius(
    const IndicesIterator&                      indices_range_first,
    const IndicesIterator&                      indices_range_last,
    const SamplesIterator&                      samples_range_first,
    const SamplesIterator&                      samples_range_last,
    std::size_t                                 n_features,
    std::size_t                                 sample_index_query,
    const typename SamplesIterator::value_type& radius,
    NearestNeighborsBufferBase<typename IndicesIterator::value_type, typename SamplesIterator::value_type>&
        nearest_neighbors_buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius(
    const IndicesIterator&                      indices_range_first,
    const IndicesIterator&                      indices_range_last,
    const SamplesIterator&                      samples_range_first,
    const SamplesIterator&                      samples_range_last,
    std::size_t                                 n_features,
    const SamplesIterator&                      feature_query_range_first,
    const SamplesIterator&                      feature_query_range_last,
    const typename SamplesIterator::value_type& radius,
    NearestNeighborsBufferBase<typename IndicesIterator::value_type, typename SamplesIterator::value_type>&
        nearest_neighbors_buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(feature_query_range_first,
                                            feature_query_range_last,
                                            samples_range_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_hyper_range(
    const IndicesIterator&                       indices_range_first,
    const IndicesIterator&                       indices_range_last,
    const SamplesIterator&                       samples_range_first,
    const SamplesIterator&                       samples_range_last,
    std::size_t                                  n_features,
    std::size_t                                  sample_index_query,
    const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    NearestNeighborsBufferBase<typename IndicesIterator::value_type, typename SamplesIterator::value_type>&
        nearest_neighbors_buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            bbox::is_sample_in_kd_bounding_box(
                samples_range_first + candidate_nearest_neighbor_index * n_features,
                samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_hyper_range(
    const IndicesIterator&                       indices_range_first,
    const IndicesIterator&                       indices_range_last,
    const SamplesIterator&                       samples_range_first,
    const SamplesIterator&                       samples_range_last,
    std::size_t                                  n_features,
    const SamplesIterator&                       feature_query_range_first,
    const SamplesIterator&                       feature_query_range_last,
    const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    NearestNeighborsBufferBase<typename IndicesIterator::value_type, typename SamplesIterator::value_type>&
        nearest_neighbors_buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (bbox::is_sample_in_kd_bounding_box(
                samples_range_first + candidate_nearest_neighbor_index * n_features,
                samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(feature_query_range_first,
                                                feature_query_range_last,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

}  // namespace ffcl::knn