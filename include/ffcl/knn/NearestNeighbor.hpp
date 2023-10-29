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
void nearest_neighbor(const IndicesIterator&                subrange_index_first,
                      const IndicesIterator&                subrange_index_last,
                      const SamplesIterator&                dataset_samples_first,
                      const SamplesIterator&                dataset_samples_last,
                      std::size_t                           n_features,
                      std::size_t                           sample_index_query,
                      ssize_t&                              current_nearest_neighbor_index,
                      typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
                current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
                current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor(const IndicesIterator&                subrange_index_first,
                      const IndicesIterator&                subrange_index_last,
                      const SamplesIterator&                dataset_samples_first,
                      const SamplesIterator&                dataset_samples_last,
                      std::size_t                           n_features,
                      const SamplesIterator&                sample_feature_query_first,
                      const SamplesIterator&                sample_feature_query_last,
                      ssize_t&                              current_nearest_neighbor_index,
                      typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
            current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
            current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors(const IndicesIterator&                       subrange_index_first,
                         const IndicesIterator&                       subrange_index_last,
                         const SamplesIterator&                       dataset_samples_first,
                         const SamplesIterator&                       dataset_samples_last,
                         std::size_t                                  n_features,
                         std::size_t                                  sample_index_query,
                         NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors(const IndicesIterator&                       subrange_index_first,
                         const IndicesIterator&                       subrange_index_last,
                         const SamplesIterator&                       dataset_samples_first,
                         const SamplesIterator&                       dataset_samples_last,
                         std::size_t                                  n_features,
                         const SamplesIterator&                       sample_feature_query_first,
                         const SamplesIterator&                       sample_feature_query_last,
                         NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius(const IndicesIterator&                      subrange_index_first,
                                         const IndicesIterator&                      subrange_index_last,
                                         const SamplesIterator&                      dataset_samples_first,
                                         const SamplesIterator&                      dataset_samples_last,
                                         std::size_t                                 n_features,
                                         std::size_t                                 sample_index_query,
                                         const typename SamplesIterator::value_type& radius,
                                         std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                ++neighbors_count;
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius(const IndicesIterator&                      subrange_index_first,
                                         const IndicesIterator&                      subrange_index_last,
                                         const SamplesIterator&                      dataset_samples_first,
                                         const SamplesIterator&                      dataset_samples_last,
                                         std::size_t                                 n_features,
                                         const SamplesIterator&                      sample_feature_query_first,
                                         const SamplesIterator&                      sample_feature_query_last,
                                         const typename SamplesIterator::value_type& radius,
                                         std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_hyper_range(const IndicesIterator&                       subrange_index_first,
                                              const IndicesIterator&                       subrange_index_last,
                                              const SamplesIterator&                       dataset_samples_first,
                                              const SamplesIterator&                       dataset_samples_last,
                                              std::size_t                                  n_features,
                                              std::size_t                                  sample_index_query,
                                              const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                              std::size_t&                                 neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_hyper_range(const IndicesIterator&                       subrange_index_first,
                                              const IndicesIterator&                       subrange_index_last,
                                              const SamplesIterator&                       dataset_samples_first,
                                              const SamplesIterator&                       dataset_samples_last,
                                              std::size_t                                  n_features,
                                              const SamplesIterator&                       sample_feature_query_first,
                                              const SamplesIterator&                       sample_feature_query_last,
                                              const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                              std::size_t&                                 neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last, sample_feature_query_first, sample_feature_query_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius(const IndicesIterator&                       subrange_index_first,
                                   const IndicesIterator&                       subrange_index_last,
                                   const SamplesIterator&                       dataset_samples_first,
                                   const SamplesIterator&                       dataset_samples_last,
                                   std::size_t                                  n_features,
                                   std::size_t                                  sample_index_query,
                                   const typename SamplesIterator::value_type&  radius,
                                   NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius(const IndicesIterator&                       subrange_index_first,
                                   const IndicesIterator&                       subrange_index_last,
                                   const SamplesIterator&                       dataset_samples_first,
                                   const SamplesIterator&                       dataset_samples_last,
                                   std::size_t                                  n_features,
                                   const SamplesIterator&                       sample_feature_query_first,
                                   const SamplesIterator&                       sample_feature_query_last,
                                   const typename SamplesIterator::value_type&  radius,
                                   NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_hyper_range(const IndicesIterator&                       subrange_index_first,
                                        const IndicesIterator&                       subrange_index_last,
                                        const SamplesIterator&                       dataset_samples_first,
                                        const SamplesIterator&                       dataset_samples_last,
                                        std::size_t                                  n_features,
                                        std::size_t                                  sample_index_query,
                                        const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                        NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_hyper_range(const IndicesIterator&                       subrange_index_first,
                                        const IndicesIterator&                       subrange_index_last,
                                        const SamplesIterator&                       dataset_samples_first,
                                        const SamplesIterator&                       dataset_samples_last,
                                        std::size_t                                  n_features,
                                        const SamplesIterator&                       sample_feature_query_first,
                                        const SamplesIterator&                       sample_feature_query_last,
                                        const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                        NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(sample_feature_query_first,
                                                sample_feature_query_last,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

}  // namespace ffcl::knn