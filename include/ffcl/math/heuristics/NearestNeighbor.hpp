#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/BoundingBox.hpp"
#include "ffcl/math/heuristics/Distances.hpp"

#include <functional>
#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

template <typename SamplesIterator>
class NearestNeighborsBuffer {
  private:
    using IndexType       = std::size_t;
    using DistanceType    = typename SamplesIterator::value_type;
    using ElementDataType = typename std::tuple<IndexType, DistanceType>;

    static constexpr auto comparison_lambda = [](const ElementDataType& left_element,
                                                 const ElementDataType& right_element) {
        // the greatest element will appear at the top of the priority queue
        return std::get<1>(left_element) < std::get<1>(right_element);
    };

    using PriorityQueueType =
        typename std::priority_queue<ElementDataType, std::vector<ElementDataType>, decltype(comparison_lambda)>;

  public:
    NearestNeighborsBuffer()
      : max_elements_{1}
      , priority_queue_{comparison_lambda} {}

    NearestNeighborsBuffer(std::size_t max_elements)
      : max_elements_{max_elements}
      , priority_queue_{comparison_lambda} {}

    std::size_t size() const {
        return priority_queue_.size();
    }

    bool empty() const {
        return priority_queue_.empty();
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        return std::get<0>(priority_queue_.top());
    }

    DistanceType furthest_k_nearest_neighbor_distance() const {
        return std::get<1>(priority_queue_.top());
    }

    auto move_data_to_indices_distances_pair() {
        std::vector<IndexType> indices;
        indices.reserve(this->size());
        std::vector<DistanceType> distances;
        distances.reserve(this->size());

        while (!priority_queue_.empty()) {
            const auto [index, distance] = priority_queue_.top();
            indices.emplace_back(index);
            distances.emplace_back(distance);
            priority_queue_.pop();
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    auto copy_data_to_indices_distances_pair() const {
        PriorityQueueType priority_queue_cpy = priority_queue_;

        std::vector<IndexType> indices;
        indices.reserve(this->size());
        std::vector<DistanceType> distances;
        distances.reserve(this->size());

        while (!priority_queue_cpy.empty()) {
            const auto [index, distance] = priority_queue_cpy.top();
            indices.emplace_back(index);
            distances.emplace_back(distance);
            priority_queue_cpy.pop();
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    bool update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // populate the priority queue if theres still empty space
        if (priority_queue_.size() < max_elements_) {
            priority_queue_.emplace(std::make_tuple(index_candidate, distance_candidate));
            return true;

        } else if (distance_candidate < std::get<1>(priority_queue_.top())) {
            // pop the element at the top of the priority queue if the candidate is smaller
            priority_queue_.pop();
            // then place the candidate index-distance pair accordingly
            priority_queue_.emplace(std::make_tuple(index_candidate, distance_candidate));
            return true;
        }
        return false;
    }

    void print() {
        PriorityQueueType priority_queue_cpy = priority_queue_;

        while (!priority_queue_cpy.empty()) {
            auto element = priority_queue_cpy.top();
            priority_queue_cpy.pop();
            // printf("(%ld, %.5f)\n", std::get<0>(element), std::get<1>(element));
            std::cout << "(" << std::get<0>(element) << ", " << std::get<1>(element) << ")\n";
        }
    }

  public:
    std::size_t       max_elements_;
    PriorityQueueType priority_queue_;
};

namespace math::heuristics {

template <typename SamplesIterator>
void nearest_neighbor_range(const SamplesIterator&                subrange_samples_first,
                            const SamplesIterator&                subrange_samples_last,
                            const SamplesIterator&                dataset_samples_first,
                            const SamplesIterator&                dataset_samples_last,
                            std::size_t                           n_features,
                            std::size_t                           sample_index_query,
                            ssize_t&                              current_nearest_neighbor_index,
                            typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
                current_nearest_neighbor_index    = subrange_offset + subrange_candidate_index;
                current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
            }
        }
    }
}

template <typename SamplesIterator>
void k_nearest_neighbors_range(const SamplesIterator&                   subrange_samples_first,
                               const SamplesIterator&                   subrange_samples_last,
                               const SamplesIterator&                   dataset_samples_first,
                               const SamplesIterator&                   dataset_samples_last,
                               std::size_t                              n_features,
                               std::size_t                              sample_index_query,
                               NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            nearest_neighbors_buffer.update(subrange_offset + subrange_candidate_index,
                                            candidate_nearest_neighbor_distance);
        }
    }
}

template <typename SamplesIterator>
void increment_neighbors_count_in_radius_range(const SamplesIterator&                      subrange_samples_first,
                                               const SamplesIterator&                      subrange_samples_last,
                                               const SamplesIterator&                      dataset_samples_first,
                                               const SamplesIterator&                      dataset_samples_last,
                                               std::size_t                                 n_features,
                                               std::size_t                                 sample_index_query,
                                               const typename SamplesIterator::value_type& radius,
                                               std::size_t&                                neighbors_count) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                ++neighbors_count;
            }
        }
    }
}

template <typename SamplesIterator>
void k_nearest_neighbors_in_radius_range(const SamplesIterator&                      subrange_samples_first,
                                         const SamplesIterator&                      subrange_samples_last,
                                         const SamplesIterator&                      dataset_samples_first,
                                         const SamplesIterator&                      dataset_samples_last,
                                         std::size_t                                 n_features,
                                         std::size_t                                 sample_index_query,
                                         const typename SamplesIterator::value_type& radius,
                                         NearestNeighborsBuffer<SamplesIterator>&    nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                nearest_neighbors_buffer.update(subrange_offset + subrange_candidate_index,
                                                candidate_nearest_neighbor_distance);
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor_indexed_range(const IndicesIterator&                subrange_index_first,
                                    const IndicesIterator&                subrange_index_last,
                                    const SamplesIterator&                dataset_samples_first,
                                    const SamplesIterator&                dataset_samples_last,
                                    std::size_t                           n_features,
                                    std::size_t                           sample_index_query,
                                    ssize_t&                              current_nearest_neighbor_index,
                                    typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
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
void nearest_neighbor_indexed_range(const IndicesIterator&                subrange_index_first,
                                    const IndicesIterator&                subrange_index_last,
                                    const SamplesIterator&                dataset_samples_first,
                                    const SamplesIterator&                dataset_samples_last,
                                    std::size_t                           n_features,
                                    const SamplesIterator&                sample_feature_query_first,
                                    const SamplesIterator&                sample_feature_query_last,
                                    ssize_t&                              current_nearest_neighbor_index,
                                    typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const DataType candidate_nearest_neighbor_distance =
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
void k_nearest_neighbors_indexed_range(const IndicesIterator&                   subrange_index_first,
                                       const IndicesIterator&                   subrange_index_last,
                                       const SamplesIterator&                   dataset_samples_first,
                                       const SamplesIterator&                   dataset_samples_last,
                                       std::size_t                              n_features,
                                       std::size_t                              sample_index_query,
                                       NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_indexed_range(const IndicesIterator&                   subrange_index_first,
                                       const IndicesIterator&                   subrange_index_last,
                                       const SamplesIterator&                   dataset_samples_first,
                                       const SamplesIterator&                   dataset_samples_last,
                                       std::size_t                              n_features,
                                       const SamplesIterator&                   sample_feature_query_first,
                                       const SamplesIterator&                   sample_feature_query_last,
                                       NearestNeighborsBuffer<SamplesIterator>& nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const DataType candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius_indexed_range(const IndicesIterator& subrange_index_first,
                                                       const IndicesIterator& subrange_index_last,
                                                       const SamplesIterator& dataset_samples_first,
                                                       const SamplesIterator& dataset_samples_last,
                                                       std::size_t            n_features,
                                                       std::size_t            sample_index_query,
                                                       const typename SamplesIterator::value_type& radius,
                                                       std::size_t&                                neighbors_count) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
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
void increment_neighbors_count_in_radius_indexed_range(const IndicesIterator& subrange_index_first,
                                                       const IndicesIterator& subrange_index_last,
                                                       const SamplesIterator& dataset_samples_first,
                                                       const SamplesIterator& dataset_samples_last,
                                                       std::size_t            n_features,
                                                       const SamplesIterator& sample_feature_query_first,
                                                       const SamplesIterator& sample_feature_query_last,
                                                       const typename SamplesIterator::value_type& radius,
                                                       std::size_t&                                neighbors_count) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const DataType candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                    subrange_index_first,
    const IndicesIterator&                    subrange_index_last,
    const SamplesIterator&                    dataset_samples_first,
    const SamplesIterator&                    dataset_samples_last,
    std::size_t                               n_features,
    std::size_t                               sample_index_query,
    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box,
    std::size_t&                              neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            kdtree::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                    subrange_index_first,
    const IndicesIterator&                    subrange_index_last,
    const SamplesIterator&                    dataset_samples_first,
    const SamplesIterator&                    dataset_samples_last,
    std::size_t                               n_features,
    const SamplesIterator&                    sample_feature_query_first,
    const SamplesIterator&                    sample_feature_query_last,
    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box,
    std::size_t&                              neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last, sample_feature_query_first, sample_feature_query_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (kdtree::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius_indexed_range(const IndicesIterator&                      subrange_index_first,
                                                 const IndicesIterator&                      subrange_index_last,
                                                 const SamplesIterator&                      dataset_samples_first,
                                                 const SamplesIterator&                      dataset_samples_last,
                                                 std::size_t                                 n_features,
                                                 std::size_t                                 sample_index_query,
                                                 const typename SamplesIterator::value_type& radius,
                                                 NearestNeighborsBuffer<SamplesIterator>&    nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
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
void k_nearest_neighbors_in_radius_indexed_range(const IndicesIterator&                      subrange_index_first,
                                                 const IndicesIterator&                      subrange_index_last,
                                                 const SamplesIterator&                      dataset_samples_first,
                                                 const SamplesIterator&                      dataset_samples_last,
                                                 std::size_t                                 n_features,
                                                 const SamplesIterator&                      sample_feature_query_first,
                                                 const SamplesIterator&                      sample_feature_query_last,
                                                 const typename SamplesIterator::value_type& radius,
                                                 NearestNeighborsBuffer<SamplesIterator>&    nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const DataType candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                    subrange_index_first,
    const IndicesIterator&                    subrange_index_last,
    const SamplesIterator&                    dataset_samples_first,
    const SamplesIterator&                    dataset_samples_last,
    std::size_t                               n_features,
    std::size_t                               sample_index_query,
    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box,
    NearestNeighborsBuffer<SamplesIterator>&  nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            kdtree::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                    subrange_index_first,
    const IndicesIterator&                    subrange_index_last,
    const SamplesIterator&                    dataset_samples_first,
    const SamplesIterator&                    dataset_samples_last,
    std::size_t                               n_features,
    const SamplesIterator&                    sample_feature_query_first,
    const SamplesIterator&                    sample_feature_query_last,
    const BoundingBoxKDType<SamplesIterator>& kd_bounding_box,
    NearestNeighborsBuffer<SamplesIterator>&  nearest_neighbors_buffer) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (kdtree::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(sample_feature_query_first,
                                                sample_feature_query_last,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

}  // namespace math::heuristics