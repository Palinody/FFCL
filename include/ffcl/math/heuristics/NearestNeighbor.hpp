#pragma once

#include "ffcl/common/Utils.hpp"
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

    static constexpr auto comparison_lambda = [](const ElementDataType& left_tuple,
                                                 const ElementDataType& right_tuple) {
        return std::get<1>(left_tuple) < std::get<1>(right_tuple);
    };

    using PriorityQueueType =
        typename std::priority_queue<ElementDataType, std::vector<ElementDataType>, decltype(comparison_lambda)>;

  public:
    explicit NearestNeighborsBuffer()
      : max_elements_{1}
      , priority_queue_{comparison_lambda} {}

    explicit NearestNeighborsBuffer(std::size_t max_elements)
      : max_elements_{max_elements}
      , priority_queue_{comparison_lambda} {}

    NearestNeighborsBuffer& operator=(const NearestNeighborsBuffer&) = default;

    NearestNeighborsBuffer(const NearestNeighborsBuffer&) = default;

    std::size_t size() const {
        return priority_queue_.size();
    }

    bool empty() const {
        return priority_queue_.empty();
    }

    bool update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // populate the priority queue if theres still empty space
        if (priority_queue_.size() < max_elements_) {
            priority_queue_.emplace(std::make_tuple(index_candidate, distance_candidate));
            return true;

        } else if (distance_candidate < std::get<1>(priority_queue_.top())) {
            // remove the element with the distance thats greater than the distance candidate
            priority_queue_.pop();
            priority_queue_.emplace(std::make_tuple(index_candidate, distance_candidate));
            return true;
        }
        return false;
    }

    void print() {
        PriorityQueueType priority_queue_cpy = priority_queue_;  // Create a copy of the priority queue

        while (!priority_queue_cpy.empty()) {
            auto element = priority_queue_cpy.top();
            priority_queue_cpy.pop();
            // printf("(%ld, %.5f)\n", std::get<0>(element), std::get<1>(element));
            std::cout << "(" << std::get<0>(element) << ", " << std::get<1>(element) << ")\n";
        }
    }

  private:
    std::size_t       max_elements_;
    PriorityQueueType priority_queue_;
};

namespace math::heuristics {

template <typename SamplesIterator>
std::pair<ssize_t, typename SamplesIterator::value_type> nearest_neighbor_range(
    const SamplesIterator&               subrange_samples_first,
    const SamplesIterator&               subrange_samples_last,
    const SamplesIterator&               dataset_samples_first,
    const SamplesIterator&               dataset_samples_last,
    std::size_t                          n_features,
    std::size_t                          sample_index_query,
    ssize_t                              current_nearest_neighbor_index = -1,
    typename SamplesIterator::value_type current_nearest_neighbor_distance =
        common::utils::infinity<typename SamplesIterator::value_type>()) {
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
    return {current_nearest_neighbor_index, current_nearest_neighbor_distance};
}

template <typename IndicesIterator, typename SamplesIterator>
std::pair<ssize_t, typename SamplesIterator::value_type> nearest_neighbor_indexed_range(
    const IndicesIterator&               index_first,
    const IndicesIterator&               index_last,
    const SamplesIterator&               samples_first,
    const SamplesIterator&               samples_last,
    std::size_t                          n_features,
    std::size_t                          sample_index_query,
    ssize_t                              current_nearest_neighbor_index = -1,
    typename SamplesIterator::value_type current_nearest_neighbor_distance =
        common::utils::infinity<typename SamplesIterator::value_type>()) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples = std::distance(index_first, index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const DataType candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(samples_first + sample_index_query * n_features,
                                                samples_first + sample_index_query * n_features + n_features,
                                                samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
                current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
                current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
            }
        }
    }
    return {current_nearest_neighbor_index, current_nearest_neighbor_distance};
}

}  // namespace math::heuristics