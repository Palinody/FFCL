#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/heuristics/Distances.hpp"

#include <vector>

namespace math::heuristics {

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

template <typename IndicesIterator, typename SamplesIterator>
std::pair<ssize_t, typename SamplesIterator::value_type> nearest_neighbor_indexed_range(
    const IndicesIterator& index_first,
    const IndicesIterator& index_last,
    const SamplesIterator& samples_first,
    const SamplesIterator& samples_last,
    std::size_t            n_features,
    std::size_t            sample_index_query,
    IndicesIterator        exclusion_index_first,
    IndicesIterator        exclusion_index_last) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples = std::distance(index_first, index_last);

    if (n_samples > 1) {
        ssize_t current_nearest_neighbor_index    = -1;
        auto    current_nearest_neighbor_distance = common::utils::infinity<DataType>();

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_nearest_neighbor_index = index_first[index];

            if (candidate_nearest_neighbor_index != sample_index_query &&
                common::utils::is_element_not_in(
                    exclusion_index_first, exclusion_index_last, candidate_nearest_neighbor_index)) {
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

    } else if (n_samples == 1 && index_first[0] != sample_index_query &&
               common::utils::is_element_not_in(exclusion_index_first, exclusion_index_last, index_first[0])) {
        return {index_first[0],
                math::heuristics::auto_distance(samples_first + sample_index_query * n_features,
                                                samples_first + sample_index_query * n_features + n_features,
                                                samples_first + index_first[0] * n_features)};
    }
    return {-1, common::utils::infinity<DataType>()};
}

template <typename SamplesIterator>
void update_nearest_neighbors_indices_buffer(const SamplesIterator&    samples_first,
                                             const SamplesIterator&    samples_last,
                                             std::size_t               n_features,
                                             std::size_t               query_index,
                                             std::size_t               candidate_nearest_neighbor_index,
                                             std::size_t               n_neighbors,
                                             std::vector<std::size_t>& indices_buffer,
                                             std::vector<typename SamplesIterator::value_type>& distances_buffer) {
    // N.B.: keep the elments sorted and check if a candidate is closer than the furthest current nearest neighbors. If
    // its the case, place it at the correct position.
    // Checks whether one of the elements of the buffer is larger than the sample candidate and replace the furthest
    // nearest neighbor if the buffer is already full

    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(samples_last);

    assert(distances_buffer.size() <= n_neighbors);

    bool is_query_valid = (query_index != candidate_nearest_neighbor_index &&
                           common::utils::is_element_not_in(
                               indices_buffer.begin(), indices_buffer.end(), candidate_nearest_neighbor_index))
                              ? true
                              : false;

    if (is_query_valid) {
        const DataType candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(samples_first + query_index * n_features,
                                            samples_first + query_index * n_features + n_features,
                                            samples_first + candidate_nearest_neighbor_index * n_features);

        // if the buffer is not full yet
        if (indices_buffer.size() < n_neighbors) {
            indices_buffer.emplace_back(candidate_nearest_neighbor_index);
            distances_buffer.emplace_back(candidate_nearest_neighbor_distance);

            // sort the indices according to the distances buffer sorting order
            std::sort(indices_buffer.begin(),
                      indices_buffer.end(),
                      [&distances_buffer](const auto& first_index, const auto& second_index) {
                          return distances_buffer[first_index] < distances_buffer[second_index];
                      });
            // then sort the distances buffer itself
            std::sort(distances_buffer.begin(), distances_buffer.end());

        } else {
            // get the position where the candidate distance should be inserted, assuming that distances_buffer is
            // sorted
            auto index_iterator =
                std::lower_bound(distances_buffer.begin(), distances_buffer.end(), candidate_nearest_neighbor_distance);
            // dont do anything if the candidate is larger than all the elements in the buffer
            if (index_iterator != distances_buffer.end()) {
                // which position the candidate should be inserted at
                const std::size_t index = std::distance(distances_buffer.begin(), index_iterator);

                distances_buffer.insert(index_iterator, candidate_nearest_neighbor_distance);
                indices_buffer.insert(indices_buffer.begin() + index, candidate_nearest_neighbor_index);

                // remove the last element that now overflows the buffers
                distances_buffer.pop_back();
                indices_buffer.pop_back();
            }
        }
    }
}

}  // namespace math::heuristics