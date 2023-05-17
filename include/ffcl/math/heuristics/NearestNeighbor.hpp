#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/heuristics/Distances.hpp"

namespace math::heuristics {

template <typename IndicesIterator, typename SamplesIterator>
std::pair<ssize_t, typename SamplesIterator::value_type> nearest_neighbor_indexed_range(
    const IndicesIterator& index_first,
    const IndicesIterator& index_last,
    const SamplesIterator& samples_first,
    const SamplesIterator& samples_last,
    std::size_t            n_features,
    std::size_t            sample_index_query) {
    using DataType = typename SamplesIterator::value_type;

    common::utils::ignore_parameters(samples_last);

    const std::size_t n_samples = std::distance(index_first, index_last);

    if (n_samples > 1) {
        ssize_t current_nearest_neighbor_index    = -1;
        auto    current_nearest_neighbor_distance = common::utils::infinity<DataType>();

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

    } else if (n_samples == 1 && index_first[0] != sample_index_query) {
        return {index_first[0],
                math::heuristics::auto_distance(samples_first + sample_index_query * n_features,
                                                samples_first + sample_index_query * n_features + n_features,
                                                samples_first + index_first[0] * n_features)};
    }
    return {-1, common::utils::infinity<DataType>()};
}

}  // namespace math::heuristics