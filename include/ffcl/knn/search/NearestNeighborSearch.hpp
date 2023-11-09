#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/knn/buffer/Base.hpp"

namespace ffcl::knn::search {

/*
template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor(const IndicesIterator&                          indices_range_first,
                      const IndicesIterator&                          indices_range_last,
                      const SamplesIterator&                          samples_range_first,
                      const SamplesIterator&                          samples_range_last,
                      std::size_t                                     n_features,
                      std::size_t                                     sample_index_query,
                      buffer::Base<IndicesIterator, SamplesIterator>& buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                common::math::heuristics::auto_distance(samples_range_first + sample_index_query * n_features,
                                                samples_range_first + sample_index_query * n_features + n_features,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor(const IndicesIterator&                          indices_range_first,
                      const IndicesIterator&                          indices_range_last,
                      const SamplesIterator&                          samples_range_first,
                      const SamplesIterator&                          samples_range_last,
                      std::size_t                                     n_features,
                      const SamplesIterator&                          feature_query_range_first,
                      const SamplesIterator&                          feature_query_range_last,
                      buffer::Base<IndicesIterator, SamplesIterator>& buffer) {
    common::utils::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        const auto candidate_nearest_neighbor_distance =
            common::math::heuristics::auto_distance(feature_query_range_first,
                                            feature_query_range_last,
                                            samples_range_first + candidate_nearest_neighbor_index * n_features);

        buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
    }
}
*/

}  // namespace ffcl::knn::search