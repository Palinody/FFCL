#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/knn/count/Range.hpp"

#include "ffcl/datastruct/BoundingBox.hpp"

namespace ffcl::knn::count {

/*
template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_hyper_range(const IndicesIterator&                       indices_range_first,
                                              const IndicesIterator&                       indices_range_last,
                                              const SamplesIterator&                       samples_range_first,
                                              const SamplesIterator&                       samples_range_last,
                                              std::size_t                                  n_features,
                                              std::size_t                                  sample_index_query,
                                              const datastruct::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                              std::size_t&                                 neighbors_count) {
    common::ignore_parameters(samples_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            datastruct::bbox::is_sample_in_kd_bounding_box(
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
                                              const datastruct::bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
                                              std::size_t&                                 neighbors_count) {
    common::ignore_parameters(samples_range_last, feature_query_range_first, feature_query_range_last);

    const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

        if (datastruct::bbox::is_sample_in_kd_bounding_box(
                samples_range_first + candidate_nearest_neighbor_index * n_features,
                samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}
*/

}  // namespace ffcl::knn::count