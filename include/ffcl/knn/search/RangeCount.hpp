#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/math/heuristics/Distances.hpp"

#include "ffcl/knn/count/Range.hpp"

#include "ffcl/datastruct/BoundingBox.hpp"

namespace ffcl::knn::count {

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

template <typename Indexer>
class RangeCount {
  private:
    using IndexType = typename Indexer::IndexType;
    using DataType  = typename Indexer::DataType;

    using HyperRangeType = bbox::HyperRangeType<typename std::vector<DataType>::iterator>;

  public:
    RangeCount(Indexer&& indexer, std::size_t query_index, const knn::count::Range<IndexType, DataType>& counter)
      : indexer_{std::move(indexer)}
      , query_index_{query_index}
      , counter_{counter} {}

  private:
    Indexer                                indexer_;
    std::size_t                            query_index_;
    knn::count::Range<IndexType, DataType> counter_;
};

}  // namespace ffcl::knn::count