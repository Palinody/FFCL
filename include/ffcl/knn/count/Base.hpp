#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::knn::count {

template <typename IndicesIterator, typename DistancesIterator>
class Base {
  public:
    using IndexType    = typename IndicesIterator::value_type;
    using DistanceType = typename DistancesIterator::value_type;

    using SamplesIterator = typename std::vector<DistanceType>::iterator;

    virtual ~Base() {}

    virtual DistanceType upper_bound() const = 0;

    virtual DistanceType upper_bound(const IndexType& feature_index) const = 0;

    virtual std::size_t n_free_slots() const = 0;

    virtual IndexType count() const = 0;

    virtual void update(const IndexType& index_candidate, const DistanceType& distance_candidate) = 0;

    virtual void search(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        std::size_t            sample_index_query) = 0;

    virtual void search(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        const SamplesIterator& feature_query_range_first,
                        const SamplesIterator& feature_query_range_last) = 0;

    virtual void print() const = 0;
};

}  // namespace ffcl::knn::count