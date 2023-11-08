#pragma once

#include "ffcl/math/heuristics/Distances.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::knn::buffer {

template <typename IndicesIterator, typename DistancesIterator>
class Base {
  public:
    using IndexType     = typename IndicesIterator::value_type;
    using DistanceType  = typename DistancesIterator::value_type;
    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using SamplesIterator = DistancesIterator;

    virtual ~Base() {}

    virtual std::size_t size() const = 0;

    virtual std::size_t n_free_slots() const = 0;

    virtual bool empty() const = 0;

    virtual IndexType furthest_k_nearest_neighbor_index() const = 0;

    virtual DistanceType upper_bound() const = 0;

    virtual DistanceType upper_bound(const IndexType& feature_index) const = 0;

    virtual IndicesType indices() const = 0;

    virtual DistancesType distances() const = 0;

    virtual IndicesType move_indices() = 0;

    virtual DistancesType move_distances() = 0;

    virtual std::tuple<IndicesType, DistancesType> move_data_to_indices_distances_pair() = 0;

    virtual void update(const IndexType& index_candidate, const DistanceType& distance_candidate) = 0;

    virtual void operator()(const IndicesIterator& indices_range_first,
                            const IndicesIterator& indices_range_last,
                            const SamplesIterator& samples_range_first,
                            const SamplesIterator& samples_range_last,
                            std::size_t            n_features,
                            std::size_t            sample_index_query) = 0;

    virtual void operator()(const IndicesIterator& indices_range_first,
                            const IndicesIterator& indices_range_last,
                            const SamplesIterator& samples_range_first,
                            const SamplesIterator& samples_range_last,
                            std::size_t            n_features,
                            const SamplesIterator& feature_query_range_first,
                            const SamplesIterator& feature_query_range_last) = 0;

    virtual void print() const = 0;
};

}  // namespace ffcl::knn::buffer