#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::knn::buffer {

template <typename IndexType, typename DistanceType>
class Base {
  public:
    virtual ~Base() {}

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

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

    virtual void update(const IndexType&    index_candidate,
                        const DistanceType& distance_candidate,
                        const IndexType&    feature_index) = 0;

    virtual void print() const = 0;
};

}  // namespace ffcl::knn::buffer