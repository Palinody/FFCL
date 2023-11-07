#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::knn::count {

template <typename IndexType, typename DistanceType>
class Base {
  public:
    virtual ~Base() {}

    virtual DistanceType upper_bound() const = 0;

    virtual DistanceType upper_bound(const IndexType& feature_index) const = 0;

    virtual std::size_t n_free_slots() const = 0;

    virtual IndexType count() const = 0;

    virtual void update(const IndexType& index_candidate, const DistanceType& distance_candidate) = 0;

    virtual void update(const IndexType&    index_candidate,
                        const DistanceType& distance_candidate,
                        const IndexType&    feature_index) = 0;

    virtual void print() const = 0;
};

}  // namespace ffcl::knn::count