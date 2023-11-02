#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::knn::counter {

template <typename IndexType, typename DistanceType>
class Base {
  public:
    virtual ~Base() {}

    virtual DistanceType upper_bound() const = 0;

    virtual IndexType counter() = 0;

    virtual void update(const IndexType& index_candidate, const DistanceType& distance_candidate) = 0;

    virtual void print() const = 0;
};

}  // namespace ffcl::knn::counter