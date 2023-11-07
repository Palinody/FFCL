#pragma once

#include "ffcl/knn/count/Base.hpp"

#include "ffcl/common/Utils.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::count {

template <typename IndexType, typename DistanceType>
class Radius : public Base<IndexType, DistanceType> {
  public:
    explicit Radius(const DistanceType& radius)
      : radius_{radius} {}

    std::size_t n_free_slots() const {
        return common::utils::infinity<IndexType>();
    }

    DistanceType upper_bound() const {
        return radius_;
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        common::utils::ignore_parameters(feature_index);
        return this->upper_bound();
    }

    IndexType count() const {
        return count_;
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // should be replaced by radius check
        common::utils::ignore_parameters(index_candidate);
        if (distance_candidate < this->upper_bound()) {
            ++count_;
        }
    }

    void update(const IndexType&    index_candidate,
                const DistanceType& distance_candidate,
                const IndexType&    feature_index) {
        common::utils::ignore_parameters(feature_index);
        this->update(index_candidate, distance_candidate);
    }

    void print() const {
        std::cout << "count: " << count_ << ", radius: " << radius_ << "\n";
    }

  private:
    DistanceType radius_;
    IndexType    count_;
};

}  // namespace ffcl::knn::count