#pragma once

#include "ffcl/knn/count/Base.hpp"

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/BoundingBox.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::count {

template <typename IndexType, typename DistanceType>
class Range : public Base<IndexType, DistanceType> {
  private:
    using HyperRangeType = bbox::HyperRangeType<typename std::vector<DistanceType>::iterator>;

  public:
    explicit Range(const HyperRangeType& kd_bounding_box)
      : kd_bounding_box_{kd_bounding_box} {}

    std::size_t n_free_slots() const {
        return common::utils::infinity<IndexType>();
    }

    DistanceType upper_bound() const {
        throw std::runtime_error("No upper bound to return if no dimension is specified for this buffer.");
        return DistanceType{};
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        return (kd_bounding_box_[feature_index].second - kd_bounding_box_[feature_index].first) / 2;
    }

    IndexType count() const {
        return count_;
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        common::utils::ignore_parameters(index_candidate, distance_candidate);
        throw std::runtime_error("Cannot update this buffer without specifying a dimension to operate on.");
    }

    void update(const IndexType&    index_candidate,
                const DistanceType& distance_candidate,
                const IndexType&    feature_index) {
        // should be replaced by bounding box check
        common::utils::ignore_parameters(index_candidate);
        if (distance_candidate < upper_bound(feature_index)) {
            ++count_;
        }
    }

    void print() const {
        std::cout << "count: " << count_ << "\n";
    }

  private:
    HyperRangeType kd_bounding_box_;
    IndexType      count_;
};

}  // namespace ffcl::knn::count