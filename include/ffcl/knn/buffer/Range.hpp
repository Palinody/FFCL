#pragma once

#include "ffcl/knn/buffer/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include "ffcl/datastruct/BoundingBox.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::buffer {

template <typename IndexType, typename DistanceType>
class Range : public Base<IndexType, DistanceType> {
  private:
    using IndicesType   = typename Base<IndexType, DistanceType>::IndicesType;
    using DistancesType = typename Base<IndexType, DistanceType>::DistancesType;

    using HyperRangeType = bbox::HyperRangeType<typename DistancesType::iterator>;

  public:
    explicit Range(const HyperRangeType& kd_bounding_box)
      : Range(kd_bounding_box, {}, {}) {}

    explicit Range(const HyperRangeType& kd_bounding_box,
                   const IndicesType&    init_neighbors_indices,
                   const DistancesType&  init_neighbors_distances)
      : kd_bounding_box_{kd_bounding_box}
      , indices_{init_neighbors_indices}
      , distances_{init_neighbors_distances} {}

    std::size_t size() const {
        return indices_.size();
    }

    std::size_t n_free_slots() const {
        return common::utils::infinity<IndexType>();
    }

    bool empty() const {
        return indices_.empty();
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        throw std::runtime_error("No furthest index to return for this type of buffer.");
        return IndexType{};
    }

    DistanceType upper_bound() const {
        throw std::runtime_error("No upper bound to return if no dimension is specified for this buffer.");
        return DistanceType{};
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        return (kd_bounding_box_[feature_index].second - kd_bounding_box_[feature_index].first) / 2;
    }

    IndicesType indices() const {
        return indices_;
    }

    DistancesType distances() const {
        return distances_;
    }

    IndicesType move_indices() {
        return std::move(indices_);
    }

    DistancesType move_distances() {
        return std::move(distances_);
    }

    std::tuple<IndicesType, DistancesType> move_data_to_indices_distances_pair() {
        return std::make_tuple(std::move(indices_), std::move(distances_));
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        common::utils::ignore_parameters(index_candidate, distance_candidate);
        throw std::runtime_error("Cannot update this buffer without specifying a dimension to operate on.");
    }

    void update(const IndexType&    index_candidate,
                const DistanceType& distance_candidate,
                const IndexType&    feature_index) {
        if (distance_candidate < upper_bound(feature_index)) {
            indices_.emplace_back(index_candidate);
            distances_.emplace_back(distance_candidate);
        }
    }

    void print() const {
        for (std::size_t index = 0; index < std::min(indices_.size(), distances_.size()); ++index) {
            std::cout << "(" << indices_[index] << ", " << distances_[index] << ")\n";
        }
    }

  private:
    HyperRangeType kd_bounding_box_;
    IndicesType    indices_;
    DistancesType  distances_;
};

}  // namespace ffcl::knn::buffer