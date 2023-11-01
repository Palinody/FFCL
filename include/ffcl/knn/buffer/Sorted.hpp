#pragma once

#include "ffcl/knn/buffer/Base.hpp"

#include "ffcl/common/Utils.hpp"

#include <algorithm>  // std::lower_bound
#include <cstddef>
#include <iostream>
#include <vector>

namespace ffcl::knn::buffer {

/**
 * @brief A sorted version of NearestNeighborsBuffer using std::lower_bound
 * A few approaches are possible (not exhaustive):
 *      - Using std::lower_bound
 *      - using an unsorted method that sorts the neighbors only once queried.
 *      An additional binary tag could be useful to avoid sorting unnecessarily.
 * The second approach would seem less useful than the first one because the client could easily sort the queried
 * nearest neighbors array himself. Which version would be better in which circumstances is unknown yet. To be tested.
 *
 * @tparam SamplesIterator
 */
template <typename IndexType, typename DistanceType>
class Sorted : public Base<IndexType, DistanceType> {
  private:
    using IndicesType   = typename Base<IndexType, DistanceType>::IndicesType;
    using DistancesType = typename Base<IndexType, DistanceType>::DistancesType;

  public:
    explicit Sorted(const IndicesType& max_capacity = common::utils::infinity<IndexType>())
      : max_capacity_{max_capacity} {}

    std::size_t size() const {
        return indices_.size();
    }

    std::size_t n_free_slots() const {
        return max_capacity_ - this->size();
    }

    bool empty() const {
        return indices_.empty();
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        return indices_.back();
    }

    DistanceType furthest_k_nearest_neighbor_distance() const {
        return distances_.back();
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

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        auto distances_it = std::lower_bound(distances_.begin(), distances_.end(), distance_candidate);

        // populate at the right index if the max capacity isnt reached
        if (this->n_free_slots()) {
            const std::size_t insertion_index = std::distance(distances_.begin(), distances_it);
            indices_.insert(indices_.begin() + insertion_index, index_candidate);
            distances_.insert(distances_it, distance_candidate);

        }
        // populate at the right index if the max capacity reached but the candidate has a closer distance
        else if (distances_it != distances_.end()) {
            const std::size_t insertion_index = std::distance(distances_.begin(), distances_it);
            indices_.insert(indices_.begin() + insertion_index, index_candidate);
            distances_.insert(distances_it, distance_candidate);
            // remove the last element now that the vectors overflow the max capacity
            indices_.pop_back();
            distances_.pop_back();
        }
    }

    void print() const {
        for (std::size_t index = 0; index < std::min(indices_.size(), distances_.size()); ++index) {
            std::cout << "(" << indices_[index] << ", " << distances_[index] << ")\n";
        }
    }

  private:
    IndicesType   indices_;
    DistancesType distances_;
    IndexType     max_capacity_;
};

}  // namespace ffcl::knn::buffer