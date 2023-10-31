#pragma once

#include "ffcl/knn/buffer/NearestNeighborsBufferBase.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::buffer {

template <typename IndexType, typename DistanceType, typename VisitedIndicesType = std::unordered_set<std::size_t>>
class NearestNeighborsBufferWithMemory : public NearestNeighborsBufferBase<IndexType, DistanceType> {
  private:
    using IndicesType   = typename NearestNeighborsBufferBase<IndexType, DistanceType>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<IndexType, DistanceType>::DistancesType;

    using IndicesIterator = typename std::vector<IndexType>::iterator;

  public:
    explicit NearestNeighborsBufferWithMemory(const IndexType& max_capacity = common::utils::infinity<IndexType>())
      : NearestNeighborsBufferWithMemory({}, {}, max_capacity) {}

    NearestNeighborsBufferWithMemory(const VisitedIndicesType& visited_indices_reference,
                                     const IndexType&          max_capacity = common::utils::infinity<IndexType>())
      : furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_reference_{visited_indices_reference} {}

    NearestNeighborsBufferWithMemory(VisitedIndicesType&& visited_indices,
                                     const IndexType&     max_capacity = common::utils::infinity<IndexType>())
      : furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{std::move(visited_indices)}
      , visited_indices_reference_{visited_indices_} {}

    NearestNeighborsBufferWithMemory(const IndicesIterator& visited_indices_first,
                                     const IndicesIterator& visited_indices_last,
                                     const IndexType&       max_capacity = common::utils::infinity<IndexType>())
      : furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{VisitedIndicesType(visited_indices_first, visited_indices_last)}
      , visited_indices_reference_{visited_indices_} {}

    NearestNeighborsBufferWithMemory(const IndicesType&   init_neighbors_indices,
                                     const DistancesType& init_neighbors_distances,
                                     const IndexType&     max_capacity = common::utils::infinity<IndexType>())
      : indices_{init_neighbors_indices}
      , distances_{init_neighbors_distances}
      , furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity > indices_.size() ? max_capacity : indices_.size()}
      , visited_indices_{VisitedIndicesType(indices_.begin(), indices_.end())}
      , visited_indices_reference_{visited_indices_} {
        if (indices_.size()) {
            if (indices_.size() == distances_.size()) {
                std::tie(furthest_buffer_index_, furthest_k_nearest_neighbor_distance_) =
                    math::statistics::get_max_index_value_pair(distances_.begin(), distances_.end());

            } else {
                throw std::runtime_error("Indices and distances buffers sizes do not match.");
            }
        }
    }

    std::size_t size() const {
        return indices_.size();
    }

    std::size_t n_free_slots() const {
        return max_capacity_ - this->size();
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        return indices_[furthest_buffer_index_];
    }

    DistanceType furthest_k_nearest_neighbor_distance() const {
        return furthest_k_nearest_neighbor_distance_;
    }

    bool empty() const {
        return indices_.empty();
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

    auto closest_neighbor_index_distance_pair() {
        const auto [closest_buffer_index, closest_nearest_neighbor_distance] =
            math::statistics::get_min_index_value_pair(distances_.begin(), distances_.end());

        return std::make_pair(indices_[closest_buffer_index], closest_nearest_neighbor_distance);
    }

    void reset_buffers_except_memory() {
        // reset all the buffers to default values
        // max_capacity_ and visited_indices_ remain unchanged
        indices_.clear();
        distances_.clear();
        furthest_buffer_index_                = 0;
        furthest_k_nearest_neighbor_distance_ = 0;
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // consider an update only if the index hasnt been visited
        if (visited_indices_reference_.find(index_candidate) == visited_indices_reference_.end()) {
            // always populate if the max capacity isnt reached
            if (this->n_free_slots()) {
                indices_.emplace_back(index_candidate);
                distances_.emplace_back(distance_candidate);
                if (distance_candidate > furthest_k_nearest_neighbor_distance_) {
                    // update the new index position of the furthest in the buffer
                    furthest_buffer_index_                = indices_.size() - 1;
                    furthest_k_nearest_neighbor_distance_ = distance_candidate;
                }
            }
            // populate if the max capacity is reached and the candidate has a closer distance
            else if (distance_candidate < furthest_k_nearest_neighbor_distance_) {
                // replace the previous greatest distance now that the vectors overflow the max capacity
                indices_[furthest_buffer_index_]   = index_candidate;
                distances_[furthest_buffer_index_] = distance_candidate;
                // find the new furthest neighbor and update the cache accordingly
                std::tie(furthest_buffer_index_, furthest_k_nearest_neighbor_distance_) =
                    math::statistics::get_max_index_value_pair(distances_.begin(), distances_.end());
            }
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
    IndexType     furthest_buffer_index_;
    DistanceType  furthest_k_nearest_neighbor_distance_;
    IndexType     max_capacity_;

    VisitedIndicesType        visited_indices_;
    const VisitedIndicesType& visited_indices_reference_;
};

}  // namespace ffcl::knn::buffer