#pragma once

#include "ffcl/knn/buffer/NearestNeighborsBufferBase.hpp"

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::buffer {

template <typename IndexType, typename DistanceType, typename UnionFindType = ffcl::datastruct::UnionFind<std::size_t>>
class NearestNeighborsBufferWithUnionFind : public NearestNeighborsBufferBase<IndexType, DistanceType> {
  private:
    using IndicesType   = typename NearestNeighborsBufferBase<IndexType, DistanceType>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<IndexType, DistanceType>::DistancesType;

  public:
    NearestNeighborsBufferWithUnionFind(const UnionFindType& union_find_ref,
                                        const IndexType&     query_representative,
                                        const IndexType&     max_capacity = common::utils::infinity<IndexType>())
      : NearestNeighborsBufferWithUnionFind({}, {}, union_find_ref, query_representative, max_capacity) {}

    NearestNeighborsBufferWithUnionFind(const IndicesType&   init_neighbors_indices,
                                        const DistancesType& init_neighbors_distances,
                                        const UnionFindType& union_find_ref,
                                        const IndexType&     query_representative,
                                        const IndexType&     max_capacity = common::utils::infinity<IndexType>())
      : indices_{init_neighbors_indices}
      , distances_{init_neighbors_distances}
      , furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity > init_neighbors_indices.size() ? max_capacity : init_neighbors_indices.size()}
      , union_find_ref_{union_find_ref}
      , query_representative_{query_representative} {
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

    bool empty() const {
        return indices_.empty();
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        return indices_[furthest_buffer_index_];
    }

    DistanceType furthest_k_nearest_neighbor_distance() const {
        return furthest_k_nearest_neighbor_distance_;
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
        // consider an update only if the candidate is not in the same component as the representative of the component
        const bool is_candidate_valid =
            union_find_ref_.find(query_representative_) != union_find_ref_.find(index_candidate);

        if (is_candidate_valid) {
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

    void reset_buffers_except_memory() {
        // reset all the buffers to default values
        // max_capacity_ remains unchanged
        indices_.clear();
        distances_.clear();
        furthest_buffer_index_                = 0;
        furthest_k_nearest_neighbor_distance_ = 0;
    }

    void print() const {
        for (std::size_t index = 0; index < std::min(indices_.size(), distances_.size()); ++index) {
            std::cout << "(" << indices_[index] << ", " << distances_[index] << ")\n";
        }
    }

  private:
    std::vector<IndexType>    indices_;
    std::vector<DistanceType> distances_;
    IndexType                 furthest_buffer_index_;
    DistanceType              furthest_k_nearest_neighbor_distance_;
    IndexType                 max_capacity_;

    const UnionFindType& union_find_ref_;
    IndexType            query_representative_;
};

}  // namespace ffcl::knn::buffer