#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/math/statistics/Statistics.hpp"

#include "ffcl/datastruct/UnionFind.hpp"

namespace ffcl::knn {

template <typename SamplesIterator>
class NearestNeighborsBufferBase {
  public:
    virtual ~NearestNeighborsBufferBase() {}

    using IndexType    = std::size_t;
    using DistanceType = typename SamplesIterator::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    virtual std::size_t size() const = 0;

    virtual std::size_t n_free_slots() const = 0;

    virtual bool empty() const = 0;

    virtual IndexType furthest_k_nearest_neighbor_index() const = 0;

    virtual DistanceType furthest_k_nearest_neighbor_distance() const = 0;

    virtual IndicesType indices() const = 0;

    virtual DistancesType distances() const = 0;

    virtual IndicesType move_indices() = 0;

    virtual DistancesType move_distances() = 0;

    virtual std::tuple<IndicesType, DistancesType> move_data_to_indices_distances_pair() = 0;

    virtual void update(const IndexType& index_candidate, const DistanceType& distance_candidate) = 0;

    virtual void print() const = 0;
};

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
template <typename SamplesIterator>
class NearestNeighborsBufferSorted : public NearestNeighborsBufferBase<SamplesIterator> {
  private:
    using IndexType    = typename NearestNeighborsBufferBase<SamplesIterator>::IndexType;
    using DistanceType = typename NearestNeighborsBufferBase<SamplesIterator>::DistanceType;

    using IndicesType   = typename NearestNeighborsBufferBase<SamplesIterator>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<SamplesIterator>::DistancesType;

  public:
    explicit NearestNeighborsBufferSorted(const IndicesType& max_capacity = common::utils::infinity<IndexType>())
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

template <typename SamplesIterator>
class NearestNeighborsBuffer : public NearestNeighborsBufferBase<SamplesIterator> {
  private:
    using IndexType    = typename NearestNeighborsBufferBase<SamplesIterator>::IndexType;
    using DistanceType = typename NearestNeighborsBufferBase<SamplesIterator>::DistanceType;

    using IndicesType   = typename NearestNeighborsBufferBase<SamplesIterator>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<SamplesIterator>::DistancesType;

  public:
    explicit NearestNeighborsBuffer(const IndexType& max_capacity = common::utils::infinity<IndexType>())
      : NearestNeighborsBuffer({}, {}, max_capacity) {}

    explicit NearestNeighborsBuffer(const IndicesType&   init_neighbors_indices,
                                    const DistancesType& init_neighbors_distances,
                                    const IndexType&     max_capacity = common::utils::infinity<IndexType>())
      : indices_{init_neighbors_indices}
      , distances_{init_neighbors_distances}
      , furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity > init_neighbors_indices.size() ? max_capacity : init_neighbors_indices.size()} {
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
    IndicesType   indices_;
    DistancesType distances_;
    IndexType     furthest_buffer_index_;
    DistanceType  furthest_k_nearest_neighbor_distance_;
    IndexType     max_capacity_;
};

template <typename SamplesIterator, typename UnionFindType = ffcl::datastruct::UnionFind<std::size_t>>
class NearestNeighborsBufferWithUnionFind : public NearestNeighborsBufferBase<SamplesIterator> {
  private:
    using IndexType    = typename NearestNeighborsBufferBase<SamplesIterator>::IndexType;
    using DistanceType = typename NearestNeighborsBufferBase<SamplesIterator>::DistanceType;

    using IndicesType   = typename NearestNeighborsBufferBase<SamplesIterator>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<SamplesIterator>::DistancesType;

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

template <typename SamplesIterator, typename VisitedIndicesType = std::unordered_set<std::size_t>>
class NearestNeighborsBufferWithMemory : public NearestNeighborsBufferBase<SamplesIterator> {
  private:
    using IndexType    = typename NearestNeighborsBufferBase<SamplesIterator>::IndexType;
    using DistanceType = typename NearestNeighborsBufferBase<SamplesIterator>::DistanceType;

    using IndicesType   = typename NearestNeighborsBufferBase<SamplesIterator>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<SamplesIterator>::DistancesType;

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

}  // namespace ffcl::knn