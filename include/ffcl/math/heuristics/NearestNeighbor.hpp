#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/BoundingBox.hpp"
#include "ffcl/math/heuristics/Distances.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <vector>

template <typename SamplesIterator>
class NearestNeighborsBufferBase {
  public:
    virtual ~NearestNeighborsBufferBase() {}

    using IndexType    = std::size_t;
    using DistanceType = typename SamplesIterator::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    virtual std::size_t size() const = 0;

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
    explicit NearestNeighborsBufferSorted(std::size_t max_capacity = common::utils::infinity<IndexType>())
      : max_capacity_{max_capacity} {}

    std::size_t size() const {
        return indices_.size();
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
        if (indices_.size() < max_capacity_) {
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
    std::size_t   max_capacity_;
};

template <typename SamplesIterator>
class NearestNeighborsBuffer : public NearestNeighborsBufferBase<SamplesIterator> {
  private:
    using IndexType    = typename NearestNeighborsBufferBase<SamplesIterator>::IndexType;
    using DistanceType = typename NearestNeighborsBufferBase<SamplesIterator>::DistanceType;

    using IndicesType   = typename NearestNeighborsBufferBase<SamplesIterator>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<SamplesIterator>::DistancesType;

  public:
    explicit NearestNeighborsBuffer(std::size_t max_capacity = common::utils::infinity<IndexType>())
      : NearestNeighborsBuffer({}, {}, max_capacity) {}

    explicit NearestNeighborsBuffer(const std::vector<IndexType>&    init_neighbors_indices,
                                    const std::vector<DistanceType>& init_neighbors_distances,
                                    std::size_t max_capacity = common::utils::infinity<IndexType>())
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
        if (indices_.size() < max_capacity_) {
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
    std::size_t               max_capacity_;
};

template <typename SamplesIterator>
class NearestNeighborsBufferWithMemory : public NearestNeighborsBufferBase<SamplesIterator> {
  private:
    using IndexType    = typename NearestNeighborsBufferBase<SamplesIterator>::IndexType;
    using DistanceType = typename NearestNeighborsBufferBase<SamplesIterator>::DistanceType;

    using IndicesType   = typename NearestNeighborsBufferBase<SamplesIterator>::IndicesType;
    using DistancesType = typename NearestNeighborsBufferBase<SamplesIterator>::DistancesType;

  public:
    explicit NearestNeighborsBufferWithMemory(std::size_t max_capacity = common::utils::infinity<IndexType>())
      : furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity} {}

    explicit NearestNeighborsBufferWithMemory(const std::vector<IndexType>& visited_indices,
                                              std::size_t max_capacity = common::utils::infinity<IndexType>())
      : furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{visited_indices.begin(), visited_indices.end()} {}

    template <typename VisitedIndicesIterator>
    explicit NearestNeighborsBufferWithMemory(const VisitedIndicesIterator& visited_indices_first,
                                              const VisitedIndicesIterator& visited_indices_last,
                                              std::size_t max_capacity = common::utils::infinity<IndexType>())
      : furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{visited_indices_first, visited_indices_last} {}

    explicit NearestNeighborsBufferWithMemory(const std::vector<IndexType>&    init_neighbors_indices,
                                              const std::vector<DistanceType>& init_neighbors_distances,
                                              std::size_t max_capacity = common::utils::infinity<IndexType>())
      : indices_{init_neighbors_indices}
      , distances_{init_neighbors_distances}
      , furthest_buffer_index_{0}
      , furthest_k_nearest_neighbor_distance_{0}
      , max_capacity_{max_capacity > init_neighbors_indices.size() ? max_capacity : init_neighbors_indices.size()}
      , visited_indices_{init_neighbors_indices.begin(), init_neighbors_indices.end()} {
        if (indices_.size()) {
            if (indices_.size() == distances_.size()) {
                std::tie(furthest_buffer_index_, furthest_k_nearest_neighbor_distance_) =
                    math::statistics::get_max_index_value_pair(distances_.begin(), distances_.end());

            } else {
                throw std::runtime_error("Indices and distances buffers sizes do not match.");
            }
        }
    }

    NearestNeighborsBufferWithMemory(const NearestNeighborsBufferWithMemory& other)
      : indices_{other.indices_}
      , distances_{other.distances_}
      , furthest_buffer_index_{other.furthest_buffer_index_}
      , furthest_k_nearest_neighbor_distance_{other.furthest_k_nearest_neighbor_distance_}
      , max_capacity_{other.max_capacity_}
      , visited_indices_{other.visited_indices_} {}

    NearestNeighborsBufferWithMemory(NearestNeighborsBufferWithMemory&& other) noexcept
      : indices_{std::move(other.indices_)}
      , distances_{std::move(other.distances_)}
      , furthest_buffer_index_{std::move(other.furthest_buffer_index_)}
      , furthest_k_nearest_neighbor_distance_{std::move(other.furthest_k_nearest_neighbor_distance_)}
      , max_capacity_{std::move(other.max_capacity_)}
      , visited_indices_{std::move(other.visited_indices_)} {}

    NearestNeighborsBufferWithMemory& operator=(const NearestNeighborsBufferWithMemory& other) {
        if (this == &other) {
            return *this;
        }
        indices_                              = other.indices_;
        distances_                            = other.distances_;
        furthest_buffer_index_                = other.furthest_buffer_index_;
        furthest_k_nearest_neighbor_distance_ = other.furthest_k_nearest_neighbor_distance_;
        max_capacity_                         = other.max_capacity_;
        visited_indices_                      = other.visited_indices_;

        return *this;
    }

    std::size_t size() const {
        return indices_.size();
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
        if (visited_indices_.find(index_candidate) == visited_indices_.end()) {
            // always populate if the max capacity isnt reached
            if (indices_.size() < max_capacity_) {
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
            // add the candidate to the set of visited indices
            // visited_indices_.insert(index_candidate);
        }
    }

    void print() const {
        for (std::size_t index = 0; index < std::min(indices_.size(), distances_.size()); ++index) {
            std::cout << "(" << indices_[index] << ", " << distances_[index] << ")\n";
        }
    }

  private:
    std::vector<IndexType>        indices_;
    std::vector<DistanceType>     distances_;
    IndexType                     furthest_buffer_index_;
    DistanceType                  furthest_k_nearest_neighbor_distance_;
    std::size_t                   max_capacity_;
    std::unordered_set<IndexType> visited_indices_;
};

template <typename SamplesIterator>
class NearestNeighborsBufferOld {
  private:
    using IndexType       = std::size_t;
    using DistanceType    = typename SamplesIterator::value_type;
    using ElementDataType = typename std::tuple<IndexType, DistanceType>;

    struct ComparisonFunctor {
        bool operator()(const ElementDataType& left_element, const ElementDataType& right_element) const {
            // the greatest element will appear at the top of the priority queue
            return std::get<1>(left_element) < std::get<1>(right_element);
        }
    };

    static constexpr ComparisonFunctor comparison_functor{};

    using PriorityQueueType =
        typename std::priority_queue<ElementDataType, std::vector<ElementDataType>, ComparisonFunctor>;

  public:
    explicit NearestNeighborsBufferOld()
      : max_elements_{1}
      , priority_queue_{comparison_functor} {}

    explicit NearestNeighborsBufferOld(std::size_t max_elements)
      : max_elements_{max_elements}
      , priority_queue_{comparison_functor} {}

    std::size_t size() const {
        return priority_queue_.size();
    }

    bool empty() const {
        return priority_queue_.empty();
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        return std::get<0>(priority_queue_.top());
    }

    DistanceType furthest_k_nearest_neighbor_distance() const {
        return std::get<1>(priority_queue_.top());
    }

    auto extract_indices() {
        std::vector<IndexType> extracted_indices;
        extracted_indices.reserve(priority_queue_.size());

        while (!priority_queue_.empty()) {
            extracted_indices.emplace_back(this->pop_and_get_index());
        }
        return extracted_indices;
    }

    auto pop_and_get_index() {
        const auto last_value_index = std::get<0>(priority_queue_.top());
        priority_queue_.pop();
        return last_value_index;
    }

    auto pop_and_get_distance() {
        const auto last_value_distance = std::get<1>(priority_queue_.top());
        priority_queue_.pop();
        return last_value_distance;
    }

    auto pop_and_get_index_distance_pair() {
        const auto index_distance_pair = priority_queue_.top();
        priority_queue_.pop();
        return index_distance_pair;
    }

    auto move_data_to_indices_distances_pair() {
        std::vector<IndexType> indices;
        indices.reserve(this->size());
        std::vector<DistanceType> distances;
        distances.reserve(this->size());

        while (!priority_queue_.empty()) {
            const auto [index, distance] = priority_queue_.top();
            indices.emplace_back(index);
            distances.emplace_back(distance);
            priority_queue_.pop();
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    auto copy_data_to_indices_distances_pair() const {
        PriorityQueueType priority_queue_cpy = priority_queue_;

        std::vector<IndexType> indices;
        indices.reserve(this->size());
        std::vector<DistanceType> distances;
        distances.reserve(this->size());

        while (!priority_queue_cpy.empty()) {
            const auto [index, distance] = priority_queue_cpy.top();
            indices.emplace_back(index);
            distances.emplace_back(distance);
            priority_queue_cpy.pop();
        }
        return std::make_tuple(std::move(indices), std::move(distances));
    }

    void emplace_back(ElementDataType&& index_distance_pair) {
        priority_queue_.emplace(std::move(index_distance_pair));
        ++max_elements_;
    }

    void add(NearestNeighborsBufferOld&& nearest_neighbors_buffer) {
        while (!nearest_neighbors_buffer.empty()) {
            this->emplace_back(std::move(nearest_neighbors_buffer.pop_and_get_index_distance_pair()));
        }
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // populate the priority queue if theres still empty space
        if (priority_queue_.size() < max_elements_) {
            priority_queue_.emplace(std::make_tuple(index_candidate, distance_candidate));

        } else if (distance_candidate < this->furthest_k_nearest_neighbor_distance()) {
            // pop the element at the top of the priority queue if the candidate is smaller
            priority_queue_.pop();
            // then place the candidate index-distance pair accordingly
            priority_queue_.emplace(std::make_tuple(index_candidate, distance_candidate));
        }
    }

    void print() const {
        PriorityQueueType priority_queue_cpy = priority_queue_;

        while (!priority_queue_cpy.empty()) {
            auto element = priority_queue_cpy.top();
            priority_queue_cpy.pop();
            std::cout << "(" << std::get<0>(element) << ", " << std::get<1>(element) << ")\n";
        }
    }

  private:
    std::size_t       max_elements_;
    PriorityQueueType priority_queue_;
};

namespace math::heuristics {

template <typename SamplesIterator>
void nearest_neighbor_range(const SamplesIterator&                subrange_samples_first,
                            const SamplesIterator&                subrange_samples_last,
                            const SamplesIterator&                dataset_samples_first,
                            const SamplesIterator&                dataset_samples_last,
                            std::size_t                           n_features,
                            std::size_t                           sample_index_query,
                            ssize_t&                              current_nearest_neighbor_index,
                            typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
                current_nearest_neighbor_index    = subrange_offset + subrange_candidate_index;
                current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
            }
        }
    }
}

template <typename SamplesIterator>
void k_nearest_neighbors_range(const SamplesIterator&                       subrange_samples_first,
                               const SamplesIterator&                       subrange_samples_last,
                               const SamplesIterator&                       dataset_samples_first,
                               const SamplesIterator&                       dataset_samples_last,
                               std::size_t                                  n_features,
                               std::size_t                                  sample_index_query,
                               NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            nearest_neighbors_buffer.update(subrange_offset + subrange_candidate_index,
                                            candidate_nearest_neighbor_distance);
        }
    }
}

template <typename SamplesIterator>
void increment_neighbors_count_in_radius_range(const SamplesIterator&                      subrange_samples_first,
                                               const SamplesIterator&                      subrange_samples_last,
                                               const SamplesIterator&                      dataset_samples_first,
                                               const SamplesIterator&                      dataset_samples_last,
                                               std::size_t                                 n_features,
                                               std::size_t                                 sample_index_query,
                                               const typename SamplesIterator::value_type& radius,
                                               std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                ++neighbors_count;
            }
        }
    }
}

template <typename SamplesIterator>
void k_nearest_neighbors_in_radius_range(const SamplesIterator&                       subrange_samples_first,
                                         const SamplesIterator&                       subrange_samples_last,
                                         const SamplesIterator&                       dataset_samples_first,
                                         const SamplesIterator&                       dataset_samples_last,
                                         std::size_t                                  n_features,
                                         std::size_t                                  sample_index_query,
                                         const typename SamplesIterator::value_type&  radius,
                                         NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    // number of samples in the subrange
    const std::size_t n_samples =
        common::utils::get_n_samples(subrange_samples_first, subrange_samples_last, n_features);

    // global index of the subrange in the entire dataset
    const std::size_t subrange_offset =
        common::utils::get_n_samples(dataset_samples_first, subrange_samples_first, n_features);

    for (std::size_t subrange_candidate_index = 0; subrange_candidate_index < n_samples; ++subrange_candidate_index) {
        if (subrange_offset + subrange_candidate_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                subrange_samples_first + subrange_candidate_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                nearest_neighbors_buffer.update(subrange_offset + subrange_candidate_index,
                                                candidate_nearest_neighbor_distance);
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor_indexed_range(const IndicesIterator&                subrange_index_first,
                                    const IndicesIterator&                subrange_index_last,
                                    const SamplesIterator&                dataset_samples_first,
                                    const SamplesIterator&                dataset_samples_last,
                                    std::size_t                           n_features,
                                    std::size_t                           sample_index_query,
                                    ssize_t&                              current_nearest_neighbor_index,
                                    typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
                current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
                current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void nearest_neighbor_indexed_range(const IndicesIterator&                subrange_index_first,
                                    const IndicesIterator&                subrange_index_last,
                                    const SamplesIterator&                dataset_samples_first,
                                    const SamplesIterator&                dataset_samples_last,
                                    std::size_t                           n_features,
                                    const SamplesIterator&                sample_feature_query_first,
                                    const SamplesIterator&                sample_feature_query_last,
                                    ssize_t&                              current_nearest_neighbor_index,
                                    typename SamplesIterator::value_type& current_nearest_neighbor_distance) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < current_nearest_neighbor_distance) {
            current_nearest_neighbor_index    = candidate_nearest_neighbor_index;
            current_nearest_neighbor_distance = candidate_nearest_neighbor_distance;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_indexed_range(const IndicesIterator&                       subrange_index_first,
                                       const IndicesIterator&                       subrange_index_last,
                                       const SamplesIterator&                       dataset_samples_first,
                                       const SamplesIterator&                       dataset_samples_last,
                                       std::size_t                                  n_features,
                                       std::size_t                                  sample_index_query,
                                       NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_indexed_range(const IndicesIterator&                       subrange_index_first,
                                       const IndicesIterator&                       subrange_index_last,
                                       const SamplesIterator&                       dataset_samples_first,
                                       const SamplesIterator&                       dataset_samples_last,
                                       std::size_t                                  n_features,
                                       const SamplesIterator&                       sample_feature_query_first,
                                       const SamplesIterator&                       sample_feature_query_last,
                                       NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius_indexed_range(const IndicesIterator& subrange_index_first,
                                                       const IndicesIterator& subrange_index_last,
                                                       const SamplesIterator& dataset_samples_first,
                                                       const SamplesIterator& dataset_samples_last,
                                                       std::size_t            n_features,
                                                       std::size_t            sample_index_query,
                                                       const typename SamplesIterator::value_type& radius,
                                                       std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                ++neighbors_count;
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_radius_indexed_range(const IndicesIterator& subrange_index_first,
                                                       const IndicesIterator& subrange_index_last,
                                                       const SamplesIterator& dataset_samples_first,
                                                       const SamplesIterator& dataset_samples_last,
                                                       std::size_t            n_features,
                                                       const SamplesIterator& sample_feature_query_first,
                                                       const SamplesIterator& sample_feature_query_last,
                                                       const typename SamplesIterator::value_type& radius,
                                                       std::size_t&                                neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                       subrange_index_first,
    const IndicesIterator&                       subrange_index_last,
    const SamplesIterator&                       dataset_samples_first,
    const SamplesIterator&                       dataset_samples_last,
    std::size_t                                  n_features,
    std::size_t                                  sample_index_query,
    const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    std::size_t&                                 neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void increment_neighbors_count_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                       subrange_index_first,
    const IndicesIterator&                       subrange_index_last,
    const SamplesIterator&                       dataset_samples_first,
    const SamplesIterator&                       dataset_samples_last,
    std::size_t                                  n_features,
    const SamplesIterator&                       sample_feature_query_first,
    const SamplesIterator&                       sample_feature_query_last,
    const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    std::size_t&                                 neighbors_count) {
    common::utils::ignore_parameters(dataset_samples_last, sample_feature_query_first, sample_feature_query_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            ++neighbors_count;
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius_indexed_range(
    const IndicesIterator&                       subrange_index_first,
    const IndicesIterator&                       subrange_index_last,
    const SamplesIterator&                       dataset_samples_first,
    const SamplesIterator&                       dataset_samples_last,
    std::size_t                                  n_features,
    std::size_t                                  sample_index_query,
    const typename SamplesIterator::value_type&  radius,
    NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            if (candidate_nearest_neighbor_distance < radius) {
                nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_radius_indexed_range(
    const IndicesIterator&                       subrange_index_first,
    const IndicesIterator&                       subrange_index_last,
    const SamplesIterator&                       dataset_samples_first,
    const SamplesIterator&                       dataset_samples_last,
    std::size_t                                  n_features,
    const SamplesIterator&                       sample_feature_query_first,
    const SamplesIterator&                       sample_feature_query_last,
    const typename SamplesIterator::value_type&  radius,
    NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        const auto candidate_nearest_neighbor_distance =
            math::heuristics::auto_distance(sample_feature_query_first,
                                            sample_feature_query_last,
                                            dataset_samples_first + candidate_nearest_neighbor_index * n_features);

        if (candidate_nearest_neighbor_distance < radius) {
            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                       subrange_index_first,
    const IndicesIterator&                       subrange_index_last,
    const SamplesIterator&                       dataset_samples_first,
    const SamplesIterator&                       dataset_samples_last,
    std::size_t                                  n_features,
    std::size_t                                  sample_index_query,
    const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (candidate_nearest_neighbor_index != sample_index_query &&
            bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(dataset_samples_first + sample_index_query * n_features,
                                                dataset_samples_first + sample_index_query * n_features + n_features,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

template <typename IndicesIterator, typename SamplesIterator>
void k_nearest_neighbors_in_kd_bounding_box_indexed_range(
    const IndicesIterator&                       subrange_index_first,
    const IndicesIterator&                       subrange_index_last,
    const SamplesIterator&                       dataset_samples_first,
    const SamplesIterator&                       dataset_samples_last,
    std::size_t                                  n_features,
    const SamplesIterator&                       sample_feature_query_first,
    const SamplesIterator&                       sample_feature_query_last,
    const bbox::HyperRangeType<SamplesIterator>& kd_bounding_box,
    NearestNeighborsBufferBase<SamplesIterator>& nearest_neighbors_buffer) {
    common::utils::ignore_parameters(dataset_samples_last);

    const std::size_t n_samples = std::distance(subrange_index_first, subrange_index_last);

    for (std::size_t index = 0; index < n_samples; ++index) {
        const std::size_t candidate_nearest_neighbor_index = subrange_index_first[index];

        if (bbox::is_sample_in_kd_bounding_box(
                dataset_samples_first + candidate_nearest_neighbor_index * n_features,
                dataset_samples_first + candidate_nearest_neighbor_index * n_features + n_features,
                kd_bounding_box)) {
            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(sample_feature_query_first,
                                                sample_feature_query_last,
                                                dataset_samples_first + candidate_nearest_neighbor_index * n_features);

            nearest_neighbors_buffer.update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }
}

}  // namespace math::heuristics