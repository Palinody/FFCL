#pragma once

#include "ffcl/knn/buffer/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

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
template <typename IndicesIterator, typename DistancesIterator>
class Sorted : public Base<IndicesIterator, DistancesIterator> {
  public:
    using IndexType     = typename Base<IndicesIterator, DistancesIterator>::IndexType;
    using DistanceType  = typename Base<IndicesIterator, DistancesIterator>::DistanceType;
    using IndicesType   = typename Base<IndicesIterator, DistancesIterator>::IndicesType;
    using DistancesType = typename Base<IndicesIterator, DistancesIterator>::DistancesType;

    using SamplesIterator = typename Base<IndicesIterator, DistancesIterator>::SamplesIterator;

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

    DistanceType upper_bound() const {
        return distances_.back();
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        common::utils::ignore_parameters(feature_index);
        return this->upper_bound();
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

    void operator()(const IndicesIterator& indices_range_first,
                    const IndicesIterator& indices_range_last,
                    const SamplesIterator& samples_range_first,
                    const SamplesIterator& samples_range_last,
                    std::size_t            n_features,
                    std::size_t            sample_index_query) {
        common::utils::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

            if (candidate_nearest_neighbor_index != sample_index_query) {
                const auto candidate_nearest_neighbor_distance = common::math::heuristics::auto_distance(
                    samples_range_first + sample_index_query * n_features,
                    samples_range_first + sample_index_query * n_features + n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features);

                this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
        }
    }

    void operator()(const IndicesIterator& indices_range_first,
                    const IndicesIterator& indices_range_last,
                    const SamplesIterator& samples_range_first,
                    const SamplesIterator& samples_range_last,
                    std::size_t            n_features,
                    const SamplesIterator& feature_query_range_first,
                    const SamplesIterator& feature_query_range_last) {
        common::utils::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

            const auto candidate_nearest_neighbor_distance = common::math::heuristics::auto_distance(
                feature_query_range_first,
                feature_query_range_last,
                samples_range_first + candidate_nearest_neighbor_index * n_features);

            this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
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