#pragma once

#include "ffcl/search/buffer/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::search::buffer {

template <typename IndicesIterator, typename DistancesIterator>
class Radius : public Base<IndicesIterator, DistancesIterator> {
  public:
    using IndexType     = typename Base<IndicesIterator, DistancesIterator>::IndexType;
    using DistanceType  = typename Base<IndicesIterator, DistancesIterator>::DistanceType;
    using IndicesType   = typename Base<IndicesIterator, DistancesIterator>::IndicesType;
    using DistancesType = typename Base<IndicesIterator, DistancesIterator>::DistancesType;

    using SamplesIterator = typename Base<IndicesIterator, DistancesIterator>::SamplesIterator;

    explicit Radius(const DistanceType& radius)
      : Radius(radius, {}, {}) {}

    explicit Radius(const DistanceType&  radius,
                    const IndicesType&   init_neighbors_indices,
                    const DistancesType& init_neighbors_distances)
      : radius_{radius}
      , indices_{init_neighbors_indices}
      , distances_{init_neighbors_distances} {}

    std::size_t size() const {
        return indices_.size();
    }

    std::size_t n_free_slots() const {
        return common::infinity<IndexType>();
    }

    bool empty() const {
        return indices_.empty();
    }

    IndexType upper_bound_index() const {
        throw std::runtime_error("No furthest index to return for this type of buffer.");
        return IndexType{};
    }

    DistanceType upper_bound() const {
        return radius_;
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        common::ignore_parameters(feature_index);
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

    std::tuple<IndicesType, DistancesType> move_data_to_indices_distances_pair() {
        return std::make_tuple(std::move(indices_), std::move(distances_));
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        if (distance_candidate < this->upper_bound()) {
            indices_.emplace_back(index_candidate);
            distances_.emplace_back(distance_candidate);
        }
    }

    void search(const IndicesIterator& indices_range_first,
                const IndicesIterator& indices_range_last,
                const SamplesIterator& samples_range_first,
                const SamplesIterator& samples_range_last,
                std::size_t            n_features,
                std::size_t            sample_index_query) {
        common::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_in_bounds_index = indices_range_first[index];

            const auto candidate_in_bounds_distance = common::math::heuristics::auto_distance(
                samples_range_first + sample_index_query * n_features,
                samples_range_first + sample_index_query * n_features + n_features,
                samples_range_first + candidate_in_bounds_index * n_features);

            this->update(candidate_in_bounds_index, candidate_in_bounds_distance);
        }
    }

    void search(const IndicesIterator& indices_range_first,
                const IndicesIterator& indices_range_last,
                const SamplesIterator& samples_range_first,
                const SamplesIterator& samples_range_last,
                std::size_t            n_features,
                const SamplesIterator& feature_query_range_first,
                const SamplesIterator& feature_query_range_last) {
        common::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_in_bounds_index = indices_range_first[index];

            const auto candidate_in_bounds_distance =
                common::math::heuristics::auto_distance(feature_query_range_first,
                                                        feature_query_range_last,
                                                        samples_range_first + candidate_in_bounds_index * n_features);

            this->update(candidate_in_bounds_index, candidate_in_bounds_distance);
        }
    }

    void print() const {
        for (std::size_t index = 0; index < std::min(indices_.size(), distances_.size()); ++index) {
            std::cout << "(" << indices_[index] << ", " << distances_[index] << ")\n";
        }
    }

  private:
    DistanceType  radius_;
    IndicesType   indices_;
    DistancesType distances_;
};

}  // namespace ffcl::search::buffer