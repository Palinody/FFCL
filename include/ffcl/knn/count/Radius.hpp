#pragma once

#include "ffcl/knn/count/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::count {

template <typename IndicesIterator, typename DistancesIterator>
class Radius : public Base<IndicesIterator, DistancesIterator> {
  public:
    using IndexType    = typename Base<IndicesIterator, DistancesIterator>::IndexType;
    using DistanceType = typename Base<IndicesIterator, DistancesIterator>::DistanceType;

    using SamplesIterator = typename Base<IndicesIterator, DistancesIterator>::SamplesIterator;

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
        std::cout << "count: " << count_ << ", radius: " << radius_ << "\n";
    }

  private:
    DistanceType radius_;
    IndexType    count_;
};

}  // namespace ffcl::knn::count