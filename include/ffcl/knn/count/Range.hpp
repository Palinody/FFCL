#pragma once

#include "ffcl/knn/count/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/datastruct/BoundingBox.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::knn::count {

template <typename IndicesIterator, typename DistancesIterator>
class Range : public Base<IndicesIterator, DistancesIterator> {
  public:
    using IndexType    = typename Base<IndicesIterator, DistancesIterator>::IndexType;
    using DistanceType = typename Base<IndicesIterator, DistancesIterator>::DistanceType;

    using HyperRangeType = datastruct::bbox::HyperRangeType<typename std::vector<DistanceType>::iterator>;

    using SamplesIterator = typename Base<IndicesIterator, DistancesIterator>::SamplesIterator;

    explicit Range(const HyperRangeType& kd_bounding_box)
      : kd_bounding_box_{kd_bounding_box} {}

    std::size_t n_free_slots() const {
        return common::infinity<IndexType>();
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
        common::ignore_parameters(index_candidate, distance_candidate);
        ++count_;
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
            const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

            if (candidate_nearest_neighbor_index != sample_index_query &&
                datastruct::bbox::is_sample_in_kd_bounding_box(
                    samples_range_first + candidate_nearest_neighbor_index * n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                    kd_bounding_box_)) {
                const auto candidate_nearest_neighbor_distance = common::math::heuristics::auto_distance(
                    samples_range_first + sample_index_query * n_features,
                    samples_range_first + sample_index_query * n_features + n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features);

                this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
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
            const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

            const auto candidate_nearest_neighbor_distance = common::math::heuristics::auto_distance(
                feature_query_range_first,
                feature_query_range_last,
                samples_range_first + candidate_nearest_neighbor_index * n_features);

            if (datastruct::bbox::is_sample_in_kd_bounding_box(
                    samples_range_first + candidate_nearest_neighbor_index * n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                    kd_bounding_box_)) {
                this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
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