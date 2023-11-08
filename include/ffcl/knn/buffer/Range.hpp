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

template <typename IndicesIterator, typename DistancesIterator>
class Range : public Base<IndicesIterator, DistancesIterator> {
  public:
    using IndexType     = typename Base<IndicesIterator, DistancesIterator>::IndexType;
    using DistanceType  = typename Base<IndicesIterator, DistancesIterator>::DistanceType;
    using IndicesType   = typename Base<IndicesIterator, DistancesIterator>::IndicesType;
    using DistancesType = typename Base<IndicesIterator, DistancesIterator>::DistancesType;

    using HyperRangeType = bbox::HyperRangeType<DistancesIterator>;

    using SamplesIterator = typename Base<IndicesIterator, DistancesIterator>::SamplesIterator;

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
        indices_.emplace_back(index_candidate);
        distances_.emplace_back(distance_candidate);
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

            if (candidate_nearest_neighbor_index != sample_index_query &&
                bbox::is_sample_in_kd_bounding_box(
                    samples_range_first + candidate_nearest_neighbor_index * n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                    kd_bounding_box_)) {
                const auto candidate_nearest_neighbor_distance = math::heuristics::auto_distance(
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

            const auto candidate_nearest_neighbor_distance =
                math::heuristics::auto_distance(feature_query_range_first,
                                                feature_query_range_last,
                                                samples_range_first + candidate_nearest_neighbor_index * n_features);

            if (bbox::is_sample_in_kd_bounding_box(
                    samples_range_first + candidate_nearest_neighbor_index * n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features + n_features,
                    kd_bounding_box_)) {
                this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
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