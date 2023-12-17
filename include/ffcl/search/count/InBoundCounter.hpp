#pragma once

#include "ffcl/search/count/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::search::count {

template <typename IndicesIterator, typename DistancesIterator>
class InBoundCounter : public StaticBase<InBoundCounter<IndicesIterator, DistancesIterator>> {
  public:
    auto count_impl() const {
        return count_;
    }

    auto n_free_slots_impl() const {
        return common::infinity<IndexType>();
    }

    auto upper_bound_impl() const {
        return radius_;
    }

    auto upper_bound_impl(const IndexType& feature_index) const {
        common::ignore_parameters(feature_index);
        return upper_bound_impl();
    }

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // should be replaced by radius check
        common::ignore_parameters(index_candidate);
        if (distance_candidate < upper_bound_impl()) {
            ++count_;
        }
    }

    void partial_search_impl(const IndicesIteratorType& indices_range_first,
                             const IndicesIteratorType& indices_range_last,
                             const SamplesIteratorType& samples_range_first,
                             const SamplesIteratorType& samples_range_last,
                             std::size_t                n_features) {
        ffcl::common::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_in_bounds_index = indices_range_first[index];

            const auto optional_candidate_distance = bound_ptr_->compute_distance_within_bounds(
                samples_range_first + candidate_in_bounds_index * n_features,
                samples_range_first + candidate_in_bounds_index * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(candidate_in_bounds_index, *optional_candidate_distance);
            }
        }
    }

  private:
    DistanceType radius_;
    IndexType    count_;
};

}  // namespace ffcl::search::count