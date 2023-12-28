#pragma once

#include "ffcl/search/count/StaticBase.hpp"

#include "ffcl/datastruct/bounds/StaticBound.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::search::count {

template <typename DistancesIterator, typename Bound>
class Counter : public StaticCBase<Counter<DistancesIterator, Bound>> {
  public:
    static_assert(common::is_iterator<DistancesIterator>::value, "DistancesIterator is not an iterator");

    static_assert(common::is_crtp_of<Bound, datastruct::bounds::StaticBound>::value,
                  "Bound does not inherit from datastruct::bounds::StaticBound<Derived>");

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    explicit Counter(Bound&& bound, const IndexType& max_capacity = common::infinity<IndexType>())
      : bound_{std::forward<Bound>(bound)}
      , upper_bound_distance_{0}
      , max_capacity_{max_capacity} {}

    explicit Counter(DistancesIterator centroid_features_query_first,
                     DistancesIterator centroid_features_query_last,
                     const IndexType&  max_capacity = common::infinity<IndexType>())
      : Counter{Bound{centroid_features_query_first, centroid_features_query_last}, max_capacity} {}

    constexpr auto centroid_begin_impl() const {
        return bound_.centroid_begin();
    }

    constexpr auto centroid_end_impl() const {
        return bound_.centroid_end();
    }

    auto count_impl() const {
        return count_;
    }

    auto n_free_slots_impl() const {
        return common::infinity<IndexType>();
    }

    DistanceType upper_bound_impl() const {
        return upper_bound_distance_;
    }

    DistanceType upper_bound_impl(const IndexType& feature_index) const {
        common::ignore_parameters(feature_index);
        return upper_bound_impl();
    }

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // should be replaced by radius check
        common::ignore_parameters(index_candidate);
        if (distance_candidate < bound_.upper_bound()) {
            ++count_;

            if (distance_candidate > upper_bound_distance_) {
                upper_bound_distance_ = distance_candidate;
            }
        }
    }

    template <typename IndicesIterator, typename SamplesIterator>
    void partial_search_impl(const IndicesIterator& indices_range_first,
                             const IndicesIterator& indices_range_last,
                             const SamplesIterator& samples_range_first,
                             const SamplesIterator& samples_range_last,
                             std::size_t            n_features) {
        ffcl::common::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t reference_index = indices_range_first[index];

            const auto optional_candidate_distance = bound_ptr_->compute_distance_within_bounds(
                samples_range_first + reference_index * n_features,
                samples_range_first + reference_index * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(reference_index, *optional_candidate_distance);
            }
        }
    }

  private:
    Bound bound_;

    DistanceType upper_bound_distance_;

    IndexType count_;
};

template <typename Bound>
Counter(Bound &&) -> Counter<typename Bound::IteratorType, Bound>;

template <typename Bound, typename IndexType>
Counter(Bound&&, const IndexType&) -> Counter<typename Bound::IteratorType, Bound>;

template <typename DistancesIteratorType, typename IndexType>
Counter(DistancesIteratorType, DistancesIteratorType, const IndexType&)
    -> Counter<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>>;

}  // namespace ffcl::search::count