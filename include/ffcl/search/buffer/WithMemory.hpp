#pragma once

#include "ffcl/search/buffer/StaticBase.hpp"

#include "ffcl/datastruct/bounds/StaticBound.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::search::buffer {

template <typename DistancesIterator,
          typename Bound          = datastruct::bounds::UnboundedBallView<DistancesIterator>,
          typename VisitedIndices = std::unordered_set<std::size_t>>
class WithMemory : public StaticBase<WithMemory<DistancesIterator, Bound, VisitedIndices>> {
  public:
    static_assert(common::is_iterator<DistancesIterator>::value, "DistancesIterator is not an iterator");
    static_assert(common::is_crtp_of<Bound, datastruct::bounds::StaticBound>::value,
                  "Bound does not inherit from datastruct::bounds::StaticBound<Derived>");
    static_assert(common::is_std_container_v<VisitedIndices>, "VisitedIndices is not a standard container");

    using IndicesIteratorType   = typename VisitedIndices::iterator;
    using DistancesIteratorType = DistancesIterator;

    using IndexType    = typename std::iterator_traits<IndicesIteratorType>::value_type;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    explicit WithMemory(Bound&& bound, const IndexType& max_capacity = common::infinity<IndexType>())
      : bound_{std::forward<Bound>(bound)}
      , upper_bound_buffer_index_{0}
      , upper_bound_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{}
      , visited_indices_const_reference_{visited_indices_} {}

    WithMemory(Bound&&               bound,
               const VisitedIndices& visited_indices_reference,
               const IndexType&      max_capacity = common::infinity<IndexType>())
      : bound_{std::forward<Bound>(bound)}
      , upper_bound_buffer_index_{0}
      , upper_bound_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{}
      , visited_indices_const_reference_{visited_indices_reference} {}

    WithMemory(Bound&&          bound,
               VisitedIndices&& visited_indices,
               const IndexType& max_capacity = common::infinity<IndexType>())
      : bound_{std::forward<Bound>(bound)}
      , upper_bound_buffer_index_{0}
      , upper_bound_distance_{0}
      , max_capacity_{max_capacity}
      , visited_indices_{std::move(visited_indices)}
      , visited_indices_const_reference_{visited_indices_} {}

    WithMemory(Bound&&                    bound,
               const IndicesIteratorType& visited_indices_first,
               const IndicesIteratorType& visited_indices_last,
               const IndexType&           max_capacity = common::infinity<IndexType>())
      : WithMemory(std::move(bound), VisitedIndices(visited_indices_first, visited_indices_last), max_capacity) {}

    WithMemory(DistancesIteratorType centroid_features_query_first,
               DistancesIteratorType centroid_features_query_last,
               const IndexType&      max_capacity = common::infinity<IndexType>())
      : WithMemory(Bound{centroid_features_query_first, centroid_features_query_last}, max_capacity) {}

    WithMemory(DistancesIteratorType centroid_features_query_first,
               DistancesIteratorType centroid_features_query_last,
               const VisitedIndices& visited_indices_reference,
               const IndexType&      max_capacity = common::infinity<IndexType>())
      : WithMemory(Bound{centroid_features_query_first, centroid_features_query_last},
                   visited_indices_reference,
                   max_capacity) {}

    WithMemory(DistancesIteratorType centroid_features_query_first,
               DistancesIteratorType centroid_features_query_last,
               VisitedIndices&&      visited_indices,
               const IndexType&      max_capacity = common::infinity<IndexType>())
      : WithMemory(Bound{centroid_features_query_first, centroid_features_query_last},
                   std::move(visited_indices),
                   max_capacity) {}

    WithMemory(DistancesIteratorType      centroid_features_query_first,
               DistancesIteratorType      centroid_features_query_last,
               const IndicesIteratorType& visited_indices_first,
               const IndicesIteratorType& visited_indices_last,
               const IndexType&           max_capacity = common::infinity<IndexType>())
      : WithMemory(Bound{centroid_features_query_first, centroid_features_query_last},
                   visited_indices_first,
                   visited_indices_last,
                   max_capacity) {}

    constexpr auto centroid_begin_impl() const {
        return bound_.centroid_begin();
    }

    constexpr auto centroid_end_impl() const {
        return bound_.centroid_end();
    }

    auto indices_impl() const {
        return indices_;
    }

    auto distances_impl() const {
        return distances_;
    }

    const auto& const_reference_indices_impl() const& {
        return indices_;
    }

    const auto& const_reference_distances_impl() const& {
        return distances_;
    }

    auto&& move_indices_impl() && {
        return std::move(indices_);
    }

    auto&& move_distances_impl() && {
        return std::move(distances_);
    }

    std::size_t size_impl() const {
        return indices_.size();
    }

    std::size_t n_free_slots_impl() const {
        return max_capacity_ - size_impl();
    }

    bool empty_impl() const {
        return indices_.empty();
    }

    IndexType upper_bound_index_impl() const {
        return indices_[upper_bound_buffer_index_];
    }

    DistanceType upper_bound_impl() const {
        return upper_bound_distance_;
    }

    DistanceType upper_bound_impl(const IndexType& feature_index) const {
        common::ignore_parameters(feature_index);
        return upper_bound_impl();
    }

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // consider an update only if the index hasnt been visited
        if (visited_indices_const_reference_.find(index_candidate) == visited_indices_const_reference_.end()) {
            // always populate if the max capacity isnt reached
            if (n_free_slots_impl()) {
                indices_.emplace_back(index_candidate);
                distances_.emplace_back(distance_candidate);
                if (distance_candidate > upper_bound_impl()) {
                    // update the new index position of the furthest in the buffer
                    upper_bound_buffer_index_ = indices_.size() - 1;
                    upper_bound_distance_     = distance_candidate;
                }
            }
            // populate if the max capacity is reached and the candidate has a closer distance
            else if (distance_candidate < upper_bound_impl()) {
                // replace the previous greatest distance now that the vectors overflow the max capacity
                indices_[upper_bound_buffer_index_]   = index_candidate;
                distances_[upper_bound_buffer_index_] = distance_candidate;
                // find the new furthest neighbor and update the cache accordingly
                std::tie(upper_bound_buffer_index_, upper_bound_distance_) =
                    common::math::statistics::get_max_index_value_pair(distances_.begin(), distances_.end());
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

        const std::size_t n_subrange_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t subrange_index = 0; subrange_index < n_subrange_samples; ++subrange_index) {
            const std::size_t reference_index = indices_range_first[subrange_index];

            const auto optional_candidate_distance = bound_.compute_distance_if_within_bounds(
                samples_range_first + reference_index * n_features,
                samples_range_first + reference_index * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(reference_index, *optional_candidate_distance);
            }
        }
    }

  private:
    Bound bound_;

    IndicesType   indices_;
    DistancesType distances_;

    IndexType    upper_bound_buffer_index_;
    DistanceType upper_bound_distance_;

    IndexType max_capacity_;

    VisitedIndices        visited_indices_;
    const VisitedIndices& visited_indices_const_reference_;
};

template <typename Bound>
WithMemory(Bound &&) -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<std::size_t>>;

template <typename Bound, typename VisitedIndices>
WithMemory(Bound&&, const VisitedIndices&) -> WithMemory<typename Bound::IteratorType, Bound, VisitedIndices>;

template <typename Bound, typename IndicesIteratorType>
WithMemory(Bound&&, const IndicesIteratorType&, const IndicesIteratorType&)
    -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<std::size_t>>;

template <typename DistancesIteratorType>
WithMemory(DistancesIteratorType, DistancesIteratorType)
    -> WithMemory<DistancesIteratorType,
                  datastruct::bounds::UnboundedBallView<DistancesIteratorType>,
                  std::unordered_set<std::size_t>>;

template <typename DistancesIteratorType, typename VisitedIndices>
WithMemory(DistancesIteratorType, DistancesIteratorType, const VisitedIndices&)
    -> WithMemory<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>, VisitedIndices>;

template <typename DistancesIteratorType, typename VisitedIndices>
WithMemory(DistancesIteratorType, DistancesIteratorType, VisitedIndices &&)
    -> WithMemory<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>, VisitedIndices>;

template <typename DistancesIteratorType, typename IndicesIteratorType>
WithMemory(DistancesIteratorType, DistancesIteratorType, const IndicesIteratorType&, const IndicesIteratorType&)
    -> WithMemory<DistancesIteratorType,
                  datastruct::bounds::UnboundedBallView<DistancesIteratorType>,
                  std::unordered_set<std::size_t>>;

// ---

template <typename Bound, typename IndexType>
WithMemory(Bound&&, const IndexType&) -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<IndexType>>;

template <typename Bound, typename VisitedIndices, typename IndexType>
WithMemory(Bound&&, const VisitedIndices&, const IndexType&)
    -> WithMemory<typename Bound::IteratorType, Bound, VisitedIndices>;

template <typename Bound, typename IndicesIteratorType, typename IndexType>
WithMemory(Bound&&, const IndicesIteratorType&, const IndicesIteratorType&, const IndexType&)
    -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<IndexType>>;

template <typename DistancesIteratorType, typename IndexType>
WithMemory(DistancesIteratorType, DistancesIteratorType, const IndexType&)
    -> WithMemory<DistancesIteratorType,
                  datastruct::bounds::UnboundedBallView<DistancesIteratorType>,
                  std::unordered_set<IndexType>>;

template <typename DistancesIteratorType, typename VisitedIndices, typename IndexType>
WithMemory(DistancesIteratorType, DistancesIteratorType, const VisitedIndices&, const IndexType&)
    -> WithMemory<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>, VisitedIndices>;

template <typename DistancesIteratorType, typename VisitedIndices, typename IndexType>
WithMemory(DistancesIteratorType, DistancesIteratorType, VisitedIndices&&, const IndexType&)
    -> WithMemory<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>, VisitedIndices>;

template <typename DistancesIteratorType, typename IndicesIteratorType, typename IndexType>
WithMemory(DistancesIteratorType,
           DistancesIteratorType,
           const IndicesIteratorType&,
           const IndicesIteratorType&,
           const IndexType&) -> WithMemory<DistancesIteratorType,
                                           datastruct::bounds::UnboundedBallView<DistancesIteratorType>,
                                           std::unordered_set<IndexType>>;

}  // namespace ffcl::search::buffer
