#pragma once

#include "ffcl/search/buffer/StaticBase.hpp"

#include "ffcl/datastruct/bounds/StaticBound.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/datastruct/UnionFind.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>
#include <vector>

namespace ffcl::search::buffer {

template <typename DistancesIterator, typename Bound = datastruct::bounds::UnboundedBallView<DistancesIterator>>
class WithUnionFind : public StaticBase<WithUnionFind<DistancesIterator, Bound>> {
  public:
    static_assert(common::is_iterator<DistancesIterator>::value, "DistancesIterator is not an iterator");
    static_assert(common::is_crtp_of<Bound, datastruct::bounds::StaticBound>::value,
                  "Bound does not inherit from datastruct::bounds::StaticBound<Derived>");

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using UnionFindConstReferenceType = const datastruct::UnionFind<IndexType>&;

    WithUnionFind(Bound&&                     bound,
                  UnionFindConstReferenceType union_find_const_reference,
                  const IndexType&            query_representative,
                  const IndexType&            max_capacity = common::infinity<IndexType>())
      : bound_{std::forward<Bound>(bound)}
      , indices_{}
      , distances_{}
      , upper_bound_buffer_index_{0}
      , upper_bound_distance_{0}
      , max_capacity_{max_capacity}
      , union_find_const_reference_{union_find_const_reference}
      , query_representative_{query_representative} {}

    WithUnionFind(DistancesIterator           centroid_features_query_first,
                  DistancesIterator           centroid_features_query_last,
                  UnionFindConstReferenceType union_find_const_reference,
                  const IndexType&            query_representative,
                  const IndexType&            max_capacity = common::infinity<IndexType>())
      : WithUnionFind(Bound(centroid_features_query_first, centroid_features_query_last),
                      union_find_const_reference,
                      query_representative,
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
        // consider an update only if the candidate is not in the same component as the representative of the component
        const bool is_candidate_valid = union_find_const_reference_.find(query_representative_) !=
                                        union_find_const_reference_.find(index_candidate);

        if (is_candidate_valid) {
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

        // reference_index_it is an iterator that iterates over all the indices from the indices_range_first to
        // indices_range_last range
        for (auto reference_index_it = indices_range_first; reference_index_it != indices_range_last;
             ++reference_index_it) {
            const auto optional_candidate_distance = bound_.compute_distance_if_within_bounds(
                samples_range_first + *reference_index_it * n_features,
                samples_range_first + *reference_index_it * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(*reference_index_it, *optional_candidate_distance);
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

    UnionFindConstReferenceType union_find_const_reference_;
    IndexType                   query_representative_;
};

template <typename Bound, typename UnionFindConstReferenceType, typename IndexType>
WithUnionFind(Bound&&, UnionFindConstReferenceType, const IndexType&)
    -> WithUnionFind<typename Bound::IteratorType, Bound>;

template <typename DistancesIteratorType, typename UnionFindConstReferenceType, typename IndexType>
WithUnionFind(DistancesIteratorType, DistancesIteratorType, UnionFindConstReferenceType, const IndexType&)
    -> WithUnionFind<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>>;

// ---

template <typename Bound, typename UnionFindConstReferenceType, typename IndexType>
WithUnionFind(Bound&&, UnionFindConstReferenceType, const IndexType&, const IndexType&)
    -> WithUnionFind<typename Bound::IteratorType, Bound>;

template <typename DistancesIteratorType, typename UnionFindConstReferenceType, typename IndexType>
WithUnionFind(DistancesIteratorType,
              DistancesIteratorType,
              UnionFindConstReferenceType,
              const IndexType&,
              const IndexType&)
    -> WithUnionFind<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>>;

}  // namespace ffcl::search::buffer