#pragma once

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/UnionFind.hpp"

#include <optional>
#include <vector>

namespace ffcl::search::buffer {

template <typename DistancesIterator, typename Bound = datastruct::bounds::UnboundedBallView<DistancesIterator>>
class WithUnionFind : public StaticBuffer<WithUnionFind<DistancesIterator, Bound>> {
  public:
    using BoundType = Bound;

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using IndicesIteratorType   = typename IndicesType::iterator;
    using DistancesIteratorType = DistancesIterator;

    using UnionFindConstReference = const datastruct::UnionFind<IndexType>&;

    WithUnionFind(BoundType&&             bound,
                  UnionFindConstReference union_find_const_ref,
                  const IndexType&        query_representative,
                  const IndexType&        max_capacity = common::infinity<IndexType>())
      : StaticBuffer<WithUnionFind<DistancesIterator, BoundType>>(std::forward<BoundType>(bound), max_capacity)
      , union_find_const_ref_{union_find_const_ref}
      , query_representative_{query_representative} {}

    WithUnionFind(const DistancesIterator& centroid_features_query_first,
                  const DistancesIterator& centroid_features_query_last,
                  UnionFindConstReference  union_find_const_ref,
                  const IndexType&         query_representative,
                  const IndexType&         max_capacity = common::infinity<IndexType>())
      : WithUnionFind(BoundType(centroid_features_query_first, centroid_features_query_last),
                      union_find_const_ref,
                      query_representative,
                      max_capacity) {}

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        const auto candidate_representative = union_find_const_ref_.find(index_candidate);

        // consider an update only if the candidate is not in the same component as the representative of the component
        const bool not_in_same_component = query_representative_ != candidate_representative;

        if (not_in_same_component) {
            this->try_update_static_buffers(index_candidate, distance_candidate);
        }
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    void partial_search_impl(const OtherIndicesIterator& indices_range_first,
                             const OtherIndicesIterator& indices_range_last,
                             const OtherSamplesIterator& samples_range_first,
                             const OtherSamplesIterator& samples_range_last,
                             std::size_t                 n_features) {
        ffcl::common::ignore_parameters(samples_range_last);

        for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
            const auto optional_candidate_distance = this->bound_.compute_distance_to_centroid_if_within_bounds(
                samples_range_first + *index_it * n_features,
                samples_range_first + *index_it * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(*index_it, *optional_candidate_distance);
            }
        }
    }

    template <typename OptionalType>
    void update_impl(const IndexType&             index_candidate,
                     const DistanceType&          distance_candidate,
                     std::optional<OptionalType>& candidate_representative) {
        candidate_representative = union_find_const_ref_.find(index_candidate);

        // consider an update only if the candidate is not in the same component as the representative of the component
        const bool not_in_same_component = query_representative_ != candidate_representative;

        if (not_in_same_component) {
            this->try_update_static_buffers(index_candidate, distance_candidate);
        }
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator, typename OptionalType>
    void partial_search_impl(const OtherIndicesIterator&  indices_range_first,
                             const OtherIndicesIterator&  indices_range_last,
                             const OtherSamplesIterator&  samples_range_first,
                             const OtherSamplesIterator&  samples_range_last,
                             std::size_t                  n_features,
                             std::optional<OptionalType>& candidates_component_membership) {
        ffcl::common::ignore_parameters(samples_range_last);

        for (auto index_it = indices_range_first; index_it != indices_range_last; ++index_it) {
            const auto optional_candidate_distance = this->bound_.compute_distance_to_centroid_if_within_bounds(
                samples_range_first + *index_it * n_features,
                samples_range_first + *index_it * n_features + n_features);

            if (optional_candidate_distance) {
                auto component_membership = std::optional<OptionalType>{std::nullopt};

                update_impl(*index_it, *optional_candidate_distance, component_membership);

                // Store the first encountered component membership if no value was saved yet.
                if (index_it == indices_range_first) {
                    candidates_component_membership = component_membership;

                } else if (component_membership != candidates_component_membership) {
                    // If the current component_membership differs from the first, mark that they are not all the same
                    candidates_component_membership = std::nullopt;
                }
            }
        }
    }

  private:
    UnionFindConstReference union_find_const_ref_;
    IndexType               query_representative_;
};

// Declare and define a static_base_traits specialization for WithUnionFind:
template <typename DistancesIterator, typename Bound>
struct static_base_traits<WithUnionFind<DistancesIterator, Bound>> {
    using BoundType = Bound;

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using IndicesIteratorType   = typename IndicesType::iterator;
    using DistancesIteratorType = DistancesIterator;

    static constexpr void call_update(WithUnionFind<DistancesIterator, Bound>* unsorted_buffer,
                                      const IndexType&                         index_candidate,
                                      const DistanceType&                      distance_candidate) {
        unsorted_buffer->update_impl(index_candidate, distance_candidate);
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    static constexpr void call_partial_search(WithUnionFind<DistancesIterator, Bound>* unsorted_buffer,
                                              const OtherIndicesIterator&              indices_range_first,
                                              const OtherIndicesIterator&              indices_range_last,
                                              const OtherSamplesIterator&              samples_range_first,
                                              const OtherSamplesIterator&              samples_range_last,
                                              std::size_t                              n_features) {
        unsorted_buffer->partial_search_impl(/**/ indices_range_first,
                                             /**/ indices_range_last,
                                             /**/ samples_range_first,
                                             /**/ samples_range_last,
                                             /**/ n_features);
    }

    template <typename OptionalType>
    static constexpr void call_update(WithUnionFind<DistancesIterator, Bound>* unsorted_buffer,
                                      const IndexType&                         index_candidate,
                                      const DistanceType&                      distance_candidate,
                                      std::optional<OptionalType>&             candidate_representative) {
        unsorted_buffer->update_impl(index_candidate, distance_candidate, candidate_representative);
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator, typename OptionalType>
    static constexpr void call_partial_search(WithUnionFind<DistancesIterator, Bound>* unsorted_buffer,
                                              const OtherIndicesIterator&              indices_range_first,
                                              const OtherIndicesIterator&              indices_range_last,
                                              const OtherSamplesIterator&              samples_range_first,
                                              const OtherSamplesIterator&              samples_range_last,
                                              std::size_t                              n_features,
                                              std::optional<OptionalType>& candidates_component_membership) {
        unsorted_buffer->partial_search_impl(/**/ indices_range_first,
                                             /**/ indices_range_last,
                                             /**/ samples_range_first,
                                             /**/ samples_range_last,
                                             /**/ n_features,
                                             /**/ candidates_component_membership);
    }
};

template <typename Bound, typename UnionFindConstReference, typename Index>
WithUnionFind(Bound&&, UnionFindConstReference, const Index&) -> WithUnionFind<typename Bound::IteratorType, Bound>;

template <typename DistancesIterator, typename UnionFindConstReference, typename Index>
WithUnionFind(const DistancesIterator&, const DistancesIterator&, UnionFindConstReference, const Index&)
    -> WithUnionFind<DistancesIterator, datastruct::bounds::UnboundedBallView<DistancesIterator>>;

// ---

template <typename Bound, typename UnionFindConstReference, typename Index>
WithUnionFind(Bound&&, UnionFindConstReference, const Index&, const Index&)
    -> WithUnionFind<typename Bound::IteratorType, Bound>;

template <typename DistancesIterator, typename UnionFindConstReference, typename Index>
WithUnionFind(const DistancesIterator&, const DistancesIterator&, UnionFindConstReference, const Index&, const Index&)
    -> WithUnionFind<DistancesIterator, datastruct::bounds::UnboundedBallView<DistancesIterator>>;

}  // namespace ffcl::search::buffer