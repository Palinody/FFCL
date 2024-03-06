#pragma once

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/datastruct/bounds/StaticBound.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"
#include "ffcl/datastruct/UnionFind.hpp"

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
                  UnionFindConstReference union_find_const_reference,
                  const IndexType&        query_representative,
                  const IndexType&        max_capacity = common::infinity<IndexType>())
      : StaticBuffer<WithUnionFind<DistancesIterator, Bound>>(std::forward<BoundType>(bound), max_capacity)
      , union_find_const_reference_{union_find_const_reference}
      , query_representative_{query_representative} {}

    WithUnionFind(const DistancesIterator& centroid_features_query_first,
                  const DistancesIterator& centroid_features_query_last,
                  UnionFindConstReference  union_find_const_reference,
                  const IndexType&         query_representative,
                  const IndexType&         max_capacity = common::infinity<IndexType>())
      : WithUnionFind(BoundType(centroid_features_query_first, centroid_features_query_last),
                      union_find_const_reference,
                      query_representative,
                      max_capacity) {}

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // consider an update only if the candidate is not in the same component as the representative of the component
        const bool is_candidate_valid = union_find_const_reference_.find(query_representative_) !=
                                        union_find_const_reference_.find(index_candidate);

        if (is_candidate_valid) {
            // always populate if the max capacity isnt reached
            if (this->remaining_capacity()) {
                this->indices_.emplace_back(index_candidate);
                this->distances_.emplace_back(distance_candidate);
                if (distance_candidate > this->furthest_distance()) {
                    // update the new index position of the furthest in the buffer
                    this->buffer_index_of_furthest_index_ = this->indices_.size() - 1;
                    this->furthest_distance_              = distance_candidate;
                }
            }
            // populate if the max capacity is reached and the candidate has a closer distance
            else if (distance_candidate < this->furthest_distance()) {
                // replace the previous greatest distance now that the vectors overflow the max capacity
                this->indices_[this->buffer_index_of_furthest_index_]   = index_candidate;
                this->distances_[this->buffer_index_of_furthest_index_] = distance_candidate;
                // find the new furthest neighbor and update the cache accordingly
                std::tie(this->buffer_index_of_furthest_index_, this->furthest_distance_) =
                    common::math::statistics::get_max_index_value_pair(this->distances_.begin(),
                                                                       this->distances_.end());
            }
        }
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    void partial_search_impl(const OtherIndicesIterator& indices_range_first,
                             const OtherIndicesIterator& indices_range_last,
                             const OtherSamplesIterator& samples_range_first,
                             const OtherSamplesIterator& samples_range_last,
                             std::size_t                 n_features) {
        ffcl::common::ignore_parameters(samples_range_last);

        // reference_index_it is an iterator that iterates over all the indices from the indices_range_first to
        // indices_range_last range
        for (auto reference_index_it = indices_range_first; reference_index_it != indices_range_last;
             ++reference_index_it) {
            const auto optional_candidate_distance = this->bound_.compute_distance_if_within_bounds(
                samples_range_first + *reference_index_it * n_features,
                samples_range_first + *reference_index_it * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(*reference_index_it, *optional_candidate_distance);
            }
        }
    }

  private:
    UnionFindConstReference union_find_const_reference_;
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