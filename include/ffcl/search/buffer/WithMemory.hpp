#pragma once

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/datastruct/bounds/StaticCentroidBasedBound.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include "ffcl/common/Utils.hpp"

#include <unordered_set>
#include <vector>

namespace ffcl::search::buffer {

template <typename DistancesIterator,
          typename Bound          = datastruct::bounds::UnboundedBallView<DistancesIterator>,
          typename VisitedIndices = std::unordered_set<std::size_t>>
class WithMemory : public StaticBuffer<WithMemory<DistancesIterator, Bound>> {
  public:
    using BoundType = Bound;

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using IndicesIteratorType   = typename IndicesType::iterator;
    using DistancesIteratorType = DistancesIterator;

    using VisitedIndicesIteratorType = typename VisitedIndices::iterator;

    explicit WithMemory(BoundType&& bound, const IndexType& max_capacity = common::infinity<IndexType>())
      : StaticBuffer<WithMemory<DistancesIteratorType, BoundType>>(std::forward<BoundType>(bound), max_capacity)
      , visited_indices_{}
      , visited_indices_const_reference_{visited_indices_} {}

    WithMemory(BoundType&&           bound,
               const VisitedIndices& visited_indices_reference,
               const IndexType&      max_capacity = common::infinity<IndexType>())
      : StaticBuffer<WithMemory<DistancesIteratorType, BoundType>>(std::forward<BoundType>(bound), max_capacity)
      , visited_indices_{}
      , visited_indices_const_reference_{visited_indices_reference} {}

    WithMemory(BoundType&&      bound,
               VisitedIndices&& visited_indices,
               const IndexType& max_capacity = common::infinity<IndexType>())
      : StaticBuffer<WithMemory<DistancesIteratorType, BoundType>>(std::forward<BoundType>(bound), max_capacity)
      , visited_indices_{std::move(visited_indices)}
      , visited_indices_const_reference_{visited_indices_} {}

    WithMemory(BoundType&&                       bound,
               const VisitedIndicesIteratorType& visited_indices_first,
               const VisitedIndicesIteratorType& visited_indices_last,
               const IndexType&                  max_capacity = common::infinity<IndexType>())
      : WithMemory(std::move(bound), VisitedIndices(visited_indices_first, visited_indices_last), max_capacity) {}

    WithMemory(const DistancesIteratorType& centroid_features_query_first,
               const DistancesIteratorType& centroid_features_query_last,
               const IndexType&             max_capacity = common::infinity<IndexType>())
      : WithMemory(BoundType{centroid_features_query_first, centroid_features_query_last}, max_capacity) {}

    WithMemory(const DistancesIteratorType& centroid_features_query_first,
               const DistancesIteratorType& centroid_features_query_last,
               const VisitedIndices&        visited_indices_reference,
               const IndexType&             max_capacity = common::infinity<IndexType>())
      : WithMemory(BoundType{centroid_features_query_first, centroid_features_query_last},
                   visited_indices_reference,
                   max_capacity) {}

    WithMemory(const DistancesIteratorType& centroid_features_query_first,
               const DistancesIteratorType& centroid_features_query_last,
               VisitedIndices&&             visited_indices,
               const IndexType&             max_capacity = common::infinity<IndexType>())
      : WithMemory(BoundType{centroid_features_query_first, centroid_features_query_last},
                   std::move(visited_indices),
                   max_capacity) {}

    WithMemory(const DistancesIteratorType&      centroid_features_query_first,
               const DistancesIteratorType&      centroid_features_query_last,
               const VisitedIndicesIteratorType& visited_indices_first,
               const VisitedIndicesIteratorType& visited_indices_last,
               const IndexType&                  max_capacity = common::infinity<IndexType>())
      : WithMemory(BoundType{centroid_features_query_first, centroid_features_query_last},
                   visited_indices_first,
                   visited_indices_last,
                   max_capacity) {}

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        // consider an update only if the candidate hasnt been visited
        const bool is_candidate_valid =
            visited_indices_const_reference_.find(index_candidate) == visited_indices_const_reference_.end();

        if (is_candidate_valid) {
            this->update_static_buffers(index_candidate, distance_candidate);
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
            const auto optional_candidate_distance = this->bound_.compute_distance_to_centroid_if_within_bounds(
                samples_range_first + *reference_index_it * n_features,
                samples_range_first + *reference_index_it * n_features + n_features);

            if (optional_candidate_distance) {
                update_impl(*reference_index_it, *optional_candidate_distance);
            }
        }
    }

  private:
    VisitedIndices        visited_indices_;
    const VisitedIndices& visited_indices_const_reference_;
};

// Declare and define a static_base_traits specialization for WithMemory:
template <typename DistancesIterator, typename Bound, typename VisitedIndices>
struct static_base_traits<WithMemory<DistancesIterator, Bound, VisitedIndices>> {
    using BoundType = Bound;

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using IndicesIteratorType   = typename IndicesType::iterator;
    using DistancesIteratorType = DistancesIterator;

    static constexpr void call_update(WithMemory<DistancesIterator, Bound, VisitedIndices>* unsorted_buffer,
                                      const IndexType&                                      index_candidate,
                                      const DistanceType&                                   distance_candidate) {
        unsorted_buffer->update_impl(index_candidate, distance_candidate);
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    static constexpr void call_partial_search(WithMemory<DistancesIterator, Bound, VisitedIndices>* unsorted_buffer,
                                              const OtherIndicesIterator&                           indices_range_first,
                                              const OtherIndicesIterator&                           indices_range_last,
                                              const OtherSamplesIterator&                           samples_range_first,
                                              const OtherSamplesIterator&                           samples_range_last,
                                              std::size_t                                           n_features) {
        unsorted_buffer->partial_search_impl(/**/ indices_range_first,
                                             /**/ indices_range_last,
                                             /**/ samples_range_first,
                                             /**/ samples_range_last,
                                             /**/ n_features);
    }
};

template <typename Bound>
WithMemory(Bound &&) -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<std::size_t>>;

template <typename Bound, typename VisitedIndices>
WithMemory(Bound&&, const VisitedIndices&) -> WithMemory<typename Bound::IteratorType, Bound, VisitedIndices>;

template <typename Bound, typename VisitedIndicesIterator>
WithMemory(Bound&&, const VisitedIndicesIterator&, const VisitedIndicesIterator&)
    -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<std::size_t>>;

template <typename DistancesIterator>
WithMemory(const DistancesIterator&, const DistancesIterator&)
    -> WithMemory<DistancesIterator,
                  datastruct::bounds::UnboundedBallView<DistancesIterator>,
                  std::unordered_set<std::size_t>>;

template <typename DistancesIterator, typename VisitedIndices>
WithMemory(const DistancesIterator&, const DistancesIterator&, const VisitedIndices&)
    -> WithMemory<DistancesIterator, datastruct::bounds::UnboundedBallView<DistancesIterator>, VisitedIndices>;

template <typename DistancesIterator, typename VisitedIndices>
WithMemory(const DistancesIterator&, const DistancesIterator&, VisitedIndices &&)
    -> WithMemory<DistancesIterator, datastruct::bounds::UnboundedBallView<DistancesIterator>, VisitedIndices>;

template <typename DistancesIterator, typename VisitedIndicesIterator>
WithMemory(const DistancesIterator&,
           const DistancesIterator&,
           const VisitedIndicesIterator&,
           const VisitedIndicesIterator&) -> WithMemory<DistancesIterator,
                                                        datastruct::bounds::UnboundedBallView<DistancesIterator>,
                                                        std::unordered_set<std::size_t>>;

// ---

template <typename Bound, typename Index>
WithMemory(Bound&&, const Index&) -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<Index>>;

template <typename Bound, typename VisitedIndices, typename Index>
WithMemory(Bound&&, const VisitedIndices&, const Index&)
    -> WithMemory<typename Bound::IteratorType, Bound, VisitedIndices>;

template <typename Bound, typename VisitedIndicesIterator, typename Index>
WithMemory(Bound&&, const VisitedIndicesIterator&, const VisitedIndicesIterator&, const Index&)
    -> WithMemory<typename Bound::IteratorType, Bound, std::unordered_set<Index>>;

template <typename DistancesIterator, typename Index>
WithMemory(const DistancesIterator&, const DistancesIterator&, const Index&)
    -> WithMemory<DistancesIterator,
                  datastruct::bounds::UnboundedBallView<DistancesIterator>,
                  std::unordered_set<Index>>;

template <typename DistancesIterator, typename VisitedIndices, typename Index>
WithMemory(const DistancesIterator&, const DistancesIterator&, const VisitedIndices&, const Index&)
    -> WithMemory<DistancesIterator, datastruct::bounds::UnboundedBallView<DistancesIterator>, VisitedIndices>;

template <typename DistancesIterator, typename VisitedIndices, typename Index>
WithMemory(const DistancesIterator&, const DistancesIterator&, VisitedIndices&&, const Index&)
    -> WithMemory<DistancesIterator, datastruct::bounds::UnboundedBallView<DistancesIterator>, VisitedIndices>;

template <typename DistancesIterator, typename VisitedIndicesIterator, typename Index>
WithMemory(const DistancesIterator&,
           const DistancesIterator&,
           const VisitedIndicesIterator&,
           const VisitedIndicesIterator&,
           const Index&) -> WithMemory<DistancesIterator,
                                       datastruct::bounds::UnboundedBallView<DistancesIterator>,
                                       std::unordered_set<Index>>;

}  // namespace ffcl::search::buffer
