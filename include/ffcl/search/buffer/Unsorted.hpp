#pragma once

#include "ffcl/search/buffer/StaticBuffer.hpp"

#include "ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include "ffcl/common/Utils.hpp"

#include <vector>

namespace ffcl::search::buffer {

template <typename DistancesIterator, typename Bound = datastruct::bounds::UnboundedBallView<DistancesIterator>>
class Unsorted : public StaticBuffer<Unsorted<DistancesIterator, Bound>> {
  public:
    using BoundType = Bound;

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using IndicesIteratorType   = typename IndicesType::iterator;
    using DistancesIteratorType = DistancesIterator;

    explicit Unsorted(BoundType&& bound, const IndexType& max_capacity = common::infinity<IndexType>())
      : StaticBuffer<Unsorted<DistancesIteratorType, BoundType>>(std::forward<BoundType>(bound), max_capacity) {}

    explicit Unsorted(const DistancesIteratorType& centroid_features_query_first,
                      const DistancesIteratorType& centroid_features_query_last,
                      const IndexType&             max_capacity = common::infinity<IndexType>())
      : Unsorted{BoundType{centroid_features_query_first, centroid_features_query_last}, max_capacity} {}

    void update_impl(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        this->update_static_buffers(index_candidate, distance_candidate);
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    void partial_search_impl(const OtherIndicesIterator& indices_range_first,
                             const OtherIndicesIterator& indices_range_last,
                             const OtherSamplesIterator& samples_range_first,
                             const OtherSamplesIterator& samples_range_last,
                             std::size_t                 n_features) {
        common::ignore_parameters(samples_range_last);

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
};

// Declare and define a static_base_traits specialization for Unsorted:
template <typename DistancesIterator, typename Bound>
struct static_base_traits<Unsorted<DistancesIterator, Bound>> {
    using BoundType = Bound;

    using IndexType    = std::size_t;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using IndicesType   = std::vector<IndexType>;
    using DistancesType = std::vector<DistanceType>;

    using IndicesIteratorType   = typename IndicesType::iterator;
    using DistancesIteratorType = DistancesIterator;

    static constexpr void call_update(Unsorted<DistancesIterator, Bound>* unsorted_buffer,
                                      const IndexType&                    index_candidate,
                                      const DistanceType&                 distance_candidate) {
        unsorted_buffer->update_impl(index_candidate, distance_candidate);
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    static constexpr void call_partial_search(Unsorted<DistancesIterator, Bound>* unsorted_buffer,
                                              const OtherIndicesIterator&         indices_range_first,
                                              const OtherIndicesIterator&         indices_range_last,
                                              const OtherSamplesIterator&         samples_range_first,
                                              const OtherSamplesIterator&         samples_range_last,
                                              std::size_t                         n_features) {
        unsorted_buffer->partial_search_impl(/**/ indices_range_first,
                                             /**/ indices_range_last,
                                             /**/ samples_range_first,
                                             /**/ samples_range_last,
                                             /**/ n_features);
    }
};

template <typename Bound>
Unsorted(Bound &&) -> Unsorted<typename Bound::IteratorType, Bound>;

template <typename DistancesIteratorType>
Unsorted(const DistancesIteratorType&, const DistancesIteratorType&)
    -> Unsorted<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>>;

// ---

template <typename Bound, typename Index>
Unsorted(Bound&&, const Index&) -> Unsorted<typename Bound::IteratorType, Bound>;

template <typename DistancesIteratorType, typename Index>
Unsorted(const DistancesIteratorType&, const DistancesIteratorType&, const Index&)
    -> Unsorted<DistancesIteratorType, datastruct::bounds::UnboundedBallView<DistancesIteratorType>>;

}  // namespace ffcl::search::buffer