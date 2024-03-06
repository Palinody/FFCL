#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/StaticBound.hpp"

#include "ffcl/datastruct/bounds/UnboundedBall.hpp"  // default bound

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::search::buffer {

// https://stackoverflow.com/questions/6006614/c-static-polymorphism-crtp-and-using-typedefs-from-derived-classes
template <typename DerivedBuffer>
struct static_base_traits;

template <class DerivedBuffer>
class StaticBuffer {
  public:
    using BoundType = typename static_base_traits<DerivedBuffer>::BoundType;

    static_assert(common::is_crtp_of<BoundType, datastruct::bounds::StaticBound>::value,
                  "BoundType does not inherit from datastruct::bounds::StaticBound<Derived>");

    using IndexType    = typename static_base_traits<DerivedBuffer>::IndexType;
    using DistanceType = typename static_base_traits<DerivedBuffer>::DistanceType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using IndicesType   = typename static_base_traits<DerivedBuffer>::IndicesType;
    using DistancesType = typename static_base_traits<DerivedBuffer>::DistancesType;

    using IndicesIteratorType   = typename static_base_traits<DerivedBuffer>::IndicesIteratorType;
    using DistancesIteratorType = typename static_base_traits<DerivedBuffer>::DistancesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<DistancesIteratorType>::value, "DistancesIteratorType is not an iterator");

    explicit StaticBuffer(BoundType&& bound, const IndexType& max_capacity = common::infinity<IndexType>())
      : bound_{std::forward<BoundType>(bound)}
      , indices_{}
      , distances_{}
      , buffer_index_of_furthest_index_{0}
      , furthest_distance_{0}
      , max_capacity_{max_capacity} {}

    constexpr auto centroid_begin() const {
        return bound_.centroid_begin();
    }

    constexpr auto centroid_end() const {
        return bound_.centroid_end();
    }

    auto indices() const {
        return indices_;
    }

    auto distances() const {
        return distances_;
    }

    const auto& const_reference_indices() const& {
        return indices_;
    }

    const auto& const_reference_distances() const& {
        return distances_;
    }

    auto&& move_indices() && {
        return std::move(indices_);
    }

    auto&& move_distances() && {
        return std::move(distances_);
    }

    std::size_t size() const {
        return indices_.size();
    }

    std::size_t remaining_capacity() const {
        return max_capacity_ - size();
    }

    bool empty() const {
        return indices_.empty();
    }

    IndexType furthest_index() const {
        return indices_[buffer_index_of_furthest_index_];
    }

    DistanceType furthest_distance() const {
        return furthest_distance_;
    }

    DistanceType furthest_distance(const IndexType& feature_index) const {
        common::ignore_parameters(feature_index);
        return furthest_distance();
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        static_base_traits<DerivedBuffer>::call_update(static_cast<DerivedBuffer*>(this),
                                                       /**/ index_candidate,
                                                       /**/ distance_candidate);
    }

    template <typename OtherIndicesIterator, typename OtherSamplesIterator>
    void partial_search(const OtherIndicesIterator& indices_range_first,
                        const OtherIndicesIterator& indices_range_last,
                        const OtherSamplesIterator& samples_range_first,
                        const OtherSamplesIterator& samples_range_last,
                        std::size_t                 n_features) {
        static_base_traits<DerivedBuffer>::call_partial_search(static_cast<DerivedBuffer*>(this),
                                                               /**/ indices_range_first,
                                                               /**/ indices_range_last,
                                                               /**/ samples_range_first,
                                                               /**/ samples_range_last,
                                                               /**/ n_features);
    }

  protected:
    BoundType bound_;

    IndicesType   indices_;
    DistancesType distances_;

    IndexType    buffer_index_of_furthest_index_;
    DistanceType furthest_distance_;

    IndexType max_capacity_;
};

}  // namespace ffcl::search::buffer
