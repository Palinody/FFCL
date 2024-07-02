#pragma once

#include <optional>
#include <vector>

#include "ffcl/common/math/heuristics/Distances.hpp"

namespace ffcl::datastruct::bounds {

template <class DerivedClass>
struct StaticBoundWithCentroid {
    template <typename DerivedType = DerivedClass>
    constexpr auto diameter() const;

    template <typename DerivedType = DerivedClass>
    auto n_features() const;

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    constexpr bool is_in_bounds(const FeaturesIterator& features_range_first,
                                const FeaturesIterator& features_range_last) const;

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    constexpr auto distance_to_centroid(const FeaturesIterator& features_range_first,
                                        const FeaturesIterator& features_range_last) const;

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    constexpr auto compute_distance_to_centroid_if_within_bounds(const FeaturesIterator& features_range_first,
                                                                 const FeaturesIterator& features_range_last) const;

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_to_furthest_bound_distance() const;

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_to_bound_distance(std::size_t feature_index) const;

    template <typename DerivedType = DerivedClass>
    auto centroid() const;

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_begin() const;

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_end() const;
};

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::diameter() const {
    return static_cast<const DerivedType*>(this)->diameter_impl();
}

template <class DerivedClass>
template <typename DerivedType>
auto StaticBoundWithCentroid<DerivedClass>::n_features() const {
    return static_cast<const DerivedType*>(this)->n_features_impl();
}

template <class DerivedClass>
template <typename FeaturesIterator, typename DerivedType>
constexpr bool StaticBoundWithCentroid<DerivedClass>::is_in_bounds(const FeaturesIterator& features_range_first,
                                                                   const FeaturesIterator& features_range_last) const {
    return static_cast<const DerivedType*>(this)->is_in_bounds_impl(features_range_first, features_range_last);
}

template <class DerivedClass>
template <typename FeaturesIterator, typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::distance_to_centroid(
    const FeaturesIterator& features_range_first,
    const FeaturesIterator& features_range_last) const {
    return static_cast<const DerivedType*>(this)->distance_to_centroid_impl(features_range_first, features_range_last);
}

template <class DerivedClass>
template <typename FeaturesIterator, typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::compute_distance_to_centroid_if_within_bounds(
    const FeaturesIterator& features_range_first,
    const FeaturesIterator& features_range_last) const {
    return static_cast<const DerivedType*>(this)->compute_distance_to_centroid_if_within_bounds_impl(
        features_range_first, features_range_last);
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::centroid_to_furthest_bound_distance() const {
    return static_cast<const DerivedType*>(this)->centroid_to_furthest_bound_distance_impl();
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::centroid_to_bound_distance(std::size_t feature_index) const {
    return static_cast<const DerivedType*>(this)->centroid_to_bound_distance_impl(feature_index);
}

template <class DerivedClass>
template <typename DerivedType>
auto StaticBoundWithCentroid<DerivedClass>::centroid() const {
    return static_cast<const DerivedType*>(this)->centroid_impl();
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::centroid_begin() const {
    return static_cast<const DerivedType*>(this)->centroid_begin_impl();
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticBoundWithCentroid<DerivedClass>::centroid_end() const {
    return static_cast<const DerivedType*>(this)->centroid_end_impl();
}

}  // namespace ffcl::datastruct::bounds