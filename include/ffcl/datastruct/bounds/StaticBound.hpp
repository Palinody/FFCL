#pragma once

#include <optional>
#include <vector>

#include "ffcl/common/math/heuristics/Distances.hpp"

namespace ffcl::datastruct::bounds {

template <typename T>
struct always_false : std::false_type {};

template <class DerivedClass>
struct StaticBound {
    template <typename DerivedType = DerivedClass>
    auto n_features() const {
        return static_cast<const DerivedType*>(this)->n_features_impl();
    }

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    constexpr bool is_in_bounds(const FeaturesIterator& features_range_first,
                                const FeaturesIterator& features_range_last) const {
        return static_cast<const DerivedType*>(this)->is_in_bounds_impl(features_range_first, features_range_last);
    }

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    constexpr auto distance(const FeaturesIterator& features_range_first,
                            const FeaturesIterator& features_range_last) const {
        return static_cast<const DerivedType*>(this)->distance_impl(features_range_first, features_range_last);
    }

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    constexpr auto compute_distance_if_within_bounds(const FeaturesIterator& features_range_first,
                                                     const FeaturesIterator& features_range_last) const {
        return static_cast<const DerivedType*>(this)->compute_distance_if_within_bounds_impl(features_range_first,
                                                                                             features_range_last);
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto length_from_centroid() const {
        return static_cast<const DerivedType*>(this)->length_from_centroid_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto length_from_centroid(std::size_t feature_index) const {
        return static_cast<const DerivedType*>(this)->length_from_centroid_impl(feature_index);
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto& centroid_reference() const {
        return static_cast<const DerivedType*>(this)->centroid_reference_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto make_centroid() const {
        return static_cast<const DerivedType*>(this)->make_centroid_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_begin() const {
        return static_cast<const DerivedType*>(this)->centroid_begin_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_end() const {
        return static_cast<const DerivedType*>(this)->centroid_end_impl();
    }
};

}  // namespace ffcl::datastruct::bounds