#pragma once

#include <optional>
#include <vector>

namespace ffcl::datastruct::bounds {

template <class DerivedClass>
struct StaticBound {
    template <typename DerivedType = DerivedClass>
    auto n_features() const {
        return static_cast<const DerivedType*>(this)->n_features_impl();
    }

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    auto is_in_bounds(const FeaturesIterator& features_range_first, const FeaturesIterator& features_range_last) const {
        return static_cast<const DerivedType*>(this)->is_in_bounds_impl(features_range_first, features_range_last);
    }

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    auto distance(const FeaturesIterator& features_range_first, const FeaturesIterator& features_range_last) const {
        return static_cast<const DerivedType*>(this)->distance_impl(features_range_first, features_range_last);
    }

    template <typename FeaturesIterator, typename DerivedType = DerivedClass>
    auto compute_distance_within_bounds(const FeaturesIterator& features_range_first,
                                        const FeaturesIterator& features_range_last) const {
        return static_cast<const DerivedType*>(this)->compute_distance_within_bounds_impl(features_range_first,
                                                                                          features_range_last);
    }

    template <typename DerivedType = DerivedClass>
    auto length_from_centroid() const {
        return static_cast<const DerivedType*>(this)->length_from_centroid_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto length_from_centroid(std::size_t feature_index) const {
        return static_cast<const DerivedType*>(this)->length_from_centroid_impl(feature_index);
    }

    template <typename DerivedType = DerivedClass>
    auto centroid() const {
        return static_cast<const DerivedType*>(this)->centroid_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto make_centroid() const {
        return static_cast<const DerivedType*>(this)->make_centroid_impl();
    }
};

}  // namespace ffcl::datastruct::bounds