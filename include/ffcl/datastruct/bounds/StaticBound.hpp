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
};

}  // namespace ffcl::datastruct::bounds