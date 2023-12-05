#pragma once

namespace ffcl::datastruct::bounds::segment_representation {

template <class DerivedClass>
struct StaticSegmentRepresentation {
    template <typename DerivedType = DerivedClass>
    constexpr auto length_from_centroid() const {
        return static_cast<const DerivedType*>(this)->length_from_centroid_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid() const {
        return static_cast<const DerivedType*>(this)->centroid_impl();
    }
};

}  // namespace ffcl::datastruct::bounds::segment_representation