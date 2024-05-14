#pragma once

namespace ffcl::datastruct::bounds::segment {

template <class DerivedClass>
struct StaticSegment {
    template <typename DerivedType = DerivedClass>
    constexpr auto first() const {
        return static_cast<const DerivedType*>(this)->read_only_first_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto& first() {
        return static_cast<DerivedType*>(this)->read_write_first_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto second() const {
        return static_cast<const DerivedType*>(this)->read_only_second_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto& second() {
        return static_cast<DerivedType*>(this)->read_write_second_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid() const {
        return static_cast<const DerivedType*>(this)->centroid_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_to_bound_length() const {
        return static_cast<const DerivedType*>(this)->centroid_to_bound_length_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr bool contains_value(const typename DerivedType::ValueType& value) const {
        return static_cast<const DerivedType*>(this)->contains_value_impl();
    }
};

}  // namespace ffcl::datastruct::bounds::segment