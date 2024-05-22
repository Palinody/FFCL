#pragma once

namespace ffcl::datastruct::bounds::segment {

template <class DerivedClass>
struct StaticSegment {
    template <typename DerivedType = DerivedClass>
    constexpr auto lower_bound() const {
        return static_cast<const DerivedType*>(this)->lower_bound_impl();
    }

    template <typename DerivedType = DerivedClass>
    void update_lower_bound(const typename DerivedType::ValueType& new_lower_bound) {
        static_cast<DerivedType*>(this)->update_lower_bound_impl(new_lower_bound);
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto upper_bound() const {
        return static_cast<const DerivedType*>(this)->upper_bound_impl();
    }

    template <typename DerivedType = DerivedClass>
    void update_upper_bound(const typename DerivedType::ValueType& new_upper_bound) {
        static_cast<DerivedType*>(this)->update_upper_bound_impl(new_upper_bound);
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
        return static_cast<const DerivedType*>(this)->contains_value_impl(value);
    }
};

}  // namespace ffcl::datastruct::bounds::segment