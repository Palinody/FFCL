#pragma once

namespace ffcl::datastruct::bounds::segment {

template <class DerivedClass>
struct StaticSegment {
    template <typename DerivedType = DerivedClass>
    constexpr auto lower_bound() const;

    template <typename DerivedType = DerivedClass>
    void update_lower_bound(const typename DerivedType::ValueType& new_lower_bound);

    template <typename DerivedType = DerivedClass>
    constexpr auto upper_bound() const;

    template <typename DerivedType = DerivedClass>
    void update_upper_bound(const typename DerivedType::ValueType& new_upper_bound);

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid() const;

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_to_bound_distance() const;

    template <typename DerivedType = DerivedClass>
    constexpr bool contains_value(const typename DerivedType::ValueType& value) const;
};

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticSegment<DerivedClass>::lower_bound() const {
    return static_cast<const DerivedType*>(this)->lower_bound_impl();
}

template <class DerivedClass>
template <typename DerivedType>
void StaticSegment<DerivedClass>::update_lower_bound(const typename DerivedType::ValueType& new_lower_bound) {
    static_cast<DerivedType*>(this)->update_lower_bound_impl(new_lower_bound);
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticSegment<DerivedClass>::upper_bound() const {
    return static_cast<const DerivedType*>(this)->upper_bound_impl();
}

template <class DerivedClass>
template <typename DerivedType>
void StaticSegment<DerivedClass>::update_upper_bound(const typename DerivedType::ValueType& new_upper_bound) {
    static_cast<DerivedType*>(this)->update_upper_bound_impl(new_upper_bound);
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticSegment<DerivedClass>::centroid() const {
    return static_cast<const DerivedType*>(this)->centroid_impl();
}

template <class DerivedClass>
template <typename DerivedType>
constexpr auto StaticSegment<DerivedClass>::centroid_to_bound_distance() const {
    return static_cast<const DerivedType*>(this)->centroid_to_bound_distance_impl();
}

template <class DerivedClass>
template <typename DerivedType>
constexpr bool StaticSegment<DerivedClass>::contains_value(const typename DerivedType::ValueType& value) const {
    return static_cast<const DerivedType*>(this)->contains_value_impl(value);
}

}  // namespace ffcl::datastruct::bounds::segment