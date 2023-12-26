#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::search::buffer {

template <class DerivedClass>
struct StaticBase {
    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_begin() const {
        return static_cast<const DerivedType*>(this)->centroid_begin_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_end() const {
        return static_cast<const DerivedType*>(this)->centroid_end_impl();
    }

    //  Templating 'DerivedClass' ensures that IndicesType is treated as a dependent type, and its lookup is deferred
    //  until the template is instantiated, at which point DerivedClass is a complete type.
    template <typename DerivedType = DerivedClass>
    auto indices() const {
        // Use 'DerivedType' to make sure that 'DerivedClass' is treated as a dependant type
        return static_cast<const DerivedType*>(this)->indices_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto distances() const {
        return static_cast<const DerivedType*>(this)->distances_impl();
    }

    template <typename DerivedType = DerivedClass>
    const auto& indices() const& {
        return static_cast<const DerivedType&>(*this).const_reference_indices_impl();
    }

    template <typename DerivedType = DerivedClass>
    const auto& distances() const& {
        return static_cast<const DerivedType&>(*this).const_reference_distances_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto&& indices() && {
        // using ForwardedType = decltype(static_cast<DerivedType&&>(*this));
        return std::move(static_cast<DerivedType&&>(*this)).move_indices_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto&& distances() && {
        // using ForwardedType = decltype(static_cast<DerivedType&&>(*this));
        return std::move(static_cast<DerivedType&&>(*this)).move_distances_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto size() const {
        return static_cast<const DerivedType*>(this)->size_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto n_free_slots() const {
        return static_cast<const DerivedType*>(this)->n_free_slots_impl();
    }

    template <typename DerivedType = DerivedClass>
    bool empty() const {
        return static_cast<const DerivedType*>(this)->empty_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto upper_bound_index() const {
        return static_cast<const DerivedType*>(this)->upper_bound_index_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto upper_bound() const {
        return static_cast<const DerivedType*>(this)->upper_bound_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto upper_bound(const typename DerivedType::IndexType& feature_index) const {
        return static_cast<const DerivedType*>(this)->upper_bound_impl(feature_index);
    }

    template <typename DerivedType = DerivedClass>
    void update(const typename DerivedType::IndexType&    index_candidate,
                const typename DerivedType::DistanceType& distance_candidate) {
        static_cast<DerivedType*>(this)->update_impl(index_candidate, distance_candidate);
    }

    template <typename IndicesIterator, typename SamplesIterator, typename DerivedType = DerivedClass>
    void partial_search(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features) {
        static_cast<DerivedType*>(this)->partial_search_impl(
            /**/ indices_range_first,
            /**/ indices_range_last,
            /**/ samples_range_first,
            /**/ samples_range_last,
            /**/ n_features);
    }
};

}  // namespace ffcl::search::buffer
