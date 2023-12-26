#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::search::count {

template <class DerivedCounter>
struct StaticCBase {
    template <typename DerivedType = DerivedCounter>
    constexpr auto centroid_begin() const {
        return static_cast<const DerivedType*>(this)->centroid_begin_impl();
    }

    template <typename DerivedType = DerivedCounter>
    constexpr auto centroid_end() const {
        return static_cast<const DerivedType*>(this)->centroid_end_impl();
    }

    template <typename DerivedType = DerivedCounter>
    auto count() const {
        return static_cast<const DerivedType*>(this)->n_free_slots_impl();
    }

    template <typename DerivedType = DerivedCounter>
    auto n_free_slots() const {
        return static_cast<const DerivedType*>(this)->n_free_slots_impl();
    }

    template <typename DerivedType = DerivedCounter>
    auto upper_bound() const {
        return static_cast<const DerivedType*>(this)->upper_bound_impl();
    }

    template <typename DerivedType = DerivedCounter>
    auto upper_bound(const typename DerivedType::IndexType& feature_index) const {
        return static_cast<const DerivedType*>(this)->upper_bound_impl(feature_index);
    }

    template <typename DerivedType = DerivedCounter>
    void update(const typename DerivedType::IndexType&    index_candidate,
                const typename DerivedType::DistanceType& distance_candidate) {
        static_cast<DerivedType*>(this)->update_impl(index_candidate, distance_candidate);
    }

    template <typename IndicesIterator, typename SamplesIterator, typename DerivedType = DerivedCounter>
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

}  // namespace ffcl::search::count