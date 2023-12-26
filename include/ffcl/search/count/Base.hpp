#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include <cstddef>
#include <tuple>
#include <vector>

namespace ffcl::search::count {

template <typename IndicesIterator, typename DistancesIterator>
class Base {
  public:
    using IndexType    = typename std::iterator_traits<IndicesIterator>::value_type;
    using DistanceType = typename std::iterator_traits<DistancesIterator>::value_type;

    using SamplesIterator = typename std::vector<DistanceType>::iterator;

    virtual ~Base() {}

    virtual DistanceType upper_bound() const = 0;

    virtual DistanceType upper_bound(const IndexType& feature_index) const = 0;

    virtual std::size_t n_free_slots() const = 0;

    virtual IndexType count() const = 0;

    virtual void update(const IndexType& index_candidate, const DistanceType& distance_candidate) = 0;

    virtual void search(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        std::size_t            sample_index_query) = 0;

    virtual void search(const IndicesIterator& indices_range_first,
                        const IndicesIterator& indices_range_last,
                        const SamplesIterator& samples_range_first,
                        const SamplesIterator& samples_range_last,
                        std::size_t            n_features,
                        const SamplesIterator& feature_query_range_first,
                        const SamplesIterator& feature_query_range_last) = 0;

    virtual void print() const = 0;
};

template <class DerivedClass>
struct StaticCounter {
    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_begin() const {
        return static_cast<const DerivedType*>(this)->centroid_begin_impl();
    }

    template <typename DerivedType = DerivedClass>
    constexpr auto centroid_end() const {
        return static_cast<const DerivedType*>(this)->centroid_end_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto count() const {
        return static_cast<const DerivedType*>(this)->n_free_slots_impl();
    }

    template <typename DerivedType = DerivedClass>
    auto n_free_slots() const {
        return static_cast<const DerivedType*>(this)->n_free_slots_impl();
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

}  // namespace ffcl::search::count