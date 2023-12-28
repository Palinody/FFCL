#pragma once

#include "ffcl/datastruct/kdtree/KDTreeAlgorithms.hpp"

namespace ffcl::datastruct::kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
struct SplittingRulePolicy {
    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    SplittingRulePolicy() = default;

    virtual ~SplittingRulePolicy() = default;

    inline virtual std::tuple<std::size_t,
                              bbox::IteratorPairType<IndicesIterator>,
                              bbox::IteratorPairType<IndicesIterator>,
                              bbox::IteratorPairType<IndicesIterator>>
    operator()(const IndicesIterator& indices_range_first,
               const IndicesIterator& indices_range_last,
               const SamplesIterator& samples_range_first,
               const SamplesIterator& samples_range_last,
               std::size_t            n_features,
               std::size_t            feature_index) const = 0;
};

template <typename IndicesIterator, typename SamplesIterator>
struct QuickselectMedianRange : public SplittingRulePolicy<IndicesIterator, SamplesIterator> {
    static_assert(common::is_iterator<IndicesIterator>::value, "IndicesIterator is not an iterator");
    static_assert(common::is_iterator<SamplesIterator>::value, "SamplesIterator is not an iterator");

    QuickselectMedianRange() = default;

    inline std::tuple<std::size_t,
                      bbox::IteratorPairType<IndicesIterator>,
                      bbox::IteratorPairType<IndicesIterator>,
                      bbox::IteratorPairType<IndicesIterator>>
    operator()(const IndicesIterator& indices_range_first,
               const IndicesIterator& indices_range_last,
               const SamplesIterator& samples_range_first,
               const SamplesIterator& samples_range_last,
               std::size_t            n_features,
               std::size_t            feature_index) const;
};

}  // namespace ffcl::datastruct::kdtree::policy

namespace ffcl::datastruct::kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
std::tuple<std::size_t,
           bbox::IteratorPairType<IndicesIterator>,
           bbox::IteratorPairType<IndicesIterator>,
           bbox::IteratorPairType<IndicesIterator>>
QuickselectMedianRange<IndicesIterator, SamplesIterator>::operator()(const IndicesIterator& indices_range_first,
                                                                     const IndicesIterator& indices_range_last,
                                                                     const SamplesIterator& samples_range_first,
                                                                     const SamplesIterator& samples_range_last,
                                                                     std::size_t            n_features,
                                                                     std::size_t            feature_index) const {
    return kdtree::algorithms::quickselect_median(
        /**/ indices_range_first,
        /**/ indices_range_last,
        /**/ samples_range_first,
        /**/ samples_range_last,
        /**/ n_features,
        /**/ feature_index);
}

}  // namespace ffcl::datastruct::kdtree::policy
