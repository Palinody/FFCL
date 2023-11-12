#pragma once

#include "ffcl/datastruct/kdtree/KDTreeAlgorithms.hpp"

namespace ffcl::datastruct::kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
struct SplittingRulePolicy {
    SplittingRulePolicy() = default;

    inline virtual std::tuple<std::size_t,
                              bbox::IteratorPairType<IndicesIterator>,
                              bbox::IteratorPairType<IndicesIterator>,
                              bbox::IteratorPairType<IndicesIterator>>
    operator()(IndicesIterator indices_range_first,
               IndicesIterator indices_range_last,
               SamplesIterator samples_range_first,
               SamplesIterator samples_range_last,
               std::size_t     n_features,
               std::size_t     feature_index) const = 0;
};

template <typename IndicesIterator, typename SamplesIterator>
struct QuickselectMedianRange : public SplittingRulePolicy<IndicesIterator, SamplesIterator> {
    inline std::tuple<std::size_t,
                      bbox::IteratorPairType<IndicesIterator>,
                      bbox::IteratorPairType<IndicesIterator>,
                      bbox::IteratorPairType<IndicesIterator>>
    operator()(IndicesIterator indices_range_first,
               IndicesIterator indices_range_last,
               SamplesIterator samples_range_first,
               SamplesIterator samples_range_last,
               std::size_t     n_features,
               std::size_t     feature_index) const;
};

}  // namespace ffcl::datastruct::kdtree::policy

namespace ffcl::datastruct::kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
std::tuple<std::size_t,
           bbox::IteratorPairType<IndicesIterator>,
           bbox::IteratorPairType<IndicesIterator>,
           bbox::IteratorPairType<IndicesIterator>>
QuickselectMedianRange<IndicesIterator, SamplesIterator>::operator()(IndicesIterator indices_range_first,
                                                                     IndicesIterator indices_range_last,
                                                                     SamplesIterator samples_range_first,
                                                                     SamplesIterator samples_range_last,
                                                                     std::size_t     n_features,
                                                                     std::size_t     feature_index) const {
    return kdtree::algorithms::quickselect_median(
        /**/ indices_range_first,
        /**/ indices_range_last,
        /**/ samples_range_first,
        /**/ samples_range_last,
        /**/ n_features,
        /**/ feature_index);
}

}  // namespace ffcl::datastruct::kdtree::policy
