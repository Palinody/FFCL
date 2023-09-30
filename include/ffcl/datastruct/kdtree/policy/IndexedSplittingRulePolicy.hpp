#pragma once

#include "ffcl/datastruct/kdtree/KDTreeAlgorithms.hpp"

namespace kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
struct IndexedSplittingRulePolicy {
    IndexedSplittingRulePolicy() = default;

    inline virtual std::tuple<std::size_t,
                              ffcl::bbox::IteratorPairType<IndicesIterator>,
                              ffcl::bbox::IteratorPairType<IndicesIterator>,
                              ffcl::bbox::IteratorPairType<IndicesIterator>>
    operator()(IndicesIterator index_first,
               IndicesIterator index_last,
               SamplesIterator samples_first,
               SamplesIterator samples_last,
               std::size_t     n_features,
               std::size_t     feature_index) const = 0;
};

template <typename IndicesIterator, typename SamplesIterator>
struct IndexedQuickselectMedianRange : public IndexedSplittingRulePolicy<IndicesIterator, SamplesIterator> {
    inline std::tuple<std::size_t,
                      ffcl::bbox::IteratorPairType<IndicesIterator>,
                      ffcl::bbox::IteratorPairType<IndicesIterator>,
                      ffcl::bbox::IteratorPairType<IndicesIterator>>
    operator()(IndicesIterator index_first,
               IndicesIterator index_last,
               SamplesIterator samples_first,
               SamplesIterator samples_last,
               std::size_t     n_features,
               std::size_t     feature_index) const;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename IndicesIterator, typename SamplesIterator>
std::tuple<std::size_t,
           ffcl::bbox::IteratorPairType<IndicesIterator>,
           ffcl::bbox::IteratorPairType<IndicesIterator>,
           ffcl::bbox::IteratorPairType<IndicesIterator>>
IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>::operator()(IndicesIterator index_first,
                                                                            IndicesIterator index_last,
                                                                            SamplesIterator samples_first,
                                                                            SamplesIterator samples_last,
                                                                            std::size_t     n_features,
                                                                            std::size_t     feature_index) const {
    return kdtree::algorithms::quickselect_median_indexed_range(
        /**/ index_first,
        /**/ index_last,
        /**/ samples_first,
        /**/ samples_last,
        /**/ n_features,
        /**/ feature_index);
}

}  // namespace kdtree::policy
