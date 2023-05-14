#pragma once

#include "ffcl/containers/kdtree/KDTreeAlgorithms.hpp"

namespace kdtree::policy {

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
struct SplittingRulePolicy {
    SplittingRulePolicy() = default;

    inline virtual std::tuple<std::size_t,
                              IteratorPairType<RandomAccessIntIterator>,
                              IteratorPairType<RandomAccessIntIterator>,
                              IteratorPairType<RandomAccessIntIterator>>
    operator()(RandomAccessIntIterator index_first,
               RandomAccessIntIterator index_last,
               RandomAccessIterator    samples_first,
               RandomAccessIterator    samples_last,
               std::size_t             n_features,
               std::size_t             feature_index) const = 0;
};

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
struct QuickselectMedianRange : public SplittingRulePolicy<RandomAccessIntIterator, RandomAccessIterator> {
    inline std::tuple<std::size_t,
                      IteratorPairType<RandomAccessIntIterator>,
                      IteratorPairType<RandomAccessIntIterator>,
                      IteratorPairType<RandomAccessIntIterator>>
    operator()(RandomAccessIntIterator index_first,
               RandomAccessIntIterator index_last,
               RandomAccessIterator    samples_first,
               RandomAccessIterator    samples_last,
               std::size_t             n_features,
               std::size_t             feature_index) const;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename RandomAccessIntIterator, typename RandomAccessIterator>
std::tuple<std::size_t,
           IteratorPairType<RandomAccessIntIterator>,
           IteratorPairType<RandomAccessIntIterator>,
           IteratorPairType<RandomAccessIntIterator>>
QuickselectMedianRange<RandomAccessIterator>::operator()(RandomAccessIntIterator index_first,
                                                         RandomAccessIntIterator index_last,
                                                         RandomAccessIterator    samples_first,
                                                         RandomAccessIterator    samples_last,
                                                         std::size_t             n_features,
                                                         std::size_t             feature_index) const {
    return kdtree::algorithms::quickselect_median_indexed_range(
        index_first, index_last, samples_first, samples_last, n_features, feature_index);
}

}  // namespace kdtree::policy
