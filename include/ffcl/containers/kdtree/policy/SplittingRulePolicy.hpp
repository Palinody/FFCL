#pragma once

#include "ffcl/containers/kdtree/KDTreeAlgorithms.hpp"

namespace kdtree::policy {

template <typename RandomAccessIterator>
struct SplittingRulePolicy {
    SplittingRulePolicy() = default;

    inline virtual std::tuple<std::size_t,
                              IteratorPairType<RandomAccessIterator>,
                              IteratorPairType<RandomAccessIterator>,
                              IteratorPairType<RandomAccessIterator>>
    operator()(RandomAccessIterator samples_first,
               RandomAccessIterator samples_last,
               std::size_t          n_features,
               std::size_t          feature_index) const = 0;
};

template <typename RandomAccessIterator>
struct QuickselectMedianRange : public SplittingRulePolicy<RandomAccessIterator> {
    inline std::tuple<std::size_t,
                      IteratorPairType<RandomAccessIterator>,
                      IteratorPairType<RandomAccessIterator>,
                      IteratorPairType<RandomAccessIterator>>
    operator()(RandomAccessIterator samples_first,
               RandomAccessIterator samples_last,
               std::size_t          n_features,
               std::size_t          feature_index) const;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename RandomAccessIterator>
std::tuple<std::size_t,
           IteratorPairType<RandomAccessIterator>,
           IteratorPairType<RandomAccessIterator>,
           IteratorPairType<RandomAccessIterator>>
QuickselectMedianRange<RandomAccessIterator>::operator()(RandomAccessIterator samples_first,
                                                         RandomAccessIterator samples_last,
                                                         std::size_t          n_features,
                                                         std::size_t          feature_index) const {
    return kdtree::algorithms::quickselect_median_range(samples_first, samples_last, n_features, feature_index);
}

}  // namespace kdtree::policy
