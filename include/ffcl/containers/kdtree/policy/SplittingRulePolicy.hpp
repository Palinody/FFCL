#pragma once

#include "ffcl/containers/kdtree/KDTreeUtils.hpp"

namespace kdtree::policy {

template <typename RandomAccessIterator>
struct SplittingRulePolicy {
    SplittingRulePolicy() = default;

    inline std::tuple<std::size_t,
                      IteratorPairType<RandomAccessIterator>,
                      IteratorPairType<RandomAccessIterator>,
                      IteratorPairType<RandomAccessIterator>>
    operator()(IteratorPairType<RandomAccessIterator> iterator_pair,
               std::size_t                            n_features,
               std::size_t                            feature_index) const;

  private:
    SplittingRulePolicy(const SplittingRulePolicy&) = delete;

    SplittingRulePolicy& operator=(const SplittingRulePolicy&) = delete;
};

template <typename RandomAccessIterator>
struct QuickselectMedianRange : public SplittingRulePolicy<RandomAccessIterator> {
    inline std::tuple<std::size_t,
                      IteratorPairType<RandomAccessIterator>,
                      IteratorPairType<RandomAccessIterator>,
                      IteratorPairType<RandomAccessIterator>>
    operator()(IteratorPairType<RandomAccessIterator> iterator_pair,
               std::size_t                            n_features,
               std::size_t                            feature_index) const;
};

}  // namespace kdtree::policy

namespace kdtree::policy {

template <typename RandomAccessIterator>
std::tuple<std::size_t,
           IteratorPairType<RandomAccessIterator>,
           IteratorPairType<RandomAccessIterator>,
           IteratorPairType<RandomAccessIterator>>
QuickselectMedianRange<RandomAccessIterator>::operator()(IteratorPairType<RandomAccessIterator> iterator_pair,
                                                         std::size_t                            n_features,
                                                         std::size_t                            feature_index) const {
    return kdtree::utils::quickselect_median_range(iterator_pair, n_features, feature_index);
}

}  // namespace kdtree::policy
