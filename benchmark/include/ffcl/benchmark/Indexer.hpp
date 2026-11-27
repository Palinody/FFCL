#pragma once

// Index construction shared by the algorithm benchmarks, so that every FFCL measurement is built the
// same way and the leaf size is decided in exactly one place.

#include "ffcl/benchmark/Dataset.hpp"

#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace ffcl::benchmark {

using IndexerType = datastruct::KDTree<std::vector<std::size_t>::iterator, std::vector<float>::iterator>;

inline auto make_indexer(std::vector<std::size_t>& indices,
                         std::vector<float>&       samples,
                         std::size_t               n_features,
                         std::size_t               bucket_size) -> IndexerType {
    using OptionsType = typename IndexerType::Options;
    // Median split on the highest-variance axis. The midpoint rule (one extent pass and one partition pass
    // instead of a selection) builds 1.6x faster but fills the buckets unevenly - 436 leaves instead of 304,
    // depth 12 instead of 9, leaves of 8 to 33 samples - and FFCL pays a function call, a bound test and a
    // sibling test per node, so it only amortises on full leaves. Measured here: kNN 0.424 -> 0.943 ms,
    // DBSCAN 10.8 -> 12.5, dual-tree MST 23.5 -> 30.4. The median rule stays the default; `MidpointRange`
    // remains available for workloads where the build dominates.
    using AxisSelectionPolicyType = datastruct::kdtree::policy::HighestVarianceBuild<std::vector<std::size_t>::iterator,
                                                                                     std::vector<float>::iterator>;
    using SplittingRulePolicyType =
        datastruct::kdtree::policy::QuickselectMedianRange<std::vector<std::size_t>::iterator,
                                                           std::vector<float>::iterator>;

    return IndexerType(indices.begin(),
                       indices.end(),
                       samples.begin(),
                       samples.end(),
                       n_features,
                       OptionsType()
                           .bucket_size(bucket_size)
                           .max_depth(static_cast<std::size_t>(std::log2(std::max<std::size_t>(2, indices.size()))) + 1)
                           .axis_selection_policy(AxisSelectionPolicyType{})
                           .splitting_rule_policy(SplittingRulePolicyType{}));
}

inline auto make_indices(std::size_t n_samples) -> std::vector<std::size_t> {
    auto indices = std::vector<std::size_t>(n_samples);

    std::iota(indices.begin(), indices.end(), static_cast<std::size_t>(0));

    return indices;
}

}  // namespace ffcl::benchmark
