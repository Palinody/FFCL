#pragma once

// Small, deterministic fixtures shared by the algorithm unit tests.
//
// These replace the previous dataset-driven tests: each one is a handful of points whose expected
// result can be verified by hand, so a test failure points at the algorithm rather than at a
// generated dataset. Anything that needs a real dataset belongs in benchmark/.

#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

namespace ffcl::fixtures {

// Two tight, well-separated squares in 2D:
//
//   cluster 0 (upper left)   x in [0,1],   y in [0,1]
//   cluster 1 (lower right)  x in [9,10],  y in [9,10]
//
// The gap between the squares is 8, so any radius in (2, 8) separates them cleanly and any
// centroid/medoid based algorithm converges immediately.
inline auto two_blobs() -> std::vector<float>& {
    static auto data = std::vector<float>{
        // clang-format off
        // cluster 0
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        0.5f, 0.5f,
        // cluster 1
        9.0f, 9.0f,
        10.0f, 9.0f,
        9.0f, 10.0f,
        9.5f, 9.5f,
        // clang-format on
    };
    return data;
}

// two_blobs() plus two isolated points, neither of which is within radius 2 of any other sample.
inline auto two_blobs_with_outliers() -> std::vector<float>& {
    static auto data = [] {
        auto values = two_blobs();
        // clang-format off
        values.insert(values.end(), { -20.0f, 5.0f });   // isolated on the left
        values.insert(values.end(), {  20.0f, 5.0f });   // isolated on the right
        // clang-format on
        return values;
    }();
    return data;
}

inline auto make_indices(std::size_t n_samples) -> std::vector<std::size_t> {
    auto indices = std::vector<std::size_t>(n_samples);

    std::iota(indices.begin(), indices.end(), static_cast<std::size_t>(0));

    return indices;
}

// Builds a KDTree over `data`.
//
// `data` and `indices` must outlive the returned tree: the tree only rearranges the index range and
// keeps iterators into both containers. The fixture containers are returned by non-const reference
// because the KDTree's SamplesIterator has to be the container's own (mutable) iterator.
template <typename DataType = float>
auto make_kdtree(std::vector<DataType>&    data,
                 std::vector<std::size_t>& indices,
                 std::size_t               n_features,
                 std::size_t               bucket_size = 0) {
    using IndicesIterator         = typename std::vector<std::size_t>::iterator;
    using SamplesIterator         = typename std::vector<DataType>::iterator;
    using IndexerType             = datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    const auto n_samples = indices.size();

    return IndexerType(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        n_features,
        OptionsType()
            .bucket_size(bucket_size != 0 ? bucket_size
                                          : std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(n_samples))))
            .max_depth(static_cast<std::size_t>(std::log2(n_samples)) + 1)
            .axis_selection_policy(AxisSelectionPolicyType{})
            .splitting_rule_policy(SplittingRulePolicyType{}));
}

// Number of distinct non-noise labels. `noise_label` is excluded from the count.
template <typename LabelType>
auto count_clusters(const std::vector<LabelType>& labels, const LabelType& noise_label) -> std::size_t {
    auto sorted = labels;

    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    const auto has_noise = !sorted.empty() && sorted.front() == noise_label;

    return sorted.size() - (has_noise ? 1 : 0);
}

// Number of distinct labels, counting every label (noise included).
template <typename LabelType>
auto count_distinct_labels(const std::vector<LabelType>& labels) -> std::size_t {
    auto sorted = labels;

    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    return sorted.size();
}

// Two labellings agree when they group the same samples together, whatever the label values are.
// Cluster labels are not comparable across implementations (or even across runs of the same one),
// so the partitions are what a test should assert.
template <typename LabelType>
auto same_partition(const std::vector<LabelType>& left, const std::vector<LabelType>& right) -> bool {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        for (std::size_t j = 0; j < left.size(); ++j) {
            if ((left[i] == left[j]) != (right[i] == right[j])) {
                return false;
            }
        }
    }
    return true;
}

// Minimum spanning trees for the cluster-tree tests, given as {sample_1, sample_2, merge_level}.
// The cluster trees only read the merge levels, so the samples behind the indices do not have to be
// real coordinates.
using MSTType = datastruct::mst::EdgesList<std::size_t, float>;

inline auto make_mst(std::vector<std::tuple<std::size_t, std::size_t, float>> edges) -> MSTType {
    return edges;
}

// A chain of merges with strictly increasing levels, so a cut level selects exactly the merges below
// it: 0-1 at 1.0, then 2 at 2.0, 3 at 4.0, 4 at 8.0 and 5 at 16.0.
inline auto staircase_mst() -> MSTType {
    return make_mst({{0, 1, 1.0F}, {1, 2, 2.0F}, {2, 3, 4.0F}, {3, 4, 8.0F}, {4, 5, 16.0F}});
}

// Two pairs merged at 1.0, bridged at 8.0, so a cut level between the two separates the pairs.
//
// The merge order is {1,0} rather than {0,1} so that the union-find representatives, which the
// single linkage tree stores as its cluster labels, differ between the two sides.
inline auto two_pairs_mst() -> MSTType {
    return make_mst({{1, 0, 1.0F}, {2, 3, 1.0F}, {1, 2, 8.0F}});
}

// One side of 3 samples merged at 1.5 and one side of 2 samples merged at 1.0, bridged at 8.0.
inline auto lopsided_mst() -> MSTType {
    return make_mst({{1, 0, 1.0F}, {1, 2, 1.5F}, {3, 4, 1.0F}, {1, 3, 8.0F}});
}

}  // namespace ffcl::fixtures
