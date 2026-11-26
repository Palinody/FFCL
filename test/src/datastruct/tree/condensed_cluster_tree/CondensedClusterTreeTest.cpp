#include <gtest/gtest.h>

#include "ffcl/datastruct/tree/condensed_cluster_tree/CondensedClusterTree.hpp"
#include "ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

#include <vector>

namespace {

namespace fixtures = ffcl::fixtures;

using ValueType = float;
using IndexType = std::size_t;

using SingleLinkageClusterTreeType = ffcl::datastruct::SingleLinkageClusterTree<IndexType, ValueType>;

// The condensed tree is built from a single linkage tree; the single linkage options do not change
// its shape, so they are left at their defaults here.
auto make_condensed_tree(const fixtures::MSTType&                                                     mst,
                         const ffcl::datastruct::CondensedClusterTree<IndexType, ValueType>::Options& options) {
    auto single_linkage_tree = SingleLinkageClusterTreeType(mst);

    return ffcl::datastruct::CondensedClusterTree<IndexType, ValueType>(single_linkage_tree.root(), options);
}

using CondensedOptions = ffcl::datastruct::CondensedClusterTree<IndexType, ValueType>::Options;

// Both sides of the bridge hold enough samples to be their own cluster, so the bridge is a real split
// and the two pairs are separated. Labels start at 1; 0 means noise.
TEST(CondensedClusterTreeTest, TwoSeparatedPairsBecomeTwoClusters) {
    auto tree = make_condensed_tree(fixtures::two_pairs_mst(), CondensedOptions().min_cluster_size(2));

    const auto labels = tree.predict();

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels[0], labels[1]);
    EXPECT_EQ(labels[2], labels[3]);
    EXPECT_NE(labels[0], labels[2]);
    EXPECT_EQ(fixtures::count_clusters(labels, IndexType{0}), 2u);
    EXPECT_EQ(fixtures::count_clusters(labels, IndexType{0}) * 2, labels.size()) << "no sample is left as noise";
}

// Raising the minimum cluster size above both sides leaves no split candidate, so the branch
// terminates at the root: the root becomes a leaf and keeps every sample in a single cluster.
TEST(CondensedClusterTreeTest, NoSplitCandidateCollapsesToASingleCluster) {
    auto tree = make_condensed_tree(fixtures::two_pairs_mst(), CondensedOptions().min_cluster_size(3));

    const auto labels = tree.predict();

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(fixtures::count_distinct_labels(labels), 1u);
    EXPECT_EQ(fixtures::count_clusters(labels, IndexType{0}), 1u);
}

// `return_leaf_nodes` stops the build from running the stability selection. The condensed tree that
// comes out has the same shape, so it must recover the same partition.
TEST(CondensedClusterTreeTest, ReturnLeafNodesMatchesTheStabilitySelection) {
    const auto mst = fixtures::two_pairs_mst();

    auto with_selection = make_condensed_tree(mst, CondensedOptions().min_cluster_size(2));

    auto without_selection = make_condensed_tree(mst, CondensedOptions().min_cluster_size(2).return_leaf_nodes(true));

    EXPECT_TRUE(fixtures::same_partition(with_selection.predict(), without_selection.predict()));
}

// `min_cluster_size` decides which sides of a split count as clusters. The 3-sample side survives
// either way; the 2-sample side is a cluster at a minimum of 2 and stops being one at a minimum of 3,
// at which point no split candidate is left and the branch terminates at the root as a single cluster.
TEST(CondensedClusterTreeTest, MinClusterSizeDecidesWhichSidesBecomeClusters) {
    const auto small_minimum = make_condensed_tree(fixtures::lopsided_mst(), CondensedOptions().min_cluster_size(2));

    const auto small_minimum_labels = small_minimum.predict();

    ASSERT_EQ(small_minimum_labels.size(), 5u);
    EXPECT_EQ(fixtures::count_clusters(small_minimum_labels, IndexType{0}), 2u);
    EXPECT_EQ(small_minimum_labels[0], small_minimum_labels[1]);
    EXPECT_EQ(small_minimum_labels[1], small_minimum_labels[2]);
    EXPECT_EQ(small_minimum_labels[3], small_minimum_labels[4]) << "the 2-sample side is its own cluster";
    EXPECT_NE(small_minimum_labels[2], small_minimum_labels[3]);

    const auto large_minimum = make_condensed_tree(fixtures::lopsided_mst(), CondensedOptions().min_cluster_size(3));

    const auto large_minimum_labels = large_minimum.predict();

    ASSERT_EQ(large_minimum_labels.size(), 5u);
    EXPECT_EQ(fixtures::count_clusters(large_minimum_labels, IndexType{0}), 1u)
        << "the 2-sample side is below the minimum and no longer splits";
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
