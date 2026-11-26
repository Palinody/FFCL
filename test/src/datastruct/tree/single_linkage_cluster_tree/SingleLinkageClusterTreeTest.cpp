#include <gtest/gtest.h>

#include "ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

#include <vector>

namespace {

namespace fixtures = ffcl::fixtures;

using ValueType = float;
using IndexType = std::size_t;

using SingleLinkageClusterTreeType = ffcl::datastruct::SingleLinkageClusterTree<IndexType, ValueType>;

using Options = SingleLinkageClusterTreeType::Options;

// The default cut level is infinite, so no merge reaches it, and the single cluster at the top of
// the tree keeps every sample.
TEST(SingleLinkageClusterTreeTest, DefaultCutLevelYieldsASingleCluster) {
    auto tree = SingleLinkageClusterTreeType(fixtures::staircase_mst());

    const auto labels = tree.predict();

    ASSERT_EQ(labels.size(), 6u);
    EXPECT_EQ(fixtures::count_distinct_labels(labels), 1u);
}

// Cutting at the level of the pair merges splits the tree there: the two pairs become separate
// clusters and the bridge, which is above the cut, does not unite them.
TEST(SingleLinkageClusterTreeTest, CutLevelSeparatesThePairsBelowTheBridge) {
    auto tree = SingleLinkageClusterTreeType(fixtures::two_pairs_mst());

    tree.set_options(Options().cut_level(1.0F));

    const auto labels = tree.predict();

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels[0], labels[1]) << "samples 0 and 1 are merged below the cut";
    EXPECT_EQ(labels[2], labels[3]) << "samples 2 and 3 are merged below the cut";
    EXPECT_NE(labels[0], labels[2]) << "the bridge is above the cut";
    EXPECT_EQ(fixtures::count_distinct_labels(labels), 2u);
}

// Raising the cut above the bridge level leaves no merge at or above the cut, so the same tree
// collapses back into one cluster. The cut level, not the tree, decides.
TEST(SingleLinkageClusterTreeTest, CutAboveTheBridgeMergesThePairs) {
    auto tree = SingleLinkageClusterTreeType(fixtures::two_pairs_mst());

    tree.set_options(Options().cut_level(2.0F));

    const auto labels = tree.predict();

    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(fixtures::count_distinct_labels(labels), 1u);
}

// A node at or above the cut whose subtree holds fewer samples than `min_cluster_size` is labelled as
// noise instead of being handed its own cluster: here the 3-sample side is below the minimum of 4 and
// becomes noise, while the other side is below the cut level and keeps the root's label.
TEST(SingleLinkageClusterTreeTest, MinClusterSizeTurnsASmallNodeIntoNoise) {
    auto tree = SingleLinkageClusterTreeType(fixtures::lopsided_mst());

    tree.set_options(Options().cut_level(1.5F).min_cluster_size(4));

    const auto labels = tree.predict();

    ASSERT_EQ(labels.size(), 5u);
    for (std::size_t sample_index = 0; sample_index < 3; ++sample_index) {
        EXPECT_EQ(labels[sample_index], 0u)
            << "sample " << sample_index << " sits in the 3-sample node, below min_cluster_size";
    }
    EXPECT_EQ(labels[3], labels[4]);
    EXPECT_NE(labels[3], 0u) << "the other side keeps the root's label";
}

// The options can be handed to the constructor or set afterwards; both must give the same tree.
TEST(SingleLinkageClusterTreeTest, OptionsCanBeSetAtConstruction) {
    const auto options = Options().cut_level(1.0F).min_cluster_size(1);

    auto from_constructor = SingleLinkageClusterTreeType(fixtures::two_pairs_mst(), options);

    auto from_setter = SingleLinkageClusterTreeType(fixtures::two_pairs_mst());
    from_setter.set_options(options);

    EXPECT_TRUE(fixtures::same_partition(from_constructor.predict(), from_setter.predict()));
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
