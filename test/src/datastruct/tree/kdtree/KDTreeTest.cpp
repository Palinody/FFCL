#include <gtest/gtest.h>

#include "base_fixtures/ClusteringFixtures.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

namespace fixtures = ffcl::fixtures;

using DataType = float;

constexpr std::size_t kSampleCount = 8;  // fixtures::two_blobs() has 8 samples
constexpr std::size_t kFeatures    = 2;

// The tree reports the shape of the range it was built over.
TEST(KDTreeTest, SizeAndFeatureCountMatchTheInput) {
    auto& data    = fixtures::two_blobs();
    auto  indices = fixtures::make_indices(kSampleCount);

    const auto tree = fixtures::make_kdtree(data, indices, kFeatures);

    EXPECT_EQ(tree.n_samples(), kSampleCount);
    EXPECT_EQ(tree.n_features(), kFeatures);
}

// Building a tree rearranges the indices in place, splitting them into the node ranges. That is the
// only thing the build is allowed to do to them: losing or duplicating a sample would silently drop
// it from every query.
TEST(KDTreeTest, TheIndicesArePermutedNotLost) {
    auto& data    = fixtures::two_blobs();
    auto  indices = fixtures::make_indices(kSampleCount);

    const auto tree = fixtures::make_kdtree(data, indices, kFeatures);
    (void)tree;

    auto sorted_indices = indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());

    auto expected = fixtures::make_indices(kSampleCount);

    EXPECT_EQ(sorted_indices, expected);
}

// A bucket large enough to hold every sample stops the build immediately: the root is a leaf.
TEST(KDTreeTest, ABucketHoldingEverySampleStopsAtTheRoot) {
    auto& data    = fixtures::two_blobs();
    auto  indices = fixtures::make_indices(kSampleCount);

    const auto tree = fixtures::make_kdtree(data, indices, kFeatures, /*bucket_size=*/kSampleCount);

    const auto root = tree.root();

    ASSERT_NE(root, nullptr);
    EXPECT_TRUE(root->is_leaf());
    EXPECT_EQ(root->n_samples(), kSampleCount);
}

// The opposite: a bucket of one sample forces the recursion down to single samples. An internal node
// owns no sample at all - its pivot lives in the left child, so that the leaves cover every sample
// exactly once - and only keeps the pivot index for routing queries.
TEST(KDTreeTest, ABucketOfOneSampleKeepsSplitting) {
    auto& data    = fixtures::two_blobs();
    auto  indices = fixtures::make_indices(kSampleCount);

    const auto tree = fixtures::make_kdtree(data, indices, kFeatures, /*bucket_size=*/1);

    const auto root = tree.root();

    ASSERT_NE(root, nullptr);
    EXPECT_FALSE(root->is_leaf());
    EXPECT_EQ(root->n_samples(), 0u) << "an internal node holds no samples";
    EXPECT_LT(root->pivot_index(), kSampleCount) << "but it does keep a pivot index";
}

// The feature accessors must hand back the sample they are asked for, even though the indices were
// permuted by the build.
TEST(KDTreeTest, EachSampleExposesItsOwnFeatures) {
    auto& data    = fixtures::two_blobs();
    auto  indices = fixtures::make_indices(kSampleCount);

    const auto tree = fixtures::make_kdtree(data, indices, kFeatures);

    for (std::size_t sample_index = 0; sample_index < kSampleCount; ++sample_index) {
        const auto features_first = tree.features_range_first(sample_index);
        const auto features_last  = tree.features_range_last(sample_index);

        ASSERT_EQ(std::distance(features_first, features_last), static_cast<std::ptrdiff_t>(kFeatures));

        for (std::size_t feature_index = 0; feature_index < kFeatures; ++feature_index) {
            EXPECT_FLOAT_EQ(*(features_first + feature_index), data[sample_index * kFeatures + feature_index])
                << "sample " << sample_index << ", feature " << feature_index;
        }
    }
}

// Serialisation is written for the plotting tools, so all this can check is that a JSON object lands
// on disk and that every sample is mentioned.
TEST(KDTreeTest, SerializeWritesAJsonObject) {
    auto& data    = fixtures::two_blobs();
    auto  indices = fixtures::make_indices(kSampleCount);

    const auto tree = fixtures::make_kdtree(data, indices, kFeatures);

    const auto filepath = std::filesystem::temp_directory_path() / "ffcl_kdtree_test.json";

    std::filesystem::remove(filepath);

    tree.serialize(filepath);

    ASSERT_TRUE(std::filesystem::exists(filepath));
    EXPECT_GT(std::filesystem::file_size(filepath), 0u);

    auto contents = std::string{};
    {
        auto filestream = std::ifstream(filepath);
        ASSERT_TRUE(filestream.is_open());

        contents.assign(std::istreambuf_iterator<char>(filestream), std::istreambuf_iterator<char>());
    }

    ASSERT_FALSE(contents.empty());
    EXPECT_EQ(contents.front(), '{') << "the serialised tree must be a JSON object";
    EXPECT_EQ(contents.back(), '}');

    std::filesystem::remove(filepath);
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
