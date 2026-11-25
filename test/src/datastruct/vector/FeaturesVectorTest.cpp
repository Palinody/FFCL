#include <gtest/gtest.h>

#include "ffcl/datastruct/vector/FeaturesVector.hpp"

namespace {

using FeaturesVector = ffcl::datastruct::FeaturesVector<float, 0>;

TEST(FeaturesVectorTest, DynamicSizeKeepsInsertionOrder) {
    const auto values = FeaturesVector({1.0f, 2.0f, 3.0f});

    EXPECT_EQ(values.size(), 3u);
    EXPECT_FLOAT_EQ(values[0], 1.0f);
    EXPECT_FLOAT_EQ(values[2], 3.0f);
}

TEST(FeaturesVectorTest, ConstructibleFromIteratorRange) {
    // the iterator constructor takes the container's own Iterator type, so the source has to be
    // mutable
    auto       source = std::vector<float>{4.0f, 5.0f, 6.0f};
    const auto values = FeaturesVector(source.begin(), source.end());

    EXPECT_EQ(values.size(), 3u);
    EXPECT_FLOAT_EQ(values[0], 4.0f);
    EXPECT_FLOAT_EQ(values[2], 6.0f);
}

TEST(FeaturesVectorTest, ElementsAreMutable) {
    auto values = FeaturesVector({1.0f, 2.0f});

    values[0] = 42.0f;

    EXPECT_FLOAT_EQ(values[0], 42.0f);
    EXPECT_FLOAT_EQ(values[1], 2.0f);
}

TEST(FeaturesVectorTest, FixedSizeStoresValuesInline) {
    const auto values = ffcl::datastruct::FeaturesVector<float, 3>(7.0f, 8.0f, 9.0f);

    EXPECT_EQ(values.size(), 3u);
    EXPECT_FLOAT_EQ(values[0], 7.0f);
    EXPECT_FLOAT_EQ(values[2], 9.0f);
}

TEST(FeaturesVectorTest, IterationCoversEveryElement) {
    const auto values = FeaturesVector({1.0f, 2.0f, 3.0f});

    auto sum = 0.0f;
    for (const auto& value : values) {
        sum += value;
    }
    EXPECT_FLOAT_EQ(sum, 6.0f);
}

}  // namespace
