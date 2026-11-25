#include <gtest/gtest.h>

#include "ffcl/datastruct/UnionFind.hpp"

namespace {

using UnionFind = ffcl::datastruct::UnionFind<std::size_t>;

TEST(UnionFindTest, EveryElementStartsAsItsOwnComponent) {
    const auto union_find = UnionFind(5);

    for (std::size_t index = 0; index < 5; ++index) {
        EXPECT_EQ(union_find.find(index), index);
    }
}

TEST(UnionFindTest, MergeMakesBothElementsShareARepresentative) {
    auto union_find = UnionFind(5);

    const auto representative = union_find.merge(std::size_t{1}, std::size_t{3});

    EXPECT_EQ(union_find.find(1), union_find.find(3));
    EXPECT_EQ(union_find.find(1), representative);
    EXPECT_NE(union_find.find(1), union_find.find(2));
}

TEST(UnionFindTest, TryMergeReportsWhetherAMergeHappened) {
    auto union_find = UnionFind(4);

    EXPECT_TRUE(union_find.try_merge(std::size_t{0}, std::size_t{1}));
    EXPECT_FALSE(union_find.try_merge(std::size_t{0}, std::size_t{1}))
        << "merging two elements of the same component must report false";
}

TEST(UnionFindTest, MergeOfAlreadyConnectedElementsIsIdempotent) {
    auto union_find = UnionFind(4);

    const auto first_representative = union_find.merge(std::size_t{0}, std::size_t{1});
    const auto second_representative = union_find.merge(std::size_t{1}, std::size_t{0});

    EXPECT_EQ(first_representative, second_representative);
    EXPECT_EQ(union_find.find(0), union_find.find(1));
}

TEST(UnionFindTest, MergingEverythingLeavesOneComponent) {
    auto union_find = UnionFind(6);

    for (std::size_t index = 1; index < 6; ++index) {
        union_find.merge(std::size_t{0}, index);
    }
    const auto representative = union_find.find(0);

    for (std::size_t index = 0; index < 6; ++index) {
        EXPECT_EQ(union_find.find(index), representative);
    }
}

// find() is const but performs path compression; calling it must not change the answer.
TEST(UnionFindTest, FindIsStableAcrossRepeatedCalls) {
    auto union_find = UnionFind(4);

    union_find.merge(std::size_t{2}, std::size_t{3});

    const auto first  = union_find.find(2);
    const auto second = union_find.find(2);

    EXPECT_EQ(first, second);
    EXPECT_EQ(union_find.find(3), first);
}

}  // namespace
