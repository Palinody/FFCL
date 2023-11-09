#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Utils.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <vector>

TEST(FunctionTest, IgnoreParametersTest) {
    // This test will just call the function, ensuring that it compiles and runs without error
    ffcl::common::ignore_parameters(1, "test", 3.14);
}

TEST(FunctionTest, InfinityTest) {
    // Test for integer type
    ASSERT_EQ(ffcl::common::infinity<int>(), std::numeric_limits<int>::max());

    // Test for floating point type
    ASSERT_EQ(ffcl::common::infinity<double>(), std::numeric_limits<double>::max());
}

TEST(FunctionTest, AbsTest) {
    // Test for integer type
    ASSERT_EQ(ffcl::common::abs(-1), 1);
    ASSERT_EQ(ffcl::common::abs(0), 0);
    ASSERT_EQ(ffcl::common::abs(1), 1);

    // Test for floating point type
    ASSERT_EQ(ffcl::common::abs(-1.0), 1.0);
    ASSERT_EQ(ffcl::common::abs(0.0), 0.0);
    ASSERT_EQ(ffcl::common::abs(1.0), 1.0);

    // Test for non-handled type
    const char* str = "test";
    EXPECT_THROW(ffcl::common::abs(str), std::invalid_argument);
}

TEST(FunctionTest, EqualityTest) {
    // Test for integer type
    ASSERT_TRUE(ffcl::common::equality(1, 1));
    ASSERT_TRUE(ffcl::common::equality(-1, -1));
    ASSERT_FALSE(ffcl::common::equality(1, -1));

    // Test for floating point type
    ASSERT_TRUE(ffcl::common::equality(1.0, 1.0));

    ASSERT_TRUE(ffcl::common::equality(-1.0, -1.0));
    // The function allows slippage of std::numeric_limits<double>::epsilon()
    ASSERT_TRUE(ffcl::common::equality(1.0, 1.0 + std::numeric_limits<double>::epsilon()));

    ASSERT_TRUE(ffcl::common::equality(1.0, 1.0 - std::numeric_limits<double>::epsilon()));
    // 2 * std::numeric_limits<double>::epsilon() is too much slippage
    ASSERT_FALSE(ffcl::common::equality(
        1.0, 1.0 + std::numeric_limits<double>::epsilon() + std::numeric_limits<double>::epsilon()));

    ASSERT_FALSE(ffcl::common::equality(
        1.0, 1.0 - std::numeric_limits<double>::epsilon() - std::numeric_limits<double>::epsilon()));

    ASSERT_TRUE(ffcl::common::equality(1.0f, 1.0f + std::numeric_limits<float>::epsilon()));

    ASSERT_TRUE(ffcl::common::equality(1.0f, 1.0f - std::numeric_limits<float>::epsilon()));

    ASSERT_FALSE(ffcl::common::equality(
        1.0f, 1.0f + std::numeric_limits<float>::epsilon() + std::numeric_limits<float>::epsilon()));

    ASSERT_FALSE(ffcl::common::equality(
        1.0f, 1.0f - std::numeric_limits<float>::epsilon() - std::numeric_limits<float>::epsilon()));

    ASSERT_FALSE(ffcl::common::equality(std::numeric_limits<float>::min(), std::numeric_limits<float>::max()));
}

TEST(FunctionTest, InequalityTest) {
    // Test for integer type
    ASSERT_FALSE(ffcl::common::inequality(1, 1));
    ASSERT_FALSE(ffcl::common::inequality(-1, -1));
    ASSERT_TRUE(ffcl::common::inequality(1, -1));

    // Test for floating point type
    ASSERT_FALSE(ffcl::common::inequality(1.0, 1.0));

    ASSERT_FALSE(ffcl::common::inequality(-1.0, -1.0));
    // The function allows slippage of std::numeric_limits<double>::epsilon()
    ASSERT_FALSE(ffcl::common::inequality(1.0, 1.0 + std::numeric_limits<double>::epsilon()));

    ASSERT_FALSE(ffcl::common::inequality(1.0, 1.0 - std::numeric_limits<double>::epsilon()));
    // 2 * std::numeric_limits<double>::epsilon() is too much slippage
    ASSERT_TRUE(ffcl::common::inequality(
        1.0, 1.0 + std::numeric_limits<double>::epsilon() + std::numeric_limits<double>::epsilon()));

    ASSERT_TRUE(ffcl::common::inequality(
        1.0, 1.0 - std::numeric_limits<double>::epsilon() - std::numeric_limits<double>::epsilon()));

    ASSERT_FALSE(ffcl::common::inequality(1.0f, 1.0f + std::numeric_limits<float>::epsilon()));

    ASSERT_FALSE(ffcl::common::inequality(1.0f, 1.0f - std::numeric_limits<float>::epsilon()));

    ASSERT_TRUE(ffcl::common::inequality(
        1.0f, 1.0f + std::numeric_limits<float>::epsilon() + std::numeric_limits<float>::epsilon()));

    ASSERT_TRUE(ffcl::common::inequality(
        1.0f, 1.0f - std::numeric_limits<float>::epsilon() - std::numeric_limits<float>::epsilon()));

    ASSERT_TRUE(ffcl::common::inequality(std::numeric_limits<float>::min(), std::numeric_limits<float>::max()));
}

TEST(FunctionTest, DivisionTest) {
    // Test for integer type
    ASSERT_EQ(ffcl::common::division(10, 2), 5);
    ASSERT_EQ(ffcl::common::division(10, 3), 3);
    ASSERT_EQ(ffcl::common::division(10, 0), 0);

    // Test for floating point type
    ASSERT_NEAR(ffcl::common::division(10.0, 2.0), 5.0, std::numeric_limits<double>::epsilon());
    ASSERT_NEAR(ffcl::common::division(10.0, 3.0), 3.3333333333333333, 1e-14);
    ASSERT_EQ(ffcl::common::division(10.0, 0.0), 0.0);
}

TEST(FunctionTest, ToTypeTest) {
    std::vector<int> v1  = {1, 2, 3};
    auto             res = ffcl::common::to_type<double>(v1.begin(), v1.end());
    ASSERT_TRUE(res.size() == 3);
    ASSERT_TRUE(ffcl::common::equality(res[0], 1.0));
    ASSERT_TRUE(ffcl::common::equality(res[1], 2.0));
    ASSERT_TRUE(ffcl::common::equality(res[2], 3.0));

    std::vector<double> v2   = {1.1, 2.2, 3.3};
    auto                res2 = ffcl::common::to_type<double>(v2.begin(), v2.end());
    ASSERT_TRUE(res2.size() == 3);
    ASSERT_TRUE(ffcl::common::equality(res2[0], 1.1));
    ASSERT_TRUE(ffcl::common::equality(res2[1], 2.2));
    ASSERT_TRUE(ffcl::common::equality(res2[2], 3.3));

    std::vector<double> v3   = {1.2, -2.3, 3.99999};
    auto                res3 = ffcl::common::to_type<int>(v3.begin(), v3.end());
    ASSERT_TRUE(res3.size() == 3);
    ASSERT_TRUE(res3[0] == 1);
    ASSERT_TRUE(res3[1] == -2);
    ASSERT_TRUE(res3[2] == 3);
}

TEST(FunctionTest, AreContainersEqualTest) {
    // Test for integral datastruct
    std::vector<int> a{1, 2, 3, 4, 5};
    std::vector<int> b{1, 2, 3, 4, 5};
    std::vector<int> c{1, 2, 3, 4, 6};
    ASSERT_TRUE(ffcl::common::are_containers_equal(a.begin(), a.end(), b.begin()));
    ASSERT_TRUE(!ffcl::common::are_containers_equal(a.begin(), a.end(), c.begin()));
    ASSERT_TRUE(ffcl::common::are_containers_equal(a, b));
    ASSERT_TRUE(!ffcl::common::are_containers_equal(a, c));

    // Test for floating-point datastruct with tolerance
    std::vector<float> d{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> e{1.0f, 2.0f, 3.0f, 4.0f, 5.000001f};
    std::vector<float> f{1.0f, 2.0f, 3.0f, 4.0f, 6.0f};
    ASSERT_TRUE(ffcl::common::are_containers_equal(d.begin(), d.end(), e.begin(), 0.00001f));
    ASSERT_TRUE(!ffcl::common::are_containers_equal(d.begin(), d.end(), f.begin(), 0.00001f));
    ASSERT_TRUE(ffcl::common::are_containers_equal(d, e, 0.00001f));
    ASSERT_TRUE(!ffcl::common::are_containers_equal(d, f, 0.00001f));

    // Test for datastruct of other types (unsupported)
    std::vector<char> g{'a', 'b', 'c', 'd', 'e'};
    std::vector<char> h{'a', 'b', 'c', 'd', 'e'};
    std::vector<char> i{'a', 'b', 'c', 'd', 'f'};
    ASSERT_TRUE(ffcl::common::are_containers_equal(g.begin(), g.end(), h.begin()));
    ASSERT_TRUE(!ffcl::common::are_containers_equal(g.begin(), g.end(), i.begin()));
    ASSERT_TRUE(ffcl::common::are_containers_equal(g, h));
    ASSERT_TRUE(!ffcl::common::are_containers_equal(g, i));
}

TEST(CountMatchesTest, CountMatchesTest) {
    std::vector<int> v1{1, 2, 3, 4, -5};
    std::vector<int> v2{1, 2, 6, 7, -5};
    std::size_t      count = ffcl::common::count_matches(v1.begin(), v1.end(), v2.begin());
    EXPECT_EQ(count, 3);
}

TEST(CountMatchesForValueTest, CountMatchesForValueTest) {
    std::vector<int> v1{1, 4, 3, 4, 5};
    std::vector<int> v2{2, 4, 6, 4, 10};
    std::size_t      count = ffcl::common::count_matches_for_value(v1.begin(), v1.end(), v2.begin(), 4);
    EXPECT_EQ(count, 2);
}

TEST(PermutationFromIndicesTest, PermutationFromIndicesTest) {
    std::vector<int> v{1, 2, 3, 4};
    std::vector<int> indices{3, 2, 0, 1};
    std::vector<int> result = ffcl::common::permutation_from_indices(indices, v);
    std::vector<int> expected{4, 3, 1, 2};
    EXPECT_EQ(result, expected);
}

TEST(RangePermutationFromIndicesTest, RangePermutationFromIndicesTest) {
    std::vector<int> v{1, 2, 3, 4, 5, 6};
    std::vector<int> indices{2, 1, 0};
    std::vector<int> result = ffcl::common::remap_ranges_from_indices(indices, v, 2);
    std::vector<int> expected{5, 6, 3, 4, 1, 2};
    EXPECT_EQ(result, expected);
}

TEST(CommonUtilsTest, GetOneSampleTest) {
    std::vector<int> data{1};
    std::size_t      n_features = 1;

    printf("Distance: %ld\n", std::distance(data.begin(), data.end()));

    std::size_t n_samples = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);
    ASSERT_EQ(n_samples, 1);

// Test assertion when input data missing values or wrong number of features specified.
// Test only if assert enabled
#ifndef NDEBUG
    n_features = 2;
    ASSERT_DEATH(ffcl::common::get_n_samples(data.begin(), data.end(), n_features), ".*");
#endif
}

TEST(CommonUtilsTest, GetNSamplesTest) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    std::size_t      n_features = 2;

    std::size_t n_samples = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);
    ASSERT_EQ(n_samples, 3);

// Test assertion when input data missing values or wrong number of features specified.
// Test only if assert enabled
#ifndef NDEBUG
    n_features = 5;
    ASSERT_DEATH(ffcl::common::get_n_samples(data.begin(), data.end(), n_features), ".*");
#endif
}

TEST(CommonUtilsTest, IsElementInTest) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    int              element = 4;

    bool is_in = ffcl::common::is_element_in(data.begin(), data.end(), element);
    ASSERT_TRUE(is_in);

    element = 7;
    is_in   = ffcl::common::is_element_in(data.begin(), data.end(), element);
    ASSERT_FALSE(is_in);
}

TEST(CommonUtilsTest, IsElementNotInTest) {
    std::vector<int> data{1, 2, 3, 4, 5, 6};
    int              element = 4;

    bool is_not_in = ffcl::common::is_element_not_in(data.begin(), data.end(), element);
    ASSERT_FALSE(is_not_in);

    element   = 7;
    is_not_in = ffcl::common::is_element_not_in(data.begin(), data.end(), element);
    ASSERT_TRUE(is_not_in);
}

TEST(CommonUtilsTest, IsElementInFirstTest) {
    std::vector<std::pair<float, int>> data{{1.1f, 1}, {2.2f, 2}, {3.3f, 3}};
    float                              element = 2.2f;

    bool is_in = ffcl::common::is_element_in_first(data.begin(), data.end(), element);
    ASSERT_TRUE(is_in);

    element = 4.4f;
    is_in   = ffcl::common::is_element_in_first(data.begin(), data.end(), element);
    ASSERT_FALSE(is_in);
}

TEST(CommonUtilsTest, IsElementNotInFirstTest) {
    std::vector<std::pair<int, float>> data{{1, 1.1f}, {2, 2.2f}, {3, 3.3f}};
    int                                element = 2;

    bool is_not_in = ffcl::common::is_element_not_in_first(data.begin(), data.end(), element);
    ASSERT_FALSE(is_not_in);

    element   = 4;
    is_not_in = ffcl::common::is_element_not_in_first(data.begin(), data.end(), element);
    ASSERT_TRUE(is_not_in);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}