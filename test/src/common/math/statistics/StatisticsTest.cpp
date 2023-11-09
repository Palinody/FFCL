#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Range2DBaseFixture.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/statistics/Statistics.hpp"

#include <vector>

template <typename DataType>
class MathStatisticsTestFixture : public Range2DBaseFixture<DataType> {
  public:
    void SetUp() override {
        if constexpr (std::is_integral_v<DataType> && std::is_signed_v<DataType>) {
            lower_bound_ = -10;
            upper_bound_ = 10;

        } else if constexpr (std::is_integral_v<DataType> && std::is_unsigned_v<DataType>) {
            lower_bound_ = 0;
            upper_bound_ = 10;

        } else if constexpr (std::is_floating_point_v<DataType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;
        }

        n_samples_  = 32;
        n_features_ = 23;
        n_tests_    = 10;
    }

  protected:
    DataType    lower_bound_;
    DataType    upper_bound_;
    std::size_t n_samples_;
    std::size_t n_features_;
    std::size_t n_tests_;
};

using DataTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MathStatisticsTestFixture, DataTypes);

TYPED_TEST(MathStatisticsTestFixture, ComputeMeanPerFeature) {
    using DataType = TypeParam;

    const std::vector<DataType>  data          = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<DataType>  expected_mean = {4, 5, 6};
    static constexpr std::size_t n_features    = 3;

    const auto result = ffcl::common::math::statistics::compute_mean_per_feature(data.begin(), data.end(), n_features);

    ASSERT_EQ(result.size(), n_features);

    ASSERT_TRUE(this->ranges_equality(result.begin(), result.end(), expected_mean.begin(), expected_mean.end()));
}

TYPED_TEST(MathStatisticsTestFixture, ComputeVariancePerFeature) {
    using DataType = TypeParam;

    const std::vector<DataType>  data              = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<DataType>  expected_variance = {9, 9, 9};
    static constexpr std::size_t n_features        = 3;

    const auto result =
        ffcl::common::math::statistics::compute_variance_per_feature(data.begin(), data.end(), n_features);

    ASSERT_EQ(result.size(), n_features);

    ASSERT_TRUE(
        this->ranges_equality(result.begin(), result.end(), expected_variance.begin(), expected_variance.end()));
}

TYPED_TEST(MathStatisticsTestFixture, ComputeVariance) {
    using DataType = TypeParam;

    const std::vector<DataType> data              = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const DataType              expected_variance = 7.5;

    const auto result1 = ffcl::common::math::statistics::compute_variance_per_feature(data.begin(), data.end(), 1)[0];
    const auto result2 = ffcl::common::math::statistics::compute_variance(data.begin(), data.end());

    EXPECT_DOUBLE_EQ(result1, static_cast<DataType>(expected_variance));
    EXPECT_DOUBLE_EQ(result2, static_cast<DataType>(expected_variance));
}

TYPED_TEST(MathStatisticsTestFixture, ArgmaxVariancePerFeature) {
    for (std::size_t test_index = 0; test_index < this->n_tests_; ++test_index) {
        const auto data = this->generate_random_uniform_vector(
            this->n_samples_, this->n_features_, this->lower_bound_, this->upper_bound_);

        const auto var_per_feat =
            ffcl::common::math::statistics::compute_variance_per_feature(data.begin(), data.end(), this->n_features_);
        const auto argmax1 = ffcl::common::math::statistics::argmax(var_per_feat.begin(), var_per_feat.end());

        const auto argmax2 =
            ffcl::common::math::statistics::argmax_variance_per_feature(data.begin(), data.end(), this->n_features_);

        EXPECT_EQ(argmax1, argmax2);
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}