#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Range2DBaseFixture.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

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

    auto result = math::statistics::compute_mean_per_feature(data.begin(), data.end(), n_features);

    ASSERT_EQ(result.size(), n_features);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], expected_mean[i]);
    }
}

TYPED_TEST(MathStatisticsTestFixture, ComputeVariancePerFeature) {
    using DataType = TypeParam;

    const std::vector<DataType>  data              = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<DataType>  expected_variance = {9, 9, 9};
    static constexpr std::size_t n_features        = 3;

    auto result = math::statistics::compute_variance_per_feature(data.begin(), data.end(), n_features);

    ASSERT_EQ(result.size(), n_features);

    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], expected_variance[i]);
    }
}

TYPED_TEST(MathStatisticsTestFixture, ComputeVarianceV2PerFeature) {
    for (std::size_t test_index = 0; test_index < this->n_tests_; ++test_index) {
        const auto data = this->generate_random_uniform_vector(
            this->n_samples_, this->n_features_, this->lower_bound_, this->upper_bound_);

        auto result    = math::statistics::compute_variance_per_feature(data.begin(), data.end(), this->n_features_);
        auto result_v2 = math::statistics::compute_variance_per_feature_v2(data.begin(), data.end(), this->n_features_);

        ASSERT_TRUE(result.size() == result_v2.size());

        for (std::size_t feature_index = 0; feature_index; ++feature_index) {
            EXPECT_DOUBLE_EQ(result[feature_index], result_v2[feature_index]);
        }
#if defined(VERBOSE) && VERBOSE == true
        this->print_data(result, this->n_features_);
        printf("\n");
        this->print_data(result_v2, this->n_features_);
        printf("\n");
#endif

        ASSERT_TRUE(this->ranges_equality(result.begin(), result.end(), result_v2.begin(), result_v2.end()));
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}