#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Range2DBaseFixture.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/linear_algebra/Transpose.hpp"

#include <vector>

template <typename DataType>
class TransposeTestFixture : public Range2DBaseFixture<DataType> {
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

        n_samples_  = 20;
        n_features_ = 7;
        n_tests_    = 10;
    }

  protected:
    DataType    lower_bound_;
    DataType    upper_bound_;
    std::size_t n_samples_;
    std::size_t n_features_;
    std::size_t n_tests_;
};

using DataTypes = ::testing::Types<int, std::size_t, float, double>;
TYPED_TEST_SUITE(TransposeTestFixture, DataTypes);

TYPED_TEST(TransposeTestFixture, TransposeSerialTest) {
    for (std::size_t test_index = 0; test_index < this->n_tests_; ++test_index) {
        const auto data = this->generate_random_uniform_vector(
            this->n_samples_, this->n_features_, this->lower_bound_, this->upper_bound_);

        const auto [transposed_data, n_samples_transposed, n_features_transposed] =
            math::linear_algebra::transpose(data.begin(), data.end(), this->n_features_);

        ASSERT_EQ(data.size(), transposed_data.size());

        // retrieve the original data
        const auto [orig_data, n_samples_orig, n_features_orig] =
            math::linear_algebra::transpose(transposed_data.begin(), transposed_data.end(), n_features_transposed);

        ASSERT_EQ(n_samples_orig, this->n_samples_);
        ASSERT_EQ(n_features_orig, this->n_features_);

        ASSERT_TRUE(this->ranges_equality(data.begin(), data.end(), orig_data.begin(), orig_data.end()));

#if defined(VERBOSE) && VERBOSE == true
        this->print_data(data, this->n_features_);
        printf("\n");
        this->print_data(transposed_data, n_features_transposed);
        printf("\n");
        this->print_data(orig_data, n_features_orig);
        printf("\n");
#endif
    }
}

TYPED_TEST(TransposeTestFixture, TransposeOpenMPTest) {
    for (std::size_t test_index = 0; test_index < this->n_tests_; ++test_index) {
        const auto data = this->generate_random_uniform_vector(
            this->n_samples_, this->n_features_, this->lower_bound_, this->upper_bound_);

        const auto [transposed_data, n_samples_transposed, n_features_transposed] =
            math::linear_algebra::transpose_parallel_openmp(data.begin(), data.end(), this->n_features_);

        ASSERT_EQ(data.size(), transposed_data.size());

        // retrieve the original data
        const auto [orig_data, n_samples_orig, n_features_orig] = math::linear_algebra::transpose_parallel_openmp(
            transposed_data.begin(), transposed_data.end(), n_features_transposed);

        ASSERT_EQ(n_samples_orig, this->n_samples_);
        ASSERT_EQ(n_features_orig, this->n_features_);

        ASSERT_TRUE(this->ranges_equality(data.begin(), data.end(), orig_data.begin(), orig_data.end()));

#if defined(VERBOSE) && VERBOSE == true
        this->print_data(data, this->n_features_);
        printf("\n");
        this->print_data(transposed_data, n_features_transposed);
        printf("\n");
        this->print_data(orig_data, n_features_orig);
        printf("\n");
#endif
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}