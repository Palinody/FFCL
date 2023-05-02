#include "ffcl/containers/kdtree/KDTreeUtils.hpp"
#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"

#include "Range2DBaseFixture.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include <iostream>
#include <random>

template <typename DataType>
class KDTreeUtilsTestFixture : public Range2DBaseFixture<DataType> {
  public:
    void SetUp() override {
        if constexpr (std::is_integral_v<DataType> && std::is_signed_v<DataType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;

        } else if constexpr (std::is_integral_v<DataType> && std::is_unsigned_v<DataType>) {
            lower_bound_ = 0;
            upper_bound_ = 1;

        } else if constexpr (std::is_floating_point_v<DataType>) {
            lower_bound_ = -1;
            upper_bound_ = 1;
        }
        min_n_samples_  = 1;
        max_n_samples_  = 10;
        n_features_     = 1;
        n_random_tests_ = 5;
    }

  protected:
    DataType    lower_bound_;
    DataType    upper_bound_;
    std::size_t min_n_samples_;
    std::size_t max_n_samples_;
    std::size_t n_features_;
    std::size_t n_random_tests_;
};

using DataTypes = ::testing::Types<int, std::size_t, float, double>;
TYPED_TEST_SUITE(KDTreeUtilsTestFixture, DataTypes);

TYPED_TEST(KDTreeUtilsTestFixture, QuickselectMedianRangeTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                // nth smallest elements is the median by default
                const auto kth_smallest_index = samples / 2;
                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    auto ascending_elements_array = this->generate_ascending_elements_array(samples * features);

                    auto shuffled_ascending_elements_array = this->shuffle_by_row(
                        ascending_elements_array.begin(), ascending_elements_array.end(), features);

                    const auto [cut_index, left_range, cut_range, right_range] =
                        kdtree::utils::quickselect_median_range(shuffled_ascending_elements_array.begin(),
                                                                shuffled_ascending_elements_array.end(),
                                                                features,
                                                                feature_index);

                    const auto [kth_smallest_begin, kth_smallest_end] = cut_range;

                    // the range returned from quickselect should be the same as in the sorted dataset at index
                    // kth_smallest_index
                    ASSERT_TRUE(this->ranges_equality(
                        kth_smallest_begin,
                        kth_smallest_end,
                        ascending_elements_array.begin() + kth_smallest_index * features,
                        ascending_elements_array.begin() + kth_smallest_index * features + features));

                    // also check that the range at kth_smallest_index in shuffled_ascending_elements_array is
                    // correct since the implementation is inplace
                    ASSERT_TRUE(this->ranges_equality(
                        shuffled_ascending_elements_array.begin() + kth_smallest_index * features,
                        shuffled_ascending_elements_array.begin() + kth_smallest_index * features + features,
                        ascending_elements_array.begin() + kth_smallest_index * features,
                        ascending_elements_array.begin() + kth_smallest_index * features + features));
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}