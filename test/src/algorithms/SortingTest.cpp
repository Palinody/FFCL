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
class SortingTestFixture : public Range2DBaseFixture<DataType> {
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
TYPED_TEST_SUITE(SortingTestFixture, DataTypes);

TYPED_TEST(SortingTestFixture, MedianIndexOfThreeRangesTest) {
    // index has to be either 0, median or last index

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto index = ffcl::algorithms::median_index_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, MedianIndexOfThreeIndexedRangesTest) {
    // index has to be either 0, median or last index

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                const auto data_indices = this->generate_indices(samples);

                // test on all the possible feature data_indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto index = ffcl::algorithms::median_index_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, MedianValuesRangeOfThreeRangesTest) {
    // range values should be equal to one of the ranges at row 0, median or last row

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(
                        this->ranges_equality(
                            row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        this->ranges_equality(
                            row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        this->ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, MedianValuesRangeOfThreeIndexedRangesTest) {
    // range values should be equal to one of the ranges at row 0, median or last row

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);
                const auto data_indices = this->generate_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(
                        this->ranges_equality(
                            row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        this->ranges_equality(
                            row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        this->ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, MedianIndexAndValuesRangeOfThreeRangesTest) {
    // range values should be equal to one of the ranges at row 0, median or last row

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [index, range] = ffcl::algorithms::median_index_and_values_range_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    const auto [row_first, row_last] = range;

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);

                    ASSERT_TRUE(
                        this->ranges_equality(
                            row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        this->ranges_equality(
                            row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        this->ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, MedianIndexAndValuesRangeOfThreeIndexedRangesTest) {
    // range values should be equal to one of the ranges at row 0, median or last row

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                const auto data_indices = this->generate_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [index, range] = ffcl::algorithms::median_index_and_values_range_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [row_first, row_last] = range;

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        this->get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);

                    ASSERT_TRUE(
                        this->ranges_equality(
                            row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        this->ranges_equality(
                            row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        this->ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, PartitionAroundNTHRangeTest) {
    // range values should be equal to one of the ranges at row 0, median or last row

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    // check on all the possible pivot indices
                    for (std::size_t pivot_index = 0; pivot_index < samples; ++pivot_index) {
                        const auto new_pivot_index = ffcl::algorithms::partition_around_nth_range(
                            data.begin(), data.end(), features, pivot_index, feature_index);

                        // the values before the pivot according to the feature_index dimension should be less
                        // the values after the pivot according to the feature_index dimension should be greater or
                        // equal
                        const auto res =
                            this->is_pivot_faulty(data.begin(), data.end(), features, new_pivot_index, feature_index);

                        // print only if this->is_pivot_faulty returned values (meaning that its not valid)
                        if (res.has_value()) {
                            printf("n_samples: %ld, n_features: %ld\n", samples, features);
                            printf("pivot_index: %ld, feature_index: %ld\n", pivot_index, feature_index);

                            const auto pivot_value = data[new_pivot_index * features + feature_index];

                            const auto [not_pivot_index, not_pivot_value] = res.value();
                            if (not_pivot_index < new_pivot_index) {
                                std::cout << "Error, expected: not_pivot[" << not_pivot_index << "] < "
                                          << "pivot[" << new_pivot_index << "] but got: " << not_pivot_value << " < "
                                          << pivot_value << ", which is wrong.\n";

                            } else {
                                std::cout << "Error, expected: not_pivot[" << not_pivot_index << "] >= "
                                          << "pivot[" << new_pivot_index << "] but got: " << not_pivot_value
                                          << " >= " << pivot_value << ", which is wrong.\n";
                            }
                            printf("\n");
                            this->print_data(data, features);
                        }
                        // the pivot is not valid if it didn't return std::nullopt
                        ASSERT_TRUE(!res.has_value());
                    }
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, PartitionAroundNTHRangeWithDuplicatesTest) {
    using DataType = TypeParam;

    static constexpr std::size_t samples  = 9;
    static constexpr std::size_t features = 1;

    static constexpr std::size_t pivot_index   = 4;
    static constexpr std::size_t feature_index = 0;

    std::vector<DataType> data = {1, 1, 1, 1, 1, 1, 0, 3, 0};

    const auto [new_pivot_index_first, new_pivot_index_second] = ffcl::algorithms::three_way_partition_around_nth_range(
        data.begin(), data.end(), features, pivot_index, feature_index);

    // the values before the pivot according to the feature_index dimension should be less
    // the values after the pivot according to the feature_index dimension should be greater or
    // equal
    const auto res = this->is_pivot_faulty(data.begin(), data.end(), features, new_pivot_index_first, feature_index);

    // print only if this->is_pivot_faulty returned values (meaning that its not valid)
    if (res.has_value()) {
        printf("n_samples: %ld, n_features: %ld\n", samples, features);
        printf("pivot_index: %ld, feature_index: %ld\n", pivot_index, feature_index);

        const auto pivot_value = data[new_pivot_index_first * features + feature_index];

        const auto [not_pivot_index, not_pivot_value] = res.value();
        if (not_pivot_index < new_pivot_index_first) {
            std::cout << "Error, expected: not_pivot[" << not_pivot_index << "] < "
                      << "pivot[" << new_pivot_index_first << "] but got: " << not_pivot_value << " < " << pivot_value
                      << ", which is wrong.\n";

        } else {
            std::cout << "Error, expected: not_pivot[" << not_pivot_index << "] >= "
                      << "pivot[" << new_pivot_index_first << "] but got: " << not_pivot_value << " >= " << pivot_value
                      << ", which is wrong.\n";
        }
        printf("\n");
        this->print_data(data, features);
    }
    // the pivot is not valid if it didn't return std::nullopt
    ASSERT_TRUE(!res.has_value());
}

TYPED_TEST(SortingTestFixture, PartitionAroundNTHIndexedRangeTest) {
    // range values should be equal to one of the ranges at row 0, median or last row

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                auto data_indices = this->generate_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    // check on all the possible pivot indices
                    for (std::size_t pivot_index = 0; pivot_index < samples; ++pivot_index) {
                        const auto new_pivot_index =
                            ffcl::algorithms::partition_around_nth_indexed_range(data_indices.begin(),
                                                                                 data_indices.end(),
                                                                                 data.begin(),
                                                                                 data.end(),
                                                                                 features,
                                                                                 pivot_index,
                                                                                 feature_index);

                        // the values before the pivot according to the feature_index dimension should be less
                        // the values after the pivot according to the feature_index dimension should be greater or
                        // equal
                        const auto res = this->is_pivot_faulty(data_indices.begin(),
                                                               data_indices.end(),
                                                               data.begin(),
                                                               data.end(),
                                                               features,
                                                               new_pivot_index,
                                                               feature_index);

                        // print only if this->is_pivot_faulty returned values (meaning that its not valid)
                        if (res.has_value()) {
                            printf("n_samples: %ld, n_features: %ld\n", samples, features);
                            printf("pivot_index: %ld, feature_index: %ld\n", pivot_index, feature_index);

                            const auto pivot_value = data[new_pivot_index * features + feature_index];

                            const auto [not_pivot_index, not_pivot_value] = res.value();
                            if (not_pivot_index < new_pivot_index) {
                                std::cout << "Error, expected: not_pivot[" << not_pivot_index << "] < "
                                          << "pivot[" << new_pivot_index << "] but got: " << not_pivot_value << " < "
                                          << pivot_value << ", which is wrong.\n";

                            } else {
                                std::cout << "Error, expected: not_pivot[" << not_pivot_index << "] >= "
                                          << "pivot[" << new_pivot_index << "] but got: " << not_pivot_value
                                          << " >= " << pivot_value << ", which is wrong.\n";
                            }
                            printf("\n");
                            this->print_data(data, features);
                        }
                        // the pivot is not valid if it didn't return std::nullopt
                        ASSERT_TRUE(!res.has_value());
                    }
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, QuickselectRangeTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                // tests on all the nth smallest elements at the target feature_index
                for (std::size_t kth_smallest_index = 0; kth_smallest_index < samples; ++kth_smallest_index) {
                    // test on all the possible feature indices
                    for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                        auto ascending_elements_array = this->generate_ascending_elements_array(samples * features);

                        auto shuffled_ascending_elements_array = this->shuffle_by_row(
                            ascending_elements_array.begin(), ascending_elements_array.end(), features);

                        const auto [kth_smallest_begin, kth_smallest_end] =
                            ffcl::algorithms::quickselect_range(shuffled_ascending_elements_array.begin(),
                                                                shuffled_ascending_elements_array.end(),
                                                                features,
                                                                kth_smallest_index,
                                                                feature_index);

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
}

TYPED_TEST(SortingTestFixture, QuickselectIndexedRangeTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                // tests on all the nth smallest elements at the target feature_index
                for (std::size_t kth_smallest_index = 0; kth_smallest_index < samples; ++kth_smallest_index) {
                    // test on all the possible feature indices
                    for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                        auto ascending_elements_array = this->generate_ascending_elements_array(samples * features);

                        auto data_indices = this->generate_indices(samples);

                        auto shuffled_ascending_elements_array = this->shuffle_by_row(
                            ascending_elements_array.begin(), ascending_elements_array.end(), features);

                        const auto [kth_smallest_index_begin, kth_smallest_index_end] =
                            ffcl::algorithms::quickselect_indexed_range(data_indices.begin(),
                                                                        data_indices.end(),
                                                                        shuffled_ascending_elements_array.begin(),
                                                                        shuffled_ascending_elements_array.end(),
                                                                        features,
                                                                        kth_smallest_index,
                                                                        feature_index);

                        // the range returned from quickselect should be the same as in the sorted dataset at index
                        // kth_smallest_index
                        ASSERT_TRUE(this->ranges_equality(
                            shuffled_ascending_elements_array.begin() + *kth_smallest_index_begin * features,
                            shuffled_ascending_elements_array.begin() + *kth_smallest_index_end * features + features,
                            ascending_elements_array.begin() + kth_smallest_index * features,
                            ascending_elements_array.begin() + kth_smallest_index * features + features));

                        // also check that the indices at kth_smallest_index are mapping to the correct range in the
                        // original dataset
                        ASSERT_TRUE(this->ranges_equality(
                            shuffled_ascending_elements_array.begin() + data_indices[kth_smallest_index] * features,
                            shuffled_ascending_elements_array.begin() + data_indices[kth_smallest_index] * features +
                                features,
                            ascending_elements_array.begin() + kth_smallest_index * features,
                            ascending_elements_array.begin() + kth_smallest_index * features + features));
                    }
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, QuicksortRangeTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                // tests on all the nth smallest elements at the target feature_index
                for (std::size_t pivot_index = 0; pivot_index < samples; ++pivot_index) {
                    // test on all the possible feature indices
                    for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                        auto ascending_elements_array = this->generate_ascending_elements_array(samples * features);

                        auto shuffled_ascending_elements_array = this->shuffle_by_row(
                            ascending_elements_array.begin(), ascending_elements_array.end(), features);

                        // new_pivot_index indicates where the original pivot has moved after quicksort
                        const auto new_pivot_index =
                            ffcl::algorithms::quicksort_range(shuffled_ascending_elements_array.begin(),
                                                              shuffled_ascending_elements_array.end(),
                                                              features,
                                                              pivot_index,
                                                              feature_index);

                        // the data sorted by quickselect should now be the same as the original dataset
                        // shuffled_ascending_elements_array has been sorted inplace by quicksort
                        ASSERT_TRUE(this->ranges_equality(shuffled_ascending_elements_array.begin(),
                                                          shuffled_ascending_elements_array.end(),
                                                          ascending_elements_array.begin(),
                                                          ascending_elements_array.end()));

                        // the data at new_pivot_index in the shuffled dataset sorted by quickselect should be equal to
                        // the original dataset
                        ASSERT_TRUE(this->ranges_equality(
                            shuffled_ascending_elements_array.begin() + new_pivot_index * features,
                            shuffled_ascending_elements_array.begin() + new_pivot_index * features + features,
                            ascending_elements_array.begin() + new_pivot_index * features,
                            ascending_elements_array.begin() + new_pivot_index * features + features));
                    }
                }
            }
        }
    }
}

TYPED_TEST(SortingTestFixture, QuicksortIndexedRangeTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                // tests on all the nth smallest elements at the target feature_index
                for (std::size_t pivot_index = 0; pivot_index < samples; ++pivot_index) {
                    // test on all the possible feature indices
                    for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                        auto ascending_elements_array = this->generate_ascending_elements_array(samples * features);

                        auto data_indices = this->generate_indices(samples);

                        auto shuffled_ascending_elements_array = this->shuffle_by_row(
                            ascending_elements_array.begin(), ascending_elements_array.end(), features);

                        // new_pivot_index indicates where the original pivot has moved after quicksort
                        const auto new_pivot_index =
                            ffcl::algorithms::quicksort_indexed_range(data_indices.begin(),
                                                                      data_indices.end(),
                                                                      shuffled_ascending_elements_array.begin(),
                                                                      shuffled_ascending_elements_array.end(),
                                                                      features,
                                                                      pivot_index,
                                                                      feature_index);

                        // sort the shuffled_ascending_elements_array with the index vector remapped inplace by
                        // quicksort_indexed_range
                        shuffled_ascending_elements_array = common::utils::remap_ranges_from_indices(
                            data_indices, shuffled_ascending_elements_array, features);

                        // the data sorted by quickselect should now be the same as the original dataset
                        // shuffled_ascending_elements_array has been sorted inplace by quicksort
                        ASSERT_TRUE(this->ranges_equality(shuffled_ascending_elements_array.begin(),
                                                          shuffled_ascending_elements_array.end(),
                                                          ascending_elements_array.begin(),
                                                          ascending_elements_array.end()));

                        // the data at new_pivot_index in the shuffled dataset sorted by quickselect should be equal to
                        // the original dataset
                        ASSERT_TRUE(this->ranges_equality(
                            shuffled_ascending_elements_array.begin() + new_pivot_index * features,
                            shuffled_ascending_elements_array.begin() + new_pivot_index * features + features,
                            ascending_elements_array.begin() + new_pivot_index * features,
                            ascending_elements_array.begin() + new_pivot_index * features + features));
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
