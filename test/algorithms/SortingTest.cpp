#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <optional>
#include <vector>

#include <iostream>
#include <random>

class SortingTestFixture : public ::testing::Test {
  protected:
    template <typename DataType>
    std::vector<DataType> generate_random_uniform_vector(std::size_t n_rows,
                                                         std::size_t n_cols,
                                                         DataType    lower_bound,
                                                         DataType    upper_bound) {
        using UniformDistributionType = std::conditional_t<std::is_integral_v<DataType>,
                                                           std::uniform_int_distribution<DataType>,
                                                           std::uniform_real_distribution<DataType>>;

        std::random_device rnd_device;
        std::mt19937       mersenne_engine{rnd_device()};

        UniformDistributionType dist{lower_bound, upper_bound};
        auto                    gen = [&dist, &mersenne_engine]() { return dist(mersenne_engine); };
        std::vector<DataType>   result(n_rows * n_cols);
        std::generate(result.begin(), result.end(), gen);
        return result;
    }

    std::vector<std::size_t> generate_sorted_indices(std::size_t n_samples) {
        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    template <typename IteratorType>
    std::pair<IteratorType, IteratorType> get_range_at_row(IteratorType element_first,
                                                           IteratorType element_last,
                                                           std::size_t  n_features,
                                                           std::size_t  target_row) {
        common::utils::ignore_parameters(element_last);

        return {element_first + target_row * n_features, element_first + target_row * n_features + n_features};
    }

    template <typename IteratorType>
    bool ranges_equality(IteratorType element_first,
                         IteratorType element_last,
                         IteratorType other_element_first,
                         IteratorType other_element_last) {
        // If the sizes of the ranges are not equal, they are not equal
        if (std::distance(element_first, element_last) != std::distance(other_element_first, other_element_last)) {
            return false;
        }
        // Iterate over both ranges and compare their elements
        auto it1 = element_first;
        auto it2 = other_element_first;
        while (it1 != element_last && it2 != other_element_last) {
            if (common::utils::inequality(*it1, *it2)) {
                return false;
            }
            ++it1;
            ++it2;
        }
        // If we iterated over all elements in both ranges, they are equal
        return (it1 == element_last && it2 == other_element_last);
    }

    template <typename IteratorType>
    std::optional<std::pair<std::size_t, typename IteratorType::value_type>> is_pivot_valid(IteratorType element_first,
                                                                                            IteratorType element_last,
                                                                                            std::size_t  n_features,
                                                                                            std::size_t  pivot_index,
                                                                                            std::size_t feature_index) {
        const std::size_t n_samples = common::utils::get_n_samples(element_first, element_last, n_features);

        if (pivot_index == 0 && n_samples == 1) {
            return std::nullopt;
        }
        const auto pivot_value = element_first[pivot_index * n_features + feature_index];
        // iterate over the elements before the pivot
        for (std::size_t before_pivot_index = 0; before_pivot_index < pivot_index; ++before_pivot_index) {
            const auto before_pivot_value = element_first[before_pivot_index * n_features + feature_index];
            // the values before the pivot at target_feature_index should be strictly less than the pivot
            if (before_pivot_value >= pivot_value) {
                return std::make_optional(std::make_pair(before_pivot_index, before_pivot_value));
            }
        }
        // iterate over the elements after the pivot
        for (std::size_t after_pivot_index = pivot_index + 1; after_pivot_index < n_samples; ++after_pivot_index) {
            const auto after_pivot_value = element_first[after_pivot_index * n_features + feature_index];
            // the values after the pivot at feature_index should be greater than or equal to the pivot
            if (after_pivot_value < pivot_value) {
                return std::make_optional(std::make_pair(after_pivot_index, after_pivot_value));
            }
        }
        return std::nullopt;
    }

    template <typename IteratorIntType, typename IteratorType>
    std::optional<std::pair<std::size_t, typename IteratorType::value_type>> is_pivot_valid(IteratorIntType index_first,
                                                                                            IteratorIntType index_last,
                                                                                            IteratorType element_first,
                                                                                            IteratorType element_last,
                                                                                            std::size_t  n_features,
                                                                                            std::size_t  pivot_index,
                                                                                            std::size_t feature_index) {
        common::utils::ignore_parameters(element_last);

        const std::size_t n_samples = std::distance(index_first, index_last);

        if (pivot_index == 0 && n_samples == 1) {
            return std::nullopt;
        }
        const auto pivot_value = element_first[index_first[pivot_index] * n_features + feature_index];
        // iterate over the elements before the pivot
        for (std::size_t before_pivot_index = 0; before_pivot_index < pivot_index; ++before_pivot_index) {
            const auto before_pivot_value = element_first[index_first[before_pivot_index] * n_features + feature_index];
            // the values before the pivot at target_feature_index should be strictly less than the pivot
            if (before_pivot_value >= pivot_value) {
                return std::make_optional(std::make_pair(index_first[before_pivot_index], before_pivot_value));
            }
        }
        // iterate over the elements after the pivot
        for (std::size_t after_pivot_index = pivot_index + 1; after_pivot_index < n_samples; ++after_pivot_index) {
            const auto after_pivot_value = element_first[index_first[after_pivot_index] * n_features + feature_index];
            // the values after the pivot at feature_index should be greater than or equal to the pivot
            if (after_pivot_value < pivot_value) {
                return std::make_optional(std::make_pair(index_first[after_pivot_index], after_pivot_value));
            }
        }
        return std::nullopt;
    }

    static constexpr std::size_t n_samples_      = 10;
    static constexpr std::size_t n_features_     = 3;
    static constexpr std::size_t n_random_tests_ = 10;
};

TEST_F(SortingTestFixture, MedianIndexOfThreeRangesIntegerTest) {
    // index has to be either 0, median or last index
    using DataType = int;

    constexpr DataType lower_bound = -10;
    constexpr DataType upper_bound = 10;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

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

TEST_F(SortingTestFixture, MedianIndexOfThreeRangesFloatTest) {
    // index has to be either 0, median or last index
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

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

TEST_F(SortingTestFixture, MedianIndexOfThreeIndexedRangesIntegerTest) {
    // index has to be either 0, median or last index
    using DataType = int;

    constexpr DataType lower_bound = -10;
    constexpr DataType upper_bound = 10;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto indices = generate_sorted_indices(samples);
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto index = ffcl::algorithms::median_index_of_three_indexed_ranges(
                        indices.begin(), indices.end(), data.begin(), data.end(), features, feature_index);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianIndexOfThreeIndexedRangesFloatTest) {
    // index has to be either 0, median or last index
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto indices = generate_sorted_indices(samples);
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto index = ffcl::algorithms::median_index_of_three_indexed_ranges(
                        indices.begin(), indices.end(), data.begin(), data.end(), features, feature_index);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianValuesRangeOfThreeRangesIntegerTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = int;

    constexpr DataType lower_bound = -10;
    constexpr DataType upper_bound = 10;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianValuesRangeOfThreeRangesFloatTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianValuesRangeOfThreeIndexedRangesIntegerTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = int;

    constexpr DataType lower_bound = -10;
    constexpr DataType upper_bound = 10;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);
                const auto data_indices = generate_sorted_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianValuesRangeOfThreeIndexedRangesFloatTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);
                const auto data_indices = generate_sorted_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianIndexAndValuesRangeOfThreeRangesIntegerTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = int;

    constexpr DataType lower_bound = -10;
    constexpr DataType upper_bound = 10;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [index, range] = ffcl::algorithms::median_index_and_values_range_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    const auto [row_first, row_last] = range;

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianIndexAndValuesRangeOfThreeRangesFloatTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [index, range] = ffcl::algorithms::median_index_and_values_range_of_three_ranges(
                        data.begin(), data.end(), features, feature_index);

                    const auto [row_first, row_last] = range;

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianIndexAndValuesRangeOfThreeIndexedRangesIntegerTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = int;

    constexpr DataType lower_bound = -10;
    constexpr DataType upper_bound = 10;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);
                const auto data_indices = generate_sorted_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [index, range] = ffcl::algorithms::median_index_and_values_range_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [row_first, row_last] = range;

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, MedianIndexAndValuesRangeOfThreeIndexedRangesFloatTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                const auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);
                const auto data_indices = generate_sorted_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [index, range] = ffcl::algorithms::median_index_and_values_range_of_three_indexed_ranges(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [row_first, row_last] = range;

                    const auto [first_row_candidate_first, first_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, 0);

                    const auto [median_row_candidate_first, median_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples / 2);

                    const auto [last_row_candidate_first, last_row_candidate_last] =
                        get_range_at_row(data.begin(), data.end(), features, samples - 1);

                    ASSERT_TRUE(index == 0 || index == samples / 2 || index == samples - 1);

                    ASSERT_TRUE(
                        ranges_equality(row_first, row_last, first_row_candidate_first, first_row_candidate_last) ||
                        ranges_equality(row_first, row_last, median_row_candidate_first, median_row_candidate_last) ||
                        ranges_equality(row_first, row_last, last_row_candidate_first, last_row_candidate_last));
                }
            }
        }
    }
}

#include <iostream>

template <typename DataType>
void print_data(const std::vector<DataType>& data, std::size_t n_features) {
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

TEST_F(SortingTestFixture, PartitionAroundNTHRangeIntegerTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = int;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

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
                            is_pivot_valid(data.begin(), data.end(), features, new_pivot_index, feature_index);

                        // print only if is_pivot_valid returned values (meaning that its not valid)
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
                            print_data(data, features);
                        }
                        // the pivot is not valid if it disnt return std::nullopt
                        ASSERT_TRUE(!res.has_value());
                    }
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, PartitionAroundNTHRangeFloatTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);

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
                            is_pivot_valid(data.begin(), data.end(), features, new_pivot_index, feature_index);

                        // print only if is_pivot_valid returned values (meaning that its not valid)
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
                            print_data(data, features);
                        }
                        // the pivot is not valid if it disnt return std::nullopt
                        ASSERT_TRUE(!res.has_value());
                    }
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, PartitionAroundNTHIndexedRangeIntegerTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = int;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);
                auto data_indices = generate_sorted_indices(samples);

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
                        const auto res = is_pivot_valid(data_indices.begin(),
                                                        data_indices.end(),
                                                        data.begin(),
                                                        data.end(),
                                                        features,
                                                        new_pivot_index,
                                                        feature_index);

                        // print only if is_pivot_valid returned values (meaning that its not valid)
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
                            print_data(data, features);
                        }
                        // the pivot is not valid if it disnt return std::nullopt
                        ASSERT_TRUE(!res.has_value());
                    }
                }
            }
        }
    }
}

TEST_F(SortingTestFixture, PartitionAroundNTHIndexedRangeFloatTest) {
    // range values should be equal to one of the ranges at row 0, median or last row
    using DataType = float;

    constexpr DataType lower_bound = -1;
    constexpr DataType upper_bound = 1;

    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index <= n_random_tests_; ++test_index) {
        // tests on data from 1 to n_samples_ samples
        for (std::size_t samples = 1; samples <= n_samples_; ++samples) {
            // tests on data from 1 to n_features_ features
            for (std::size_t features = 1; features <= n_features_; ++features) {
                auto data = generate_random_uniform_vector<DataType>(samples, features, lower_bound, upper_bound);
                auto data_indices = generate_sorted_indices(samples);

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
                        const auto res = is_pivot_valid(data_indices.begin(),
                                                        data_indices.end(),
                                                        data.begin(),
                                                        data.end(),
                                                        features,
                                                        new_pivot_index,
                                                        feature_index);

                        // print only if is_pivot_valid returned values (meaning that its not valid)
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
                            print_data(data, features);
                        }
                        // the pivot is not valid if it disnt return std::nullopt
                        ASSERT_TRUE(!res.has_value());
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
