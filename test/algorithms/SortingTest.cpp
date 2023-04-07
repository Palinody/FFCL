#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"

#include "gtest/gtest.h"

#include <algorithm>
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
                                                           std::uniform_real_distribution<DataType> >;

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

    static constexpr std::size_t n_samples_      = 5;
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
                    auto comparator = [feature_index](const auto& left_range_first, const auto& right_range_first) {
                        // assumes that:
                        //   * both ranges have length: n_features
                        //   * feature_index in range [0, n_features)
                        return *(left_range_first + feature_index) < *(right_range_first + feature_index);
                    };

                    const auto index =
                        ffcl::algorithms::median_index_of_three_ranges(data.begin(), data.end(), features, comparator);

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
                    auto comparator = [feature_index](const auto& left_range_first, const auto& right_range_first) {
                        // assumes that:
                        //   * both ranges have length: n_features
                        //   * feature_index in range [0, n_features)
                        return *(left_range_first + feature_index) < *(right_range_first + feature_index);
                    };

                    const auto index =
                        ffcl::algorithms::median_index_of_three_ranges(data.begin(), data.end(), features, comparator);

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
                    auto comparator = [feature_index](const auto& left_range_first, const auto& right_range_first) {
                        // assumes that:
                        //   * both ranges have length: n_features
                        //   * feature_index in range [0, n_features)
                        return *(left_range_first + feature_index) < *(right_range_first + feature_index);
                    };

                    const auto index = ffcl::algorithms::median_index_of_three_indexed_ranges(
                        indices.begin(), indices.end(), data.begin(), data.end(), features, comparator);

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
                    auto comparator = [feature_index](const auto& left_range_first, const auto& right_range_first) {
                        // assumes that:
                        //   * both ranges have length: n_features
                        //   * feature_index in range [0, n_features)
                        return *(left_range_first + feature_index) < *(right_range_first + feature_index);
                    };

                    const auto index = ffcl::algorithms::median_index_of_three_indexed_ranges(
                        indices.begin(), indices.end(), data.begin(), data.end(), features, comparator);

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
                    auto comparator = [feature_index](const auto& left_range_first, const auto& right_range_first) {
                        // assumes that:
                        //   * both ranges have length: n_features
                        //   * feature_index in range [0, n_features)
                        return *(left_range_first + feature_index) < *(right_range_first + feature_index);
                    };

                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_ranges(
                        data.begin(), data.end(), features, comparator);

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
                    auto comparator = [feature_index](const auto& left_range_first, const auto& right_range_first) {
                        // assumes that:
                        //   * both ranges have length: n_features
                        //   * feature_index in range [0, n_features)
                        return *(left_range_first + feature_index) < *(right_range_first + feature_index);
                    };

                    const auto [row_first, row_last] = ffcl::algorithms::median_values_range_of_three_ranges(
                        data.begin(), data.end(), features, comparator);

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

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
