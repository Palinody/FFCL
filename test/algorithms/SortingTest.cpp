#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include <iostream>
#include <random>

template <typename DataType>
class SortingTestFixture : public ::testing::Test {
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
        min_n_samples_  = 1;
        max_n_samples_  = 10;
        n_features_     = 3;
        n_random_tests_ = 3;
    }

  protected:
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

    std::vector<std::size_t> generate_indices(std::size_t n_samples) {
        std::vector<std::size_t> elements(n_samples);
        std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
        return elements;
    }

    std::vector<DataType> generate_ascending_elements_array(std::size_t n_samples) {
        std::vector<DataType> elements(n_samples);
        std::iota(elements.begin(), elements.end(), static_cast<DataType>(0));
        return elements;
    }

    std::vector<DataType> generate_descending_elements_array(std::size_t n_samples) {
        std::vector<DataType> elements(n_samples);
        std::iota(elements.rbegin(), elements.rend(), static_cast<DataType>(0));
        return elements;
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

    template <typename IteratorType>
    std::vector<typename IteratorType::value_type> shuffle_by_row(IteratorType element_first,
                                                                  IteratorType element_last,
                                                                  std::size_t  n_features) {
        const std::size_t n_samples = common::utils::get_n_samples(element_first, element_last, n_features);

        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), static_cast<DataType>(0));

        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        return this->remap_dataset(indices.begin(), indices.end(), element_first, element_last, n_features);
    }

    template <typename RandomAccessIntIterator, typename RandomAccessIterator>
    std::vector<typename RandomAccessIterator::value_type> remap_dataset(RandomAccessIntIterator index_first,
                                                                         RandomAccessIntIterator index_last,
                                                                         RandomAccessIterator    first,
                                                                         RandomAccessIterator    last,
                                                                         std::size_t             n_features) {
        const auto n_samples = common::utils::get_n_samples(first, last, n_features);

        assert(static_cast<std::ptrdiff_t>(n_samples) == std::distance(index_first, index_last));

        common::utils::ignore_parameters(index_last);

        auto remapped_flattened_vector = std::vector<typename RandomAccessIterator::value_type>(n_samples * n_features);

        for (std::size_t index = 0; index < n_samples; ++index) {
            std::copy(first + index_first[index] * n_features,
                      first + index_first[index] * n_features + n_features,
                      remapped_flattened_vector.begin() + index * n_features);
        }
        return remapped_flattened_vector;
    }

    void print_data(const std::vector<DataType>& data, std::size_t n_features) {
        const std::size_t n_samples = data.size() / n_features;

        for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
            for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
                std::cout << data[sample_index * n_features + feature_index] << " ";
            }
            std::cout << "\n";
        }
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
                            this->is_pivot_valid(data.begin(), data.end(), features, new_pivot_index, feature_index);

                        // print only if this->is_pivot_valid returned values (meaning that its not valid)
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
                        // the pivot is not valid if it disnt return std::nullopt
                        ASSERT_TRUE(!res.has_value());
                    }
                }
            }
        }
    }
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
                        const auto res = this->is_pivot_valid(data_indices.begin(),
                                                              data_indices.end(),
                                                              data.begin(),
                                                              data.end(),
                                                              features,
                                                              new_pivot_index,
                                                              feature_index);

                        // print only if this->is_pivot_valid returned values (meaning that its not valid)
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
                        // the pivot is not valid if it disnt return std::nullopt
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

                        const auto [kth_smallest_begin, kth_smallest_end] =
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
                            kth_smallest_begin,
                            kth_smallest_end,
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
                        shuffled_ascending_elements_array =
                            this->remap_dataset(data_indices.begin(),
                                                data_indices.end(),
                                                shuffled_ascending_elements_array.begin(),
                                                shuffled_ascending_elements_array.end(),
                                                features);

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
