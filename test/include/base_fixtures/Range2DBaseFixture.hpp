#pragma once

#include "ffcl/common/Utils.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include <iostream>
#include <random>

template <typename DataType>
class Range2DBaseFixture : public ::testing::Test {
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
};