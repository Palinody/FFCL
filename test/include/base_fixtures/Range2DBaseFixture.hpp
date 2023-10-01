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

    template <typename SamplesIterator>
    std::pair<SamplesIterator, SamplesIterator> get_range_at_row(SamplesIterator samples_first,
                                                                 SamplesIterator samples_last,
                                                                 std::size_t     n_features,
                                                                 std::size_t     target_row) {
        common::utils::ignore_parameters(samples_last);

        return {samples_first + target_row * n_features, samples_first + target_row * n_features + n_features};
    }

    template <typename SamplesIterator>
    std::vector<typename SamplesIterator::value_type> get_column(SamplesIterator samples_first,
                                                                 SamplesIterator samples_last,
                                                                 std::size_t     n_features,
                                                                 std::size_t     target_column) {
        const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

        auto column = std::vector<typename SamplesIterator::value_type>(n_samples);

        for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
            column[sample_index] = samples_first[sample_index * n_features + target_column];
        }
        return column;
    }

    template <typename SamplesIterator>
    bool ranges_equality(SamplesIterator samples_first,
                         SamplesIterator samples_last,
                         SamplesIterator other_element_first,
                         SamplesIterator other_element_last) {
        // If the sizes of the ranges are not equal, they are not equal
        if (std::distance(samples_first, samples_last) != std::distance(other_element_first, other_element_last)) {
            return false;
        }
        return common::utils::are_containers_equal(samples_first, samples_last, other_element_first);
    }

    template <typename SamplesIterator>
    std::optional<std::pair<std::size_t, typename SamplesIterator::value_type>> is_pivot_faulty(
        SamplesIterator samples_first,
        SamplesIterator samples_last,
        std::size_t     n_features,
        std::size_t     pivot_index,
        std::size_t     feature_index) {
        const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

        if (pivot_index == 0 && n_samples == 1) {
            return std::nullopt;
        }
        const auto pivot_value = samples_first[pivot_index * n_features + feature_index];
        // iterate over the elements before the pivot
        for (std::size_t before_pivot_index = 0; before_pivot_index < pivot_index; ++before_pivot_index) {
            const auto before_pivot_value = samples_first[before_pivot_index * n_features + feature_index];
            // the values before the pivot at target_feature_index should be strictly less than the pivot
            if (before_pivot_value >= pivot_value) {
                return std::make_optional(std::make_pair(before_pivot_index, before_pivot_value));
            }
        }
        // iterate over the elements after the pivot
        for (std::size_t after_pivot_index = pivot_index + 1; after_pivot_index < n_samples; ++after_pivot_index) {
            const auto after_pivot_value = samples_first[after_pivot_index * n_features + feature_index];
            // the values after the pivot at feature_index should be greater than or equal to the pivot
            if (after_pivot_value < pivot_value) {
                return std::make_optional(std::make_pair(after_pivot_index, after_pivot_value));
            }
        }
        return std::nullopt;
    }

    template <typename IndicesIterator, typename SamplesIterator>
    std::optional<std::pair<std::size_t, typename SamplesIterator::value_type>> is_pivot_faulty(
        IndicesIterator indices_first,
        IndicesIterator indices_last,
        SamplesIterator samples_first,
        SamplesIterator samples_last,
        std::size_t     n_features,
        std::size_t     pivot_index,
        std::size_t     feature_index) {
        common::utils::ignore_parameters(samples_last);

        const std::size_t n_samples = std::distance(indices_first, indices_last);

        if (pivot_index == 0 && n_samples == 1) {
            return std::nullopt;
        }
        const auto pivot_value = samples_first[indices_first[pivot_index] * n_features + feature_index];
        // iterate over the elements before the pivot
        for (std::size_t before_pivot_index = 0; before_pivot_index < pivot_index; ++before_pivot_index) {
            const auto before_pivot_value =
                samples_first[indices_first[before_pivot_index] * n_features + feature_index];
            // the values before the pivot at target_feature_index should be strictly less than the pivot
            if (before_pivot_value >= pivot_value) {
                return std::make_optional(std::make_pair(indices_first[before_pivot_index], before_pivot_value));
            }
        }
        // iterate over the elements after the pivot
        for (std::size_t after_pivot_index = pivot_index + 1; after_pivot_index < n_samples; ++after_pivot_index) {
            const auto after_pivot_value = samples_first[indices_first[after_pivot_index] * n_features + feature_index];
            // the values after the pivot at feature_index should be greater than or equal to the pivot
            if (after_pivot_value < pivot_value) {
                return std::make_optional(std::make_pair(indices_first[after_pivot_index], after_pivot_value));
            }
        }
        return std::nullopt;
    }

    template <typename SamplesIterator>
    std::vector<typename SamplesIterator::value_type> shuffle_by_row(SamplesIterator samples_first,
                                                                     SamplesIterator samples_last,
                                                                     std::size_t     n_features) {
        const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), static_cast<DataType>(0));

        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        return common::utils::remap_ranges_from_indices(indices, std::vector(samples_first, samples_last), n_features);
    }

    template <typename Type = DataType>
    void print_data(const std::vector<Type>& data, std::size_t n_features) {
        const std::size_t n_samples = data.size() / n_features;

        for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
            for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
                std::cout << data[sample_index * n_features + feature_index] << " ";
            }
            std::cout << "\n";
        }
    }
};