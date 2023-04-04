#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/algorithms/Sorting.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/math/random/Distributions.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

class UtilsErrorsTest : public ::testing::Test {
  public:
    using DataType = int;

  protected:
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

    template <typename T>
    void print_flattened_matrix(const std::vector<T>& container, std::size_t n_features) {
        const std::size_t n_samples = container.size() / n_features;

        for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
            for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
                std::cout << container[sample_index * n_features + feature_index] << " ";
            }
            std::cout << "\n";
        }
    }

    template <typename DataType>
    std::vector<DataType> generate_flattened_matrix(std::size_t n_samples,
                                                    std::size_t n_features,
                                                    DataType    lower_bound = 0,
                                                    DataType    upper_bound = 10) {
        math::random::uniform_distribution<DataType> random_uniform(lower_bound, upper_bound);

        auto result = std::vector<DataType>(n_samples * n_features);

        std::generate(result.begin(), result.end(), random_uniform);

        return result;
    }

    static constexpr std::size_t n_samples_  = 5;
    static constexpr std::size_t n_features_ = 3;
};

TEST_F(UtilsErrorsTest, NTHElementTest) {
    auto data = generate_flattened_matrix<DataType>(n_samples_, n_features_, -10, 10);
    // keep a save of the original since data will be modified inplace
    const auto original_data = data;

    print_flattened_matrix<DataType>(data, n_features_);

    for (std::size_t pivot_index = 0; pivot_index < n_samples_; ++pivot_index) {
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            const auto new_pivot_index = ffcl::algorithms::partition_around_nth_range(
                data.begin(),
                data.end(),
                pivot_index,
                n_features_,
                [feature_index](const auto& range1_first, const auto& range2_first) {
                    // assumes that:
                    //   * both ranges have length: n_features_
                    //   * feature_index in range [0, n_features_)
                    return *(range1_first + feature_index) < *(range2_first + feature_index);
                });
            std::cout << "Partial sort w.r.t. feature index: " << feature_index << "\n";
            std::cout << "pivot index remapped: " << pivot_index << " -> " << new_pivot_index << "\n";
            print_flattened_matrix<DataType>(data, n_features_);
            std::cout << "---\n";

            data = original_data;
        }
    }
}

TEST_F(UtilsErrorsTest, RandomDatasetPivotTest) {
    auto data = generate_flattened_matrix<DataType>(n_samples_, n_features_, -10, 10);
    // keep a save of the original since data will be modified inplace
    const auto original_data = data;

    print_flattened_matrix<DataType>(data, n_features_);

    for (std::size_t pivot_index = 0; pivot_index < n_samples_; ++pivot_index) {
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            const auto new_pivot_index = ffcl::algorithms::quicksort_range(
                data.begin(),
                data.end(),
                pivot_index,
                n_features_,
                [feature_index](const auto& range1_first, const auto& range2_first) {
                    // assumes that:
                    //   * both ranges have length: n_features_
                    //   * feature_index in range [0, n_features_)
                    return *(range1_first + feature_index) < *(range2_first + feature_index);
                });
            std::cout << "Sorted w.r.t. feature index: " << feature_index << "\n";
            std::cout << "pivot index remapped: " << pivot_index << " -> " << new_pivot_index << "\n";
            print_flattened_matrix<DataType>(data, n_features_);
            std::cout << "---\n";

            data = original_data;
        }
    }
}

TEST_F(UtilsErrorsTest, QuickselectTest) {
    auto data = generate_flattened_matrix<DataType>(n_samples_, n_features_, -10, 10);
    // keep a save of the original since data will be modified inplace
    const auto original_data = data;

    print_flattened_matrix<DataType>(data, n_features_);

    for (std::size_t kth_smallest_index = 0; kth_smallest_index < n_samples_; ++kth_smallest_index) {
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            const auto [kth_smallest_begin, kth_smallest_end] = ffcl::algorithms::quickselect_range(
                data.begin(),
                data.end(),
                kth_smallest_index,
                n_features_,
                [feature_index](const auto& range1_first, const auto& range2_first) {
                    // assumes that:
                    //   * both ranges have length: n_features_
                    //   * feature_index in range [0, n_features_)
                    return *(range1_first + feature_index) < *(range2_first + feature_index);
                });

            std::cout << "Partially sorted w.r.t. feature index: " << feature_index << "\n";
            print_flattened_matrix<DataType>(data, n_features_);

            printf("Kth smallest index quiery: %ld\n", kth_smallest_index);
            printf("Pivot range: ");
            print_flattened_matrix<DataType>(std::vector<DataType>(kth_smallest_begin, kth_smallest_end), n_features_);
            printf("\n");

            data = original_data;
        }
    }
}

TEST_F(UtilsErrorsTest, PartitionAroundNTHIndexedRangeTest) {
    auto indices = std::vector<std::size_t>(n_samples_);
    std::iota(indices.begin(), indices.end(), 0);
    auto data = generate_flattened_matrix<DataType>(n_samples_, n_features_, -10, 10);

    constexpr std::size_t feature_index = 1;

    constexpr std::size_t pivot_index = n_samples_ / 2;

    static_assert(feature_index < n_features_);

    print_flattened_matrix(indices, 1);
    print_flattened_matrix<DataType>(data, n_features_);

    const auto new_pivot_index = ffcl::algorithms::partition_around_nth_indexed_range(
        indices.begin(),
        indices.end(),
        data.begin(),
        data.end(),
        pivot_index,
        n_features_,
        [feature_index](const auto& range1_first, const auto& range2_first) {
            // assumes that:
            //   * both ranges have length: n_features_
            //   * feature_index in range [0, n_features_)
            return range1_first[feature_index] < range2_first[feature_index];
        });

    const auto remapped_flattened_vector =
        remap_dataset(indices.begin(), indices.end(), data.begin(), data.end(), n_features_);

    printf("Sorted indices:\n");
    print_flattened_matrix(indices, 1);
    printf("Sorted dataset:\n");
    print_flattened_matrix<DataType>(remapped_flattened_vector, n_features_);
    printf("Pivot index: %ld\n", pivot_index);
    printf("New pivot index: %ld\n", new_pivot_index);
}

TEST_F(UtilsErrorsTest, QuickselectIndexedRangeTest) {
    auto indices = std::vector<std::size_t>(n_samples_);
    std::iota(indices.begin(), indices.end(), 0);
    auto data = generate_flattened_matrix<DataType>(n_samples_, n_features_, -10, 10);

    // keep a save of the original since data will be modified inplace
    const auto original_data = data;

    print_flattened_matrix<DataType>(data, n_features_);

    for (std::size_t kth_smallest_index = 0; kth_smallest_index < n_samples_; ++kth_smallest_index) {
        for (std::size_t feature_index = 0; feature_index < n_features_; ++feature_index) {
            const auto [kth_smallest_begin, kth_smallest_end] = ffcl::algorithms::quickselect_indexed_range(
                indices.begin(),
                indices.end(),
                data.begin(),
                data.end(),
                kth_smallest_index,
                n_features_,
                [feature_index](const auto& range1_first, const auto& range2_first) {
                    // assumes that:
                    //   * both ranges have length: n_features_
                    //   * feature_index in range [0, n_features_)
                    return *(range1_first + feature_index) < *(range2_first + feature_index);
                });

            std::cout << "Partially sorted w.r.t. feature index: " << feature_index << "\n";
            print_flattened_matrix<DataType>(data, n_features_);

            printf("Kth smallest index quiery: %ld\n", kth_smallest_index);
            printf("Pivot range: ");
            print_flattened_matrix<DataType>(std::vector<DataType>(kth_smallest_begin, kth_smallest_end), n_features_);
            printf("\n");

            data = original_data;
        }
    }
}

TEST_F(UtilsErrorsTest, QuicksortIndexedRangeTest) {
    auto indices = std::vector<std::size_t>(n_samples_);
    std::iota(indices.begin(), indices.end(), 0);
    auto data = generate_flattened_matrix<DataType>(n_samples_, n_features_, -10, 10);

    constexpr std::size_t feature_index = 1;

    constexpr std::size_t pivot_index = n_samples_ / 2;

    static_assert(feature_index < n_features_);

    print_flattened_matrix(indices, 1);
    print_flattened_matrix<DataType>(data, n_features_);

    const auto new_pivot_index =
        ffcl::algorithms::quicksort_indexed_range(indices.begin(),
                                                  indices.end(),
                                                  data.begin(),
                                                  data.end(),
                                                  pivot_index,
                                                  n_features_,
                                                  [feature_index](const auto& range1_first, const auto& range2_first) {
                                                      // assumes that:
                                                      //   * both ranges have length: n_features_
                                                      //   * feature_index in range [0, n_features_)
                                                      return range1_first[feature_index] < range2_first[feature_index];
                                                  });

    print_flattened_matrix(indices, 1);
    const auto remapped_flattened_vector =
        remap_dataset(indices.begin(), indices.end(), data.begin(), data.end(), n_features_);
    print_flattened_matrix<DataType>(remapped_flattened_vector, n_features_);
    printf("Pivot index: %ld\n", pivot_index);
    printf("New pivot index: %ld\n", new_pivot_index);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}