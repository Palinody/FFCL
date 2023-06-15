#include "ffcl/containers/kdtree/KDTreeAlgorithms.hpp"
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

#include "ffcl/common/Timer.hpp"

template <typename DataType>
class KDTreeAlgorithmsTestFixture : public Range2DBaseFixture<DataType> {
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
        n_features_     = 3;
        n_random_tests_ = 10;
    }

  protected:
    DataType    lower_bound_;
    DataType    upper_bound_;
    std::size_t min_n_samples_;
    std::size_t max_n_samples_;
    std::size_t n_features_;
    std::size_t n_random_tests_;
};

// using DataTypes = ::testing::Types<int>;
using DataTypes = ::testing::Types<int, std::size_t, float, double>;
TYPED_TEST_SUITE(KDTreeAlgorithmsTestFixture, DataTypes);

/*
template <typename IndicesIterator>
std::vector<typename IndicesIterator::value_type> kdtree_index_ranges_to_vector(
    IndicesIterator left_range_indices_first,
    IndicesIterator left_range_indices_last,
    IndicesIterator median_range_indices_first,
    IndicesIterator median_range_indices_last,
    IndicesIterator right_range_indices_first,
    IndicesIterator right_range_indices_last) {
    using DataType = typename IndicesIterator::value_type;

    std::vector<DataType> indices;
    indices.reserve(std::distance(left_range_indices_first, left_range_indices_last) +
                    std::distance(median_range_indices_first, median_range_indices_last) +
                    std::distance(right_range_indices_first, right_range_indices_last));

    std::copy(left_range_indices_first, left_range_indices_last, std::back_inserter(indices));
    std::copy(median_range_indices_first, median_range_indices_last, std::back_inserter(indices));
    std::copy(right_range_indices_first, right_range_indices_last, std::back_inserter(indices));

    return indices;
}

TYPED_TEST(KDTreeAlgorithmsTestFixture, ALotOfDuplicatesInDatasetToIndexTest) {
    const std::size_t samples       = 6;
    const std::size_t features      = 3;
    const std::size_t feature_index = 2;

    auto data = this->generate_random_uniform_vector(samples, features, 0, 2);
    // std::vector({2, 2, 2, 0, 5, 10});  // this->generate_ascending_elements_array(samples * features);

    auto data_indices = this->generate_indices(samples);

    const auto [cut_index, left_indexed_range, cut_indexed_range, right_indexed_range] =
        kdtree::algorithms::quickselect_median_indexed_range(
            data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

    const auto [indexed_cut_range_begin, indexed_cut_range_end] = cut_indexed_range;

    printf("n_left_indices: %ld", std::distance(left_indexed_range.first, left_indexed_range.second));
    printf("\n---\n");

    printf("left indices:\n");
    this->print_data(std::vector(left_indexed_range.first, left_indexed_range.second), 1);
    printf("\n---\n");

    printf("n_median_indices: %ld", std::distance(indexed_cut_range_begin, indexed_cut_range_end));
    printf("\n---\n");

    printf("Cut index: %ld\n", cut_index);

    printf("median indices:\n");
    this->print_data(std::vector(indexed_cut_range_begin, indexed_cut_range_end), 1);
    printf("\n---\n");

    printf("n_right_indices: %ld", std::distance(right_indexed_range.first, right_indexed_range.second));
    printf("\n---\n");

    printf("right indices:\n");
    this->print_data(std::vector(right_indexed_range.first, right_indexed_range.second), 1);
    printf("\n---\n");

    printf("Dataset:\n");
    this->print_data(data, features);
    printf("\n---\n");

    printf("All indices:\n");
    const auto all_indices = kdtree_index_ranges_to_vector(left_indexed_range.first,
                                                           left_indexed_range.second,
                                                           cut_indexed_range.first,
                                                           cut_indexed_range.second,
                                                           right_indexed_range.first,
                                                           right_indexed_range.second);
    this->print_data(all_indices, 1);
    printf("\n---\n");

    const auto remapped_dataset = common::utils::remap_ranges_from_indices(all_indices, data, features);

    printf("Remapped dataset:\n");
    this->print_data(remapped_dataset, features);
}
*/

/*
TYPED_TEST(KDTreeAlgorithmsTestFixture, ALotOfDuplicatesInDatasetTest) {
    const std::size_t samples       = 6;
    const std::size_t features      = 3;
    const std::size_t feature_index = 2;

    auto data = this->generate_random_uniform_vector(samples, features, 0, 2);
    // std::vector({2, 2, 2, 0, 5, 10});  // this->generate_ascending_elements_array(samples * features);

    printf("Dataset (original):\n");
    this->print_data(data, features);
    printf("\n---\n");

    const auto [cut_index, left_range, cut_range, right_range] =
        kdtree::algorithms::quickselect_median_range(data.begin(), data.end(), features, feature_index);

    const auto [cut_range_begin, cut_range_end] = cut_range;

    printf("n_samples_left: %ld", common::utils::get_n_samples(left_range.first, left_range.second, features));
    printf("\n---\n");

    printf("left values:\n");
    this->print_data(std::vector(left_range.first, left_range.second), features);
    printf("\n---\n");

    printf("Cut index: %ld\n", cut_index);

    printf("median values:\n");
    this->print_data(std::vector(cut_range_begin, cut_range_end), features);
    printf("\n---\n");

    printf("n_samples_right: %ld", common::utils::get_n_samples(right_range.first, right_range.second, features));
    printf("\n---\n");

    printf("right values:\n");
    this->print_data(std::vector(right_range.first, right_range.second), features);
    printf("\n---\n");

    printf("Dataset (remapped):\n");
    this->print_data(data, features);
}
*/

TYPED_TEST(KDTreeAlgorithmsTestFixture, Make1DBoundingBoxTest) {
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
                    const auto [min, max] =
                        kdtree::make_1d_bounding_box(data.begin(), data.end(), features, feature_index);

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(common::utils::equality(min, target_min) && common::utils::equality(max, target_max));
                }
            }
        }
    }
}

TYPED_TEST(KDTreeAlgorithmsTestFixture, Make1DBoundingBoxIndexedTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                auto data_indices = this->generate_indices(samples);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [min, max] = kdtree::make_1d_bounding_box(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(common::utils::equality(min, target_min) && common::utils::equality(max, target_max));
                }
            }
        }
    }
}

TYPED_TEST(KDTreeAlgorithmsTestFixture, MakeKDBoundingBoxTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                const auto kd_bounding_box = kdtree::make_kd_bounding_box(data.begin(), data.end(), features);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [min, max] = kd_bounding_box[feature_index];

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(common::utils::equality(min, target_min) && common::utils::equality(max, target_max));
                }
            }
        }
    }
}

TYPED_TEST(KDTreeAlgorithmsTestFixture, MakeKDBoundingBoxIndexedTest) {
    // the number of times to perform the tests
    for (std::size_t test_index = 0; test_index < this->n_random_tests_; ++test_index) {
        // tests on data from 1 to this->max_n_samples_ samples
        for (std::size_t samples = this->min_n_samples_; samples <= this->max_n_samples_; ++samples) {
            // tests on data from 1 to this->n_features_ features
            for (std::size_t features = 1; features <= this->n_features_; ++features) {
                const auto data =
                    this->generate_random_uniform_vector(samples, features, this->lower_bound_, this->upper_bound_);

                auto data_indices = this->generate_indices(samples);

                const auto kd_bounding_box = kdtree::make_kd_bounding_box(
                    data_indices.begin(), data_indices.end(), data.begin(), data.end(), features);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [min, max] = kd_bounding_box[feature_index];

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(common::utils::equality(min, target_min) && common::utils::equality(max, target_max));
                }
            }
        }
    }
}

TYPED_TEST(KDTreeAlgorithmsTestFixture, QuickselectMedianRangeTest) {
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
                        kdtree::algorithms::quickselect_median_range(shuffled_ascending_elements_array.begin(),
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

                    const auto res = this->is_pivot_faulty(shuffled_ascending_elements_array.begin(),
                                                           shuffled_ascending_elements_array.end(),
                                                           features,
                                                           kth_smallest_index,
                                                           feature_index);

                    ASSERT_TRUE(!res.has_value());
                }
            }
        }
    }
}

TYPED_TEST(KDTreeAlgorithmsTestFixture, QuickselectMedianIndexedRangeTest) {
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

                    auto data_indices = this->generate_indices(samples);

                    auto shuffled_ascending_elements_array = this->shuffle_by_row(
                        ascending_elements_array.begin(), ascending_elements_array.end(), features);

                    const auto [cut_index, left_indexed_range, cut_indexed_range, right_indexed_range] =
                        kdtree::algorithms::quickselect_median_indexed_range(data_indices.begin(),
                                                                             data_indices.end(),
                                                                             shuffled_ascending_elements_array.begin(),
                                                                             shuffled_ascending_elements_array.end(),
                                                                             features,
                                                                             feature_index);

                    const auto [indexed_cut_range_begin, indexed_cut_range_end] = cut_indexed_range;

                    // the range returned from quickselect should be the same as in the sorted dataset at index
                    // kth_smallest_index
                    ASSERT_TRUE(this->ranges_equality(
                        shuffled_ascending_elements_array.begin() + *indexed_cut_range_begin * features,
                        shuffled_ascending_elements_array.begin() + *indexed_cut_range_begin * features + features,
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

                    const auto res = this->is_pivot_faulty(data_indices.begin(),
                                                           data_indices.end(),
                                                           shuffled_ascending_elements_array.begin(),
                                                           shuffled_ascending_elements_array.end(),
                                                           features,
                                                           kth_smallest_index,
                                                           feature_index);

                    // print only if this->is_pivot_faulty returned values (meaning that its not valid)
                    if (res.has_value()) {
                        printf("------\nShuffled dataset with pivot_index: %ld\n", kth_smallest_index);
                        this->print_data(shuffled_ascending_elements_array, features);

                        printf("------\nPivot range:\n");
                        this->print_data(
                            std::vector(shuffled_ascending_elements_array.begin() + *indexed_cut_range_begin * features,
                                        shuffled_ascending_elements_array.begin() + *indexed_cut_range_end * features +
                                            features),
                            features);

                        printf("---\nRemapped dataset with pivot_index: %ld\n", kth_smallest_index);
                        const auto remapped_shuffled_ascending_elements_array =
                            common::utils::remap_ranges_from_indices(
                                std::vector(data_indices.begin(), data_indices.end()),
                                std::vector(shuffled_ascending_elements_array.begin(),
                                            shuffled_ascending_elements_array.end()),
                                features);

                        this->print_data(remapped_shuffled_ascending_elements_array, features);
                    }
                    // the pivot is valid if it didn't return std::nullopt
                    ASSERT_TRUE(!res.has_value());
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}