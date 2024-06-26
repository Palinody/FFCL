#include "ffcl/datastruct/tree/kdtree/KDTreeAlgorithms.hpp"
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

// /*
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
                    const auto interval =
                        ffcl::datastruct::make_tight_segment(data.begin(), data.end(), features, feature_index);

                    const auto [min, max] = std::make_pair(interval.lower_bound(), interval.upper_bound());

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(ffcl::common::equality(min, target_min) && ffcl::common::equality(max, target_max));
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
                    const auto interval = ffcl::datastruct::make_tight_segment(
                        data_indices.begin(), data_indices.end(), data.begin(), data.end(), features, feature_index);

                    const auto [min, max] = std::make_pair(interval.lower_bound(), interval.upper_bound());

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(ffcl::common::equality(min, target_min) && ffcl::common::equality(max, target_max));
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

                const auto bound = ffcl::datastruct::make_tight_bound(data.begin(), data.end(), features);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [min, max] = std::make_pair(bound.segment_at(feature_index).lower_bound(),
                                                           bound.segment_at(feature_index).upper_bound());

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(ffcl::common::equality(min, target_min) && ffcl::common::equality(max, target_max));
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

                const auto bound = ffcl::datastruct::make_tight_bound(
                    data_indices.begin(), data_indices.end(), data.begin(), data.end(), features);

                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    const auto [min, max] = std::make_pair(bound.segment_at(feature_index).lower_bound(),
                                                           bound.segment_at(feature_index).upper_bound());

                    const auto target_column = this->get_column(data.begin(), data.end(), features, feature_index);
                    const auto target_min    = *std::min_element(target_column.begin(), target_column.end());
                    const auto target_max    = *std::max_element(target_column.begin(), target_column.end());

                    ASSERT_TRUE(ffcl::common::equality(min, target_min) && ffcl::common::equality(max, target_max));
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
                const auto kth_smallest_index = (samples - 1) / 2;
                // test on all the possible feature indices
                for (std::size_t feature_index = 0; feature_index < features; ++feature_index) {
                    auto ascending_elements_array = this->generate_ascending_elements_array(samples * features);

                    auto data_indices = this->generate_indices(samples);

                    auto shuffled_ascending_elements_array = this->shuffle_by_row(
                        ascending_elements_array.begin(), ascending_elements_array.end(), features);

                    const auto [cut_index, left_indexed_range, cut_indexed_range, right_indexed_range] =
                        ffcl::datastruct::kdtree::algorithms::quickselect_median(
                            data_indices.begin(),
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
                        const auto remapped_shuffled_ascending_elements_array = ffcl::common::remap_ranges_from_indices(
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
// */

// /*
TYPED_TEST(KDTreeAlgorithmsTestFixture, PrintQuickselectMedianRangeTest) {
    // auto data = std::vector<float>{2, 1, 2, 3, 2, 1, 2, 3, -1, -2, 3, 0, 0, 0, 0, 9, 9, 7};
    auto data = std::vector<float>{1, 2, 3, -1, 5, -6, -2, 3, 0, 0, 0};
    // auto data = std::vector<int>{0, 10, -10};

    const std::size_t n_samples     = data.size();
    const std::size_t n_features    = 1;
    const std::size_t feature_index = 0;

    auto data_indices  = this->generate_indices(n_samples);
    auto shuffled_data = this->shuffle_by_row(data.begin(), data.end(), n_features);

    const auto [cut_index, left_indices_range, cut_indices_range, right_indices_range] =
        ffcl::datastruct::kdtree::algorithms::quickselect_median(data_indices.begin(),
                                                                 data_indices.end(),
                                                                 shuffled_data.begin(),
                                                                 shuffled_data.end(),
                                                                 n_features,
                                                                 feature_index);

    std::cout << "data:\n";
    this->print_data(shuffled_data, n_samples);

    std::cout << "left partition:\n";
    this->print_data(std::vector(left_indices_range.first, left_indices_range.second),
                     std::distance(left_indices_range.first, left_indices_range.second));

    std::cout << "cut_index:\n" << cut_index << "\n";

    std::cout << "cut partition:\n";
    this->print_data(std::vector(cut_indices_range.first, cut_indices_range.second),
                     std::distance(cut_indices_range.first, cut_indices_range.second));

    std::cout << "right partition:\n";
    this->print_data(std::vector(right_indices_range.first, right_indices_range.second),
                     std::distance(right_indices_range.first, right_indices_range.second));

    const auto remapped_shuffled_data =
        ffcl::common::remap_ranges_from_indices(std::vector(data_indices.begin(), data_indices.end()),
                                                std::vector(shuffled_data.begin(), shuffled_data.end()),
                                                n_features);
    std::cout << "remapped data:\n";
    this->print_data(remapped_shuffled_data, n_samples);

    const auto optional_fault = this->is_pivot_faulty(data_indices.begin(),
                                                      data_indices.end(),
                                                      shuffled_data.begin(),
                                                      shuffled_data.end(),
                                                      n_features,
                                                      cut_index,
                                                      feature_index);

    if (optional_fault) {
        std::cout << "pivot was found faulty\n";

    } else {
        std::cout << "left partition < pivot <= right partition\n";
    }
}
// */

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}