#pragma once

#include "cpp_clustering/heuristics/Heuristics.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

namespace cpp_clustering::containers {

template <typename Iterator>
std::vector<std::vector<typename Iterator::value_type>> make_pairwise_low_triangle_distance_matrix(
    const Iterator& samples_first,
    const Iterator& samples_last,
    std::size_t     n_features) {
    using ValueType = typename Iterator::value_type;

    const std::size_t n_samples = common::utils::get_n_samples(samples_first, samples_last, n_features);

    auto low_triangle_distance_matrix = std::vector<std::vector<ValueType>>(n_samples);

#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::size_t i = 0; i < n_samples; ++i) {
        // Make one row at a time of the lower triangle part of the matrix. The diagonal is created to avoid adding
        // conditions during the access of the elements. The values are not computed (= zero)
        std::vector<ValueType> temp_row(i + 1);

        for (std::size_t j = 0; j < i; ++j) {
            temp_row[j] = cpp_clustering::heuristic::heuristic(
                /*first sample begin=*/samples_first + i * n_features,
                /*first sample end=*/samples_first + i * n_features + n_features,
                /*second sample begin=*/samples_first + j * n_features);
        }
#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp critical
#endif
        low_triangle_distance_matrix[i] = std::move(temp_row);
    }
    return low_triangle_distance_matrix;
}

template <typename Iterator>
std::vector<std::vector<typename Iterator::value_type>> make_pairwise_low_triangle_distance_matrix(
    const std::tuple<Iterator, Iterator, std::size_t>& dataset_descriptor) {
    return make_pairwise_low_triangle_distance_matrix(
        std::get<0>(dataset_descriptor), std::get<1>(dataset_descriptor), std::get<2>(dataset_descriptor));
}

template <typename Iterator>
class LowerTriangleMatrix {
  public:
    using ValueType             = typename Iterator::value_type;
    using DatasetDescriptorType = std::tuple<Iterator, Iterator, std::size_t>;

    LowerTriangleMatrix(const Iterator& samples_first, const Iterator& samples_last, std::size_t n_features)
      : data_{make_pairwise_low_triangle_distance_matrix(samples_first, samples_last, n_features)} {}

    LowerTriangleMatrix(const DatasetDescriptorType& dataset_descriptor)
      : data_{make_pairwise_low_triangle_distance_matrix(dataset_descriptor)} {}

    ValueType& operator()(std::size_t sample_index, std::size_t feature_index) {
        // swap the indices if an upper triangle (diagonal excluded) quiery is made
        if (feature_index > sample_index) {
            return data_[feature_index][sample_index];
        }
        return data_[sample_index][feature_index];
    }

    const ValueType& operator()(std::size_t sample_index, std::size_t feature_index) const {
        // swap the indices if an upper triangle (diagonal excluded) quiery is made
        if (feature_index > sample_index) {
            return data_[feature_index][sample_index];
        }
        return data_[sample_index][feature_index];
    }

    std::size_t n_samples() const {
        return data_.size();
    }

  private:
    std::vector<std::vector<ValueType>> data_;
};

template <typename Iterator>
class LowerTriangleMatrixDynamic {
  public:
    using ValueType             = typename Iterator::value_type;
    using DatasetDescriptorType = std::tuple<Iterator, Iterator, std::size_t>;

    LowerTriangleMatrixDynamic(const Iterator& samples_first, const Iterator& samples_last, std::size_t n_features)
      : samples_first_{samples_first}
      , samples_last_{samples_last}
      , n_features_{n_features} {}

    LowerTriangleMatrixDynamic(const DatasetDescriptorType& dataset_descriptor)
      : LowerTriangleMatrixDynamic<Iterator>(std::get<0>(dataset_descriptor),
                                             std::get<1>(dataset_descriptor),
                                             std::get<2>(dataset_descriptor)) {}

    ValueType operator()(std::size_t sample_index, std::size_t feature_index) const {
        // swap the indices if an upper triangle (diagonal excluded) quiery is made
        if (feature_index > sample_index) {
            return cpp_clustering::heuristic::heuristic(
                /*first sample begin=*/samples_first_ + feature_index * n_features_,
                /*first sample end=*/samples_first_ + feature_index * n_features_ + n_features_,
                /*other sample begin=*/samples_first_ + sample_index * n_features_);
        }
        return cpp_clustering::heuristic::heuristic(
            /*first sample begin=*/samples_first_ + sample_index * n_features_,
            /*first sample end=*/samples_first_ + sample_index * n_features_ + n_features_,
            /*other sample begin=*/samples_first_ + feature_index * n_features_);
    }

    std::size_t n_samples() const {
        return common::utils::get_n_samples(samples_first_, samples_last_, n_features_);
    }

  private:
    Iterator    samples_first_;
    Iterator    samples_last_;
    std::size_t n_features_;
};

}  // namespace cpp_clustering::containers