#pragma once

#include "ffcl/math/heuristics/Distances.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cstddef>  // std::size_t
#include <iomanip>  // std::setw, std::fixed
#include <iostream>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

namespace ffcl::datastruct {

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
            temp_row[j] = math::heuristics::auto_distance(samples_first + i * n_features,
                                                          samples_first + i * n_features + n_features,
                                                          samples_first + j * n_features);
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
class PrecomputedDistanceMatrix {
  public:
    using ValueType             = typename Iterator::value_type;
    using DatasetDescriptorType = std::tuple<Iterator, Iterator, std::size_t>;

    PrecomputedDistanceMatrix(const Iterator& samples_first, const Iterator& samples_last, std::size_t n_features);

    PrecomputedDistanceMatrix(const DatasetDescriptorType& dataset_descriptor);

    ValueType& operator()(std::size_t sample_index, std::size_t feature_index);

    const ValueType& operator()(std::size_t sample_index, std::size_t feature_index) const;

    std::size_t n_samples() const {
        return data_.size();
    }

  private:
    std::vector<std::vector<ValueType>> data_;
};

template <typename Iterator>
PrecomputedDistanceMatrix<Iterator>::PrecomputedDistanceMatrix(const Iterator& samples_first,
                                                               const Iterator& samples_last,
                                                               std::size_t     n_features)
  : data_{make_pairwise_low_triangle_distance_matrix(samples_first, samples_last, n_features)} {}

template <typename Iterator>
PrecomputedDistanceMatrix<Iterator>::PrecomputedDistanceMatrix(const DatasetDescriptorType& dataset_descriptor)
  : data_{make_pairwise_low_triangle_distance_matrix(dataset_descriptor)} {}

template <typename Iterator>
typename PrecomputedDistanceMatrix<Iterator>::ValueType& PrecomputedDistanceMatrix<Iterator>::operator()(
    std::size_t sample_index,
    std::size_t feature_index) {
    // swap the indices if an upper triangle (diagonal excluded) quiery is made
    if (feature_index > sample_index) {
        return data_[feature_index][sample_index];
    }
    return data_[sample_index][feature_index];
}

template <typename Iterator>
const typename PrecomputedDistanceMatrix<Iterator>::ValueType& PrecomputedDistanceMatrix<Iterator>::operator()(
    std::size_t sample_index,
    std::size_t feature_index) const {
    // swap the indices if an upper triangle (diagonal excluded) quiery is made
    if (feature_index > sample_index) {
        return data_[feature_index][sample_index];
    }
    return data_[sample_index][feature_index];
}

template <typename Iterator>
class PrecomputedDistanceMatrixDynamic {
  public:
    using ValueType             = typename Iterator::value_type;
    using DatasetDescriptorType = std::tuple<Iterator, Iterator, std::size_t>;

    PrecomputedDistanceMatrixDynamic(const Iterator& samples_first,
                                     const Iterator& samples_last,
                                     std::size_t     n_features);

    PrecomputedDistanceMatrixDynamic(const DatasetDescriptorType& dataset_descriptor);

    ValueType operator()(std::size_t sample_index, std::size_t feature_index) const;

    std::size_t n_samples() const;

  private:
    Iterator    samples_first_;
    Iterator    samples_last_;
    std::size_t n_features_;
};

template <typename Iterator>
PrecomputedDistanceMatrixDynamic<Iterator>::PrecomputedDistanceMatrixDynamic(const Iterator& samples_first,
                                                                             const Iterator& samples_last,
                                                                             std::size_t     n_features)
  : samples_first_{samples_first}
  , samples_last_{samples_last}
  , n_features_{n_features} {}

template <typename Iterator>
PrecomputedDistanceMatrixDynamic<Iterator>::PrecomputedDistanceMatrixDynamic(
    const DatasetDescriptorType& dataset_descriptor)
  : PrecomputedDistanceMatrixDynamic<Iterator>(std::get<0>(dataset_descriptor),
                                               std::get<1>(dataset_descriptor),
                                               std::get<2>(dataset_descriptor)) {}

template <typename Iterator>
typename PrecomputedDistanceMatrixDynamic<Iterator>::ValueType PrecomputedDistanceMatrixDynamic<Iterator>::operator()(
    std::size_t sample_index,
    std::size_t feature_index) const {
    // swap the indices if an upper triangle (diagonal excluded) quiery is made
    if (feature_index > sample_index) {
        return math::heuristics::auto_distance(samples_first_ + feature_index * n_features_,
                                               samples_first_ + feature_index * n_features_ + n_features_,
                                               samples_first_ + sample_index * n_features_);
    }
    return math::heuristics::auto_distance(samples_first_ + sample_index * n_features_,
                                           samples_first_ + sample_index * n_features_ + n_features_,
                                           samples_first_ + feature_index * n_features_);
}

template <typename Iterator>
std::size_t PrecomputedDistanceMatrixDynamic<Iterator>::n_samples() const {
    return common::utils::get_n_samples(samples_first_, samples_last_, n_features_);
}

template <typename Matrix>
void print_matrix(const Matrix& matrix) {
    static constexpr std::size_t integral_cout_width = 3;
    static constexpr std::size_t decimal_cout_width  = 3;

    for (std::size_t sample_index = 0; sample_index < matrix.n_samples(); ++sample_index) {
        for (std::size_t other_sample_index = 0; other_sample_index < matrix.n_samples(); ++other_sample_index) {
            // Set the output format
            std::cout << std::setw(integral_cout_width + decimal_cout_width + 1) << std::fixed
                      << std::setprecision(decimal_cout_width) << matrix(sample_index, other_sample_index) << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace ffcl::datastruct