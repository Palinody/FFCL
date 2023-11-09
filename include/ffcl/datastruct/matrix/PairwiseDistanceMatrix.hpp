#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cstddef>  // std::size_t
#include <iomanip>  // std::setw, std::fixed
#include <iostream>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

namespace ffcl::datastruct {

template <typename SamplesIterator>
class PairwiseDistanceMatrix {
  public:
    using ValueType             = typename SamplesIterator::value_type;
    using DatasetDescriptorType = std::tuple<SamplesIterator, SamplesIterator, std::size_t>;

    PairwiseDistanceMatrix(const SamplesIterator& samples_range_first,
                           const SamplesIterator& samples_range_last,
                           std::size_t            n_features);

    PairwiseDistanceMatrix(const DatasetDescriptorType& dataset_descriptor);

    auto operator()(std::size_t row_index, std::size_t column_index) const;

    std::size_t n_rows() const {
        return n_samples_;
    }

    std::size_t n_columns() const {
        return this->n_rows();
    }

  private:
    auto compute_pairwise_distances_sequential(const SamplesIterator& samples_range_first,
                                               const SamplesIterator& samples_range_last,
                                               std::size_t            n_features);

    auto compute_pairwise_distances_parallel(const SamplesIterator& samples_range_first,
                                             const SamplesIterator& samples_range_last,
                                             std::size_t            n_features);

    auto compute_pairwise_distances(const SamplesIterator& samples_range_first,
                                    const SamplesIterator& samples_range_last,
                                    std::size_t            n_features);

    auto compute_pairwise_distances(
        const std::tuple<SamplesIterator, SamplesIterator, std::size_t>& dataset_descriptor);

    std::size_t n_samples_;

    std::vector<ValueType> data_;
};

template <typename SamplesIterator>
PairwiseDistanceMatrix<SamplesIterator>::PairwiseDistanceMatrix(const SamplesIterator& samples_range_first,
                                                                const SamplesIterator& samples_range_last,
                                                                std::size_t            n_features)
  : n_samples_{common::get_n_samples(samples_range_first, samples_range_last, n_features)}
  , data_{compute_pairwise_distances(samples_range_first, samples_range_last, n_features)} {}

template <typename SamplesIterator>
PairwiseDistanceMatrix<SamplesIterator>::PairwiseDistanceMatrix(const DatasetDescriptorType& dataset_descriptor)
  : n_samples_{common::get_n_samples(std::get<0>(dataset_descriptor),
                                     std::get<1>(dataset_descriptor),
                                     std::get<2>(dataset_descriptor))}
  , data_{compute_pairwise_distances(dataset_descriptor)} {}

template <typename SamplesIterator>
auto PairwiseDistanceMatrix<SamplesIterator>::operator()(std::size_t row_index, std::size_t column_index) const {
    if (row_index == column_index) {
        return static_cast<ValueType>(0);
    }
    // swap the indices if an upper triangle (diagonal excluded) quiery is made
    if (row_index < column_index) {
        std::swap(row_index, column_index);
    }
    const std::size_t flat_index = row_index * (row_index - 1) / 2;

    return data_[flat_index + column_index];
}

template <typename SamplesIterator>
auto PairwiseDistanceMatrix<SamplesIterator>::compute_pairwise_distances_parallel(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features) {
    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    auto low_triangle_distance_matrix = std::vector<ValueType>(n_samples * (n_samples - 1) / 2);

#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t row_index = 1; row_index < n_samples; ++row_index) {
        const std::size_t flat_index = row_index * (row_index - 1) / 2;

        for (std::size_t column_index = 0; column_index < row_index; ++column_index) {
            low_triangle_distance_matrix[flat_index + column_index] =
                common::math::heuristics::auto_distance(samples_range_first + row_index * n_features,
                                                        samples_range_first + row_index * n_features + n_features,
                                                        samples_range_first + column_index * n_features);
        }
    }
    return low_triangle_distance_matrix;
}

template <typename SamplesIterator>
auto PairwiseDistanceMatrix<SamplesIterator>::compute_pairwise_distances_sequential(
    const SamplesIterator& samples_range_first,
    const SamplesIterator& samples_range_last,
    std::size_t            n_features) {
    const std::size_t n_samples = common::get_n_samples(samples_range_first, samples_range_last, n_features);

    auto low_triangle_distance_matrix = std::vector<ValueType>(n_samples * (n_samples - 1) / 2);

    for (std::size_t row_index = 1; row_index < n_samples; ++row_index) {
        const std::size_t flat_index = row_index * (row_index - 1) / 2;

        for (std::size_t column_index = 0; column_index < row_index; ++column_index) {
            low_triangle_distance_matrix[flat_index + column_index] =
                common::math::heuristics::auto_distance(samples_range_first + row_index * n_features,
                                                        samples_range_first + row_index * n_features + n_features,
                                                        samples_range_first + column_index * n_features);
        }
    }
    return low_triangle_distance_matrix;
}

template <typename SamplesIterator>
auto PairwiseDistanceMatrix<SamplesIterator>::compute_pairwise_distances(const SamplesIterator& samples_range_first,
                                                                         const SamplesIterator& samples_range_last,
                                                                         std::size_t            n_features) {
#if defined(_OPENMP) && THREADS_ENABLED == true
    return compute_pairwise_distances_parallel(samples_range_first, samples_range_last, n_features);
#else
    return compute_pairwise_distances_sequential(samples_range_first, samples_range_last, n_features);
#endif
}

template <typename SamplesIterator>
auto PairwiseDistanceMatrix<SamplesIterator>::compute_pairwise_distances(
    const std::tuple<SamplesIterator, SamplesIterator, std::size_t>& dataset_descriptor) {
    return compute_pairwise_distances(
        std::get<0>(dataset_descriptor), std::get<1>(dataset_descriptor), std::get<2>(dataset_descriptor));
}

template <typename Matrix>
void print_matrix(const Matrix& matrix) {
    static constexpr std::size_t integral_cout_width = 3;
    static constexpr std::size_t decimal_cout_width  = 3;

    for (std::size_t row_index = 0; row_index < matrix.n_rows(); ++row_index) {
        for (std::size_t column_index = 0; column_index < matrix.n_columns(); ++column_index) {
            // Set the output format
            std::cout << std::setw(integral_cout_width + decimal_cout_width + 1) << std::fixed
                      << std::setprecision(decimal_cout_width) << matrix(row_index, column_index) << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace ffcl::datastruct