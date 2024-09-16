#pragma once

#include "ffcl/common/Utils.hpp"

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#elif !defined(_OPENMP) && THREADS_ENABLED == true
#include <thread>
#endif

namespace ffcl::common::math::linear_algebra {

template <typename SamplesIterator>
auto transpose_sequential(const SamplesIterator& samples_range_first,
                          const SamplesIterator& samples_range_last,
                          std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    std::vector<DataType> transposed(n_samples * n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            transposed[feature_index * n_samples + sample_index] =
                samples_range_first[sample_index * n_features + feature_index];
        }
    }
    // transposed data, number of samples (transposed), number of features (transposed)
    return std::make_tuple(transposed, n_features, n_samples);
}

template <typename SamplesIterator>
auto transpose_parallel_openmp(const SamplesIterator& samples_range_first,
                               const SamplesIterator& samples_range_last,
                               std::size_t            n_features) {
    using DataType = typename std::iterator_traits<SamplesIterator>::value_type;

    const std::size_t n_samples = get_n_samples(samples_range_first, samples_range_last, n_features);

    std::vector<DataType> transposed(n_samples * n_features);
    std::size_t           output_n_samples  = n_features;
    std::size_t           output_n_features = n_samples;

    static constexpr std::size_t block_size = 64;

#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for collapse(2)
#endif
    for (std::size_t sample_index = 0; sample_index < n_samples; sample_index += block_size) {
        for (std::size_t feature_index = 0; feature_index < n_features; feature_index += block_size) {
            // transpose the block
            for (std::size_t i = sample_index; i < sample_index + block_size && i < n_samples; ++i) {
                for (std::size_t j = feature_index; j < feature_index + block_size && j < n_features; ++j) {
                    transposed[j * n_samples + i] = samples_range_first[i * n_features + j];
                }
            }
        }
    }
    // transposed data, number of samples (transposed), number of features (transposed)
    return std::make_tuple(transposed, output_n_samples, output_n_features);
}

template <typename SamplesIterator>
auto transpose(const SamplesIterator& samples_range_first,
               const SamplesIterator& samples_range_last,
               std::size_t            n_features) {
#if defined(_OPENMP) && THREADS_ENABLED == true
    return transpose_parallel_openmp(samples_range_first, samples_range_last, n_features);
#else
    return transpose_sequential(samples_range_first, samples_range_last, n_features);
#endif
}

}  // namespace ffcl::common::math::linear_algebra