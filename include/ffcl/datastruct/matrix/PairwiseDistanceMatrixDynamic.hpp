#pragma once

#include "ffcl/math/heuristics/Distances.hpp"

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
class PairwiseDistanceMatrixDynamic {
  public:
    using ValueType             = typename SamplesIterator::value_type;
    using DatasetDescriptorType = std::tuple<SamplesIterator, SamplesIterator, std::size_t>;

    PairwiseDistanceMatrixDynamic(const SamplesIterator& samples_first,
                                  const SamplesIterator& samples_last,
                                  std::size_t            n_features);

    PairwiseDistanceMatrixDynamic(const DatasetDescriptorType& dataset_descriptor);

    auto operator()(std::size_t row_index, std::size_t column_index) const;

    std::size_t n_rows() const;

    std::size_t n_columns() const;

  private:
    SamplesIterator samples_first_;
    SamplesIterator samples_last_;
    std::size_t     n_features_;
};

template <typename SamplesIterator>
PairwiseDistanceMatrixDynamic<SamplesIterator>::PairwiseDistanceMatrixDynamic(const SamplesIterator& samples_first,
                                                                              const SamplesIterator& samples_last,
                                                                              std::size_t            n_features)
  : samples_first_{samples_first}
  , samples_last_{samples_last}
  , n_features_{n_features} {}

template <typename SamplesIterator>
PairwiseDistanceMatrixDynamic<SamplesIterator>::PairwiseDistanceMatrixDynamic(
    const DatasetDescriptorType& dataset_descriptor)
  : PairwiseDistanceMatrixDynamic<SamplesIterator>(std::get<0>(dataset_descriptor),
                                                   std::get<1>(dataset_descriptor),
                                                   std::get<2>(dataset_descriptor)) {}

template <typename SamplesIterator>
auto PairwiseDistanceMatrixDynamic<SamplesIterator>::operator()(std::size_t row_index, std::size_t column_index) const {
    // swap the indices if an lower triangle (diagonal excluded) quiery is made
    if (row_index > column_index) {
        std::swap(row_index, column_index);

        return math::heuristics::auto_distance(samples_first_ + column_index * n_features_,
                                               samples_first_ + column_index * n_features_ + n_features_,
                                               samples_first_ + row_index * n_features_);
    }
    return math::heuristics::auto_distance(samples_first_ + row_index * n_features_,
                                           samples_first_ + row_index * n_features_ + n_features_,
                                           samples_first_ + column_index * n_features_);
}

template <typename SamplesIterator>
std::size_t PairwiseDistanceMatrixDynamic<SamplesIterator>::n_rows() const {
    return common::utils::get_n_samples(samples_first_, samples_last_, n_features_);
}

template <typename SamplesIterator>
std::size_t PairwiseDistanceMatrixDynamic<SamplesIterator>::n_columns() const {
    return this->n_rows();
}

}  // namespace ffcl::datastruct