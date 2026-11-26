#pragma once

#include "ffcl/search/Search.hpp"
#include "ffcl/search/buffer/Unsorted.hpp"

#include <memory>

#if defined(_OPENMP) && THREADS_ENABLED == true
#include <omp.h>
#endif

namespace ffcl::datastruct::mst {

template <typename Value>
class CoreDistancesArray {
  public:
    CoreDistancesArray(std::size_t size) noexcept;

    Value& operator[](std::size_t index);

    const Value& operator[](std::size_t index) const;

    auto data() -> Value*;

    auto data() const -> const Value*;

    auto size() const -> std::size_t;

  private:
    std::unique_ptr<Value[]> data_;
    std::size_t              size_;
};

template <typename Value>
CoreDistancesArray<Value>::CoreDistancesArray(std::size_t size) noexcept
  : data_{std::make_unique<Value[]>(size)}
  , size_{size} {}

template <typename Value>
Value& CoreDistancesArray<Value>::operator[](std::size_t index) {
    return data_[index];
}

template <typename Value>
const Value& CoreDistancesArray<Value>::operator[](std::size_t index) const {
    return data_[index];
}

template <typename Value>
auto CoreDistancesArray<Value>::data() -> Value* {
    return data_.get();
}

template <typename Value>
auto CoreDistancesArray<Value>::data() const -> const Value* {
    return data_.get();
}

template <typename Value>
auto CoreDistancesArray<Value>::size() const -> std::size_t {
    return size_;
}

// The sample indices in tree order: internal nodes own no samples, so the span runs from the leftmost
// leaf's first index to the rightmost leaf's last one.
template <typename Indexer>
auto leftmost_sample_iterator(const search::Searcher<Indexer>& searcher) {
    auto node = searcher.indexer().root();
    while (node && !node->is_leaf()) {
        node = node->left_;
    }
    return node ? node->indices_range_.first : searcher.indexer().root()->indices_range_.first;
}

template <typename Indexer>
auto rightmost_sample_iterator(const search::Searcher<Indexer>& searcher) {
    auto node = searcher.indexer().root();
    while (node && !node->is_leaf()) {
        node = node->right_;
    }
    return node ? node->indices_range_.second : searcher.indexer().root()->indices_range_.second;
}

// Core distance of every sample: the distance to its k-th nearest neighbour. The sample itself is one of the
// k candidates, so `k_nearest_neighbors = k` yields the distance to the (k-1)-th other sample.
//
// The kNN queries are independent and are parallelised when the library is compiled with
// `-DTHREADS_ENABLED=true` (same convention as `datastruct::PairwiseDistanceMatrix`).
template <typename Indexer>
auto make_static_core_distances(const search::Searcher<Indexer>&   searcher,
                                const typename Indexer::IndexType& k_nearest_neighbors)
    -> CoreDistancesArray<typename Indexer::DataType> {
    using DataType = typename Indexer::DataType;

    auto core_distances = CoreDistancesArray<DataType>(searcher.n_samples());

    // Query the samples in tree order rather than in index order. The datasets are shuffled, so in index
    // order consecutive queries jump across the tree; the leaves are contiguous in the index array, so in
    // tree order consecutive queries walk the same nodes. The results are stored by sample index either
    // way, so nothing depends on the order.
    const auto index_range_first = leftmost_sample_iterator(searcher);
    const auto index_range_last  = rightmost_sample_iterator(searcher);

    const auto n_queries = static_cast<std::size_t>(std::distance(index_range_first, index_range_last));

#if defined(_OPENMP) && THREADS_ENABLED == true
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (std::size_t query_index = 0; query_index < n_queries; ++query_index) {
        const auto sample_index = static_cast<std::size_t>(index_range_first[query_index]);

        auto nn_buffer_query = search::buffer::Unsorted(searcher.features_range_first(sample_index),
                                                        searcher.features_range_last(sample_index),
                                                        k_nearest_neighbors);

        core_distances[sample_index] = searcher(std::move(nn_buffer_query)).furthest_distance();
    }
    return core_distances;
}

// Same as above but returns the squared core distances, which is what the squared-distance Boruvka pass needs.
template <typename Indexer>
auto make_squared_core_distances(const search::Searcher<Indexer>&   searcher,
                                 const typename Indexer::IndexType& k_nearest_neighbors)
    -> CoreDistancesArray<typename Indexer::DataType> {
    auto core_distances = make_static_core_distances(searcher, k_nearest_neighbors);

    for (std::size_t sample_index = 0; sample_index < core_distances.size(); ++sample_index) {
        core_distances[sample_index] *= core_distances[sample_index];
    }
    return core_distances;
}

}  // namespace ffcl::datastruct::mst
