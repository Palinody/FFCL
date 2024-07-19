#pragma once

#include "ffcl/search/Search.hpp"
#include "ffcl/search/buffer/Unsorted.hpp"

#include <memory>

namespace ffcl::datastruct::mst {

template <typename Value>
class CoreDistancesArray {
  public:
    CoreDistancesArray(std::size_t size) noexcept;

    Value& operator[](std::size_t index) {
        return data_[index];
    }

    const Value& operator[](std::size_t index) const {
        return data_[index];
    }

  private:
    std::unique_ptr<Value[]> data_;
    std::size_t              size_;
};

template <typename Value>
CoreDistancesArray<Value>::CoreDistancesArray(std::size_t size) noexcept
  : data_{std::make_unique<Value[]>(size)}
  , size_{size_} {}

template <typename Indexer>
auto make_static_core_distances(const search::Searcher<Indexer>&   searcher,
                                const typename Indexer::IndexType& k_nearest_neighbors)
    -> CoreDistancesArray<typename Indexer::DataType> {
    // just a temporary array that allocates enough memory for the core distances
    auto core_distances = CoreDistancesArray<typename Indexer::DataType>(searcher.n_samples());

    for (std::size_t sample_index = 0; sample_index < searcher.n_samples(); ++sample_index) {
        auto nn_buffer_query = search::buffer::Unsorted(searcher.features_range_first(sample_index),
                                                        searcher.features_range_last(sample_index),
                                                        k_nearest_neighbors);

        core_distances[sample_index] = searcher(std::move(nn_buffer_query)).furthest_distance();
    }
    return core_distances;
}

}  // namespace ffcl::datastruct::mst