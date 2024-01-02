#pragma once

#include "ffcl/common/Utils.hpp"

namespace indexer {

template <typename IndexContainer, typename SamplesIterator>
class BaseIndexer {
  public:
    using IndexType = typename IndexContainer::value_type;
    using DataType  = typename std::iterator_traits<SamplesIterator>::value_type;

    using IndicesIteratorType = typename IndexContainer::iterator;
    using SamplesIteratorType = SamplesIterator;

    class BaseNearestNeighborsBuffer {
      public:
        BaseNearestNeighborsBuffer(IndexContainer&& indices, std::vector<DataType>&& distances) noexcept
          : indices_{std::move(indices)}
          , distances_{std::move(distances)} {}

        std::size_t size() const {
            return indices_.size();
        }

        IndexContainer indices() const {
            return indices_;
        }

        std::vector<DataType> distances() const {
            return distances_;
        }

        IndexContainer move_indices() {
            return std::move(indices_);
        }

        std::vector<DataType> move_distances() {
            return std::move(distances_);
        }

      private:
        IndexContainer        indices_;
        std::vector<DataType> distances_;
    };

    BaseIndexer(SamplesIterator data_first, SamplesIterator data_last, std::size_t n_features)
      : data_first_{data_first}
      , data_last_{data_last}
      , n_samples_{ffcl::common::get_n_samples(data_first, data_last, n_features)}
      , n_features_{n_features} {}

    virtual ~BaseIndexer() = default;

    virtual std::size_t n_samples() const = 0;

    virtual std::size_t n_features() const = 0;

    virtual BaseNearestNeighborsBuffer radius_search(std::size_t sample_index_query, const DataType& radius) const = 0;

    virtual BaseNearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                                      std::size_t k_nearest_neighbors) const = 0;

  protected:
    SamplesIterator data_first_, data_last_;
    std::size_t     n_samples_, n_features_;
};

}  // namespace indexer
