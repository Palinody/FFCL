#pragma once

#include "ffcl/common/Utils.hpp"

namespace indexer {

template <typename IndexContainer, typename SamplesIterator>
class BaseIndexer {
  public:
    using DataType = typename SamplesIterator::value_type;

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

        IndexContainer move_indices() {
            return std::move(indices_);
        }

      private:
        IndexContainer        indices_;
        std::vector<DataType> distances_;
    };

    virtual ~BaseIndexer() = default;

    virtual std::size_t n_samples() const = 0;

    virtual std::size_t n_features() const = 0;

    virtual BaseNearestNeighborsBuffer radiusSearch(std::size_t sample_index_query, const DataType& radius) const = 0;

    virtual BaseNearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                                      std::size_t k_nearest_neighbors) const = 0;
};

}  // namespace indexer