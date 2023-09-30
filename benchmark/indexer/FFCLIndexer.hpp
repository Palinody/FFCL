#pragma once

#include "BaseIndexer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTreeIndexed.hpp"

#include <cmath>

namespace indexer {

template <typename IndexContainer, typename SamplesIterator>
class FFCLIndexer : public BaseIndexer<IndexContainer, SamplesIterator> {
  public:
    using DataType = typename BaseIndexer<IndexContainer, SamplesIterator>::DataType;
    using BaseNearestNeighborsBuffer =
        typename BaseIndexer<IndexContainer, SamplesIterator>::BaseNearestNeighborsBuffer;

    using IndicesIterator = typename IndexContainer::iterator;

    using IndexerType             = ffcl::datastruct::KDTreeIndexed<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>;

    FFCLIndexer(IndicesIterator indices_first,
                IndicesIterator indices_last,
                SamplesIterator data_first,
                SamplesIterator data_last,
                std::size_t     n_features,
                std::size_t     max_leaf_size = 0)
      : BaseIndexer<IndexContainer, SamplesIterator>(data_first, data_last, n_features)
      , max_leaf_size_{max_leaf_size ? max_leaf_size : static_cast<std::size_t>(std::sqrt(this->n_samples_))}
      , indices_{IndexContainer(indices_first, indices_last)}
      , kd_tree_{ffcl::datastruct::KDTreeIndexed<IndicesIterator, SamplesIterator>(
            indices_.begin(),
            indices_.end(),
            this->data_first_,
            this->data_last_,
            this->n_features_,
            OptionsType()
                .bucket_size(max_leaf_size_)
                .max_depth(std::log2(this->n_samples_))
                .axis_selection_policy(AxisSelectionPolicyType())
                .splitting_rule_policy(SplittingRulePolicyType()))} {}

    std::size_t n_samples() const override {
        return this->n_samples_;
    }

    std::size_t n_features() const override {
        return this->n_features_;
    }

    BaseNearestNeighborsBuffer radius_search(std::size_t sample_index_query, const DataType& radius) const override {
        auto nearest_neighbors_buffer = kd_tree_.radius_search_around_query_index(sample_index_query, radius);

        return BaseNearestNeighborsBuffer(nearest_neighbors_buffer.move_indices(),
                                          nearest_neighbors_buffer.move_distances());
    }

    BaseNearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                              std::size_t k_nearest_neighbors) const override {
        auto nearest_neighbors_buffer =
            kd_tree_.k_nearest_neighbors_around_query_index(sample_index_query, k_nearest_neighbors);

        return BaseNearestNeighborsBuffer(nearest_neighbors_buffer.move_indices(),
                                          nearest_neighbors_buffer.move_distances());
    }

  private:
    std::size_t max_leaf_size_;

    IndexContainer indices_;
    IndexerType    kd_tree_;
};

}  // namespace indexer