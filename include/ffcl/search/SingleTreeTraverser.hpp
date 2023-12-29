#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/StaticBase.hpp"
#include "ffcl/search/count/StaticBase.hpp"

namespace ffcl::search {

template <typename ReferenceIndexer>
class SingleTreeTraverser {
  public:
    using IndexType = typename ReferenceIndexer::IndexType;
    using DataType  = typename ReferenceIndexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    using IndicesIteratorType = typename ReferenceIndexer::IndicesIteratorType;
    using SamplesIteratorType = typename ReferenceIndexer::SamplesIteratorType;

    static_assert(common::is_iterator<IndicesIteratorType>::value, "IndicesIteratorType is not an iterator");
    static_assert(common::is_iterator<SamplesIteratorType>::value, "SamplesIteratorType is not an iterator");

    using KDNodeViewPtr = typename ReferenceIndexer::KDNodeViewPtr;

    static_assert(common::is_raw_or_smart_ptr<KDNodeViewPtr>(), "KDNodeViewPtr is not a row or smart pointer");

    SingleTreeTraverser(const ReferenceIndexer& reference_indexer)
      : reference_indexer_{reference_indexer} {}

    SingleTreeTraverser(ReferenceIndexer&& reference_indexer)
      : reference_indexer_{std::move(reference_indexer)} {}

    template <typename Buffer>
    Buffer operator()(Buffer&& input_buffer) const {
        static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                      "Provided a Buffer that does not inherit from StaticBase<Derived>");

        auto processed_buffer = std::forward<Buffer>(input_buffer);
        single_tree_traversal(reference_indexer_.root(), processed_buffer);
        return processed_buffer;
    }

    std::size_t n_samples() const {
        return reference_indexer_.n_samples();
    }

    constexpr auto features_range_first(std::size_t sample_index) const {
        return reference_indexer_.features_range_first(sample_index);
    }

    constexpr auto features_range_last(std::size_t sample_index) const {
        return reference_indexer_.features_range_last(sample_index);
    }

  private:
    template <typename Buffer>
    void single_tree_traversal(KDNodeViewPtr node, Buffer& buffer) const {
        // current_node is currently a leaf node (and root in the special case where the entire tree is in a single
        // node)
        auto current_kdnode = recursive_search_to_leaf_node(
            /**/ node,
            /**/ buffer);

        // performs a nearest neighbor search one step at a time from the leaf node until the input node is reached if
        // node parameter is a subtree. A search through the entire tree
        while (current_kdnode != node) {
            // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
            // (nullptr otherwise)
            current_kdnode = get_parent_node_after_sibling_traversal(
                /**/ current_kdnode,
                /**/ buffer);
        }
    }

    template <typename Buffer>
    auto recursive_search_to_leaf_node(KDNodeViewPtr node, Buffer& buffer) const -> decltype(node) {
        buffer.partial_search(node->indices_range_.first,
                              node->indices_range_.second,
                              reference_indexer_.begin(),
                              reference_indexer_.end(),
                              reference_indexer_.n_features());

        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            // get the pivot sample index in the dataset
            const auto pivot_index = node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = reference_indexer_[pivot_index][node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = buffer.centroid_begin()[node->cut_feature_index_];

            // traverse either the left or right child node depending on where the target sample is located relatively
            // to the cut value
            if (query_split_value < pivot_split_value) {
                node = recursive_search_to_leaf_node(
                    /**/ node->left_,
                    /**/ buffer);
            } else {
                node = recursive_search_to_leaf_node(
                    /**/ node->right_,
                    /**/ buffer);
            }
        }
        return node;
    }

    template <typename Buffer>
    auto get_parent_node_after_sibling_traversal(const KDNodeViewPtr& node, Buffer& buffer) const
        -> decltype(node->parent_.lock()) {
        auto parent_node = node->parent_.lock();
        // if node has a parent
        if (parent_node) {
            // get the pivot sample index in the dataset
            const auto pivot_index = parent_node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = reference_indexer_[pivot_index][parent_node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = buffer.centroid_begin()[parent_node->cut_feature_index_];
            // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a
            // nearest neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are
            // put to the right
            const bool visit_sibling =
                node->is_left_child()
                    ? buffer.n_free_slots() || common::abs(pivot_split_value - query_split_value) <=
                                                   buffer.upper_bound(parent_node->cut_feature_index_)
                    : buffer.n_free_slots() || common::abs(pivot_split_value - query_split_value) <
                                                   buffer.upper_bound(parent_node->cut_feature_index_);
            // we perform the nearest neighbor algorithm on the subtree starting from the sibling if the split value is
            // closer to the query sample than the current nearest neighbor
            if (visit_sibling) {
                // if the sibling node is not nullptr
                if (auto sibling_node = node->get_sibling_node()) {
                    // get the nearest neighbor from the sibling node
                    single_tree_traversal(
                        /**/ sibling_node,
                        /**/ buffer);
                }
            }
        }
        // returns nullptr if node doesnt have parent (or is root)
        return parent_node;
    }

    ReferenceIndexer reference_indexer_;
};

}  // namespace ffcl::search