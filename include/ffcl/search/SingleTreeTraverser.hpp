#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/Base.hpp"
#include "ffcl/search/count/Base.hpp"

namespace ffcl::search {

template <typename IndexerPtr>
class SingleTreeTraverser {
  public:
    static_assert(common::is_raw_or_smart_ptr<IndexerPtr>());

    using IndexType           = typename IndexerPtr::element_type::IndexType;
    using DataType            = typename IndexerPtr::element_type::DataType;
    using IndicesIteratorType = typename IndexerPtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename IndexerPtr::element_type::SamplesIteratorType;

    using KDNodeViewPtr = typename IndexerPtr::element_type::KDNodeViewPtr;

    SingleTreeTraverser(IndexerPtr query_indexer_ptr)
      : query_indexer_ptr_{query_indexer_ptr} {}

    template <typename Buffer>
    Buffer operator()(Buffer&& input_buffer) {
        static_assert(common::is_crtp_of<Buffer, buffer::StaticBase>::value,
                      "Derived class does not inherit from StaticBase<Derived>");

        auto processed_buffer = std::forward<Buffer>(input_buffer);
        single_tree_traversal(query_indexer_ptr_->root(), processed_buffer);
        return processed_buffer;
    }

  private:
    template <typename Buffer>
    void single_tree_traversal(KDNodeViewPtr node, Buffer& buffer) {
        // current_node is currently a leaf node (and root in the special case where the entire tree is in a single
        // node)
        auto current_kdnode = recurse_to_closest_leaf_node(
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
    KDNodeViewPtr recurse_to_closest_leaf_node(KDNodeViewPtr node, Buffer& buffer) {
        buffer.partial_search(node->indices_range_.first,
                              node->indices_range_.second,
                              query_indexer_ptr_->begin(),
                              query_indexer_ptr_->end(),
                              query_indexer_ptr_->n_features());

        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            // get the pivot sample index in the dataset
            const auto pivot_index = node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = (*query_indexer_ptr_)[pivot_index][node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = buffer.centroid_begin()[node->cut_feature_index_];

            // traverse either the left or right child node depending on where the target sample is located relatively
            // to the cut value
            if (query_split_value < pivot_split_value) {
                node = recurse_to_closest_leaf_node(
                    /**/ node->left_,
                    /**/ buffer);
            } else {
                node = recurse_to_closest_leaf_node(
                    /**/ node->right_,
                    /**/ buffer);
            }
        }
        return node;
    }

    template <typename Buffer>
    KDNodeViewPtr get_parent_node_after_sibling_traversal(KDNodeViewPtr node, Buffer& buffer) {
        auto parent_node = node->parent_.lock();
        // if node has a parent
        if (parent_node) {
            // get the pivot sample index in the dataset
            const auto pivot_index = parent_node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = (*query_indexer_ptr_)[pivot_index][parent_node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = buffer.centroid_begin()[parent_node->cut_feature_index_];
            // if the axiswise distance is equal to the current furthest nearest neighbor distance, there could be a
            // nearest neighbor to the other side of the hyperrectangle since the values that are equal to the pivot are
            // put to the right
            bool visit_sibling = node->is_left_child()
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

    IndexerPtr query_indexer_ptr_;
};

}  // namespace ffcl::search