#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/knn/buffer/Base.hpp"
#include "ffcl/knn/count/Base.hpp"

namespace ffcl::knn::count {

template <typename IndexerPtr>
class SingleTreeTraverser {
  public:
    using IndexType           = typename IndexerPtr::element_type::IndexType;
    using DataType            = typename IndexerPtr::element_type::DataType;
    using IndicesIteratorType = typename IndexerPtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename IndexerPtr::element_type::SamplesIteratorType;

    using KDNodeViewPtr = typename IndexerPtr::element_type::KDNodeViewPtr;

    SingleTreeTraverser(IndexerPtr query_indexer_ptr)
      : query_indexer_ptr_{query_indexer_ptr} {}

    template <typename BufferType>
    BufferType operator()(std::size_t query_index, BufferType& buffer) {
        single_tree_traversal(query_index, buffer, query_indexer_ptr_->root());
        return buffer;
    }

    template <typename BufferType>
    BufferType operator()(const SamplesIteratorType& query_feature_first,
                          const SamplesIteratorType& query_feature_last,
                          BufferType&                buffer) {
        single_tree_traversal(query_feature_first, query_feature_last, buffer, query_indexer_ptr_->root());
        return buffer;
    }

  private:
    template <typename BufferType>
    void single_tree_traversal(std::size_t query_index, BufferType& buffer, const KDNodeViewPtr& node) {
        // current_node is currently a leaf node (and root in the special case where the entire tree is in a single
        // node)
        auto current_kdnode = recurse_to_closest_leaf_node(
            /**/ query_index,
            /**/ buffer,
            /**/ node);

        // performs a nearest neighbor search one step at a time from the leaf node until the input node is reached if
        // node parameter is a subtree. A search through the entire tree
        while (current_kdnode != node) {
            // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
            // (nullptr otherwise)
            current_kdnode = get_parent_node_after_sibling_traversal(
                /**/ query_index,
                /**/ buffer,
                /**/ current_kdnode);
        }
    }

    template <typename BufferType>
    auto recurse_to_closest_leaf_node(std::size_t query_index, BufferType& buffer, const KDNodeViewPtr& node) {
        buffer.search(node->indices_range_.first,
                      node->indices_range_.second,
                      query_indexer_ptr_->begin(),
                      query_indexer_ptr_->end(),
                      query_indexer_ptr_->n_features(),
                      query_index);

        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            // get the pivot sample index in the dataset
            const auto pivot_index = node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = (*query_indexer_ptr_)[pivot_index][node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = (*query_indexer_ptr_)[query_index][node->cut_feature_index_];

            // traverse either the left or right child node depending on where the target sample is located relatively
            // to the cut value
            if (query_split_value < pivot_split_value) {
                node = recurse_to_closest_leaf_node(
                    /**/ query_index,
                    /**/ buffer,
                    /**/ node->left_);
            } else {
                node = recurse_to_closest_leaf_node(
                    /**/ query_index,
                    /**/ buffer,
                    /**/ node->right_);
            }
        }
        return node;
    }

    template <typename BufferType>
    auto get_parent_node_after_sibling_traversal(std::size_t          query_index,
                                                 BufferType&          buffer,
                                                 const KDNodeViewPtr& node) {
        auto parent_node = node->parent_.lock();
        // if node has a parent
        if (parent_node) {
            // get the pivot sample index in the dataset
            const auto pivot_index = parent_node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = (*query_indexer_ptr_)[pivot_index][parent_node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = (*query_indexer_ptr_)[query_index][parent_node->cut_feature_index_];
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
                        /**/ query_index,
                        /**/ buffer,
                        /**/ sibling_node);
                }
            }
        }
        // returns nullptr if node doesnt have parent (or is root)
        return parent_node;
    }

    template <typename BufferType>
    void single_tree_traversal(const SamplesIteratorType& query_feature_first,
                               const SamplesIteratorType& query_feature_last,
                               BufferType&                buffer,
                               const KDNodeViewPtr&       node) {
        // current_node is currently a leaf node (and root in the special case where the entire tree is in a single
        // node)
        auto current_kdnode = recurse_to_closest_leaf_node(
            /**/ query_feature_first,
            /**/ query_feature_last,
            /**/ buffer,
            /**/ node);

        // performs a nearest neighbor search one step at a time from the leaf node until the input node is reached if
        // node parameter is a subtree. A search through the entire tree
        while (current_kdnode != node) {
            // performs a nearest neighbor search starting from the specified node then returns its parent if it exists
            // (nullptr otherwise)
            current_kdnode = get_parent_node_after_sibling_traversal(
                /**/ query_feature_first,
                /**/ query_feature_last,
                /**/ buffer,
                /**/ current_kdnode);
        }
    }

    template <typename BufferType>
    auto recurse_to_closest_leaf_node(const SamplesIteratorType& query_feature_first,
                                      const SamplesIteratorType& query_feature_last,
                                      BufferType&                buffer,
                                      const KDNodeViewPtr&       node) {
        buffer.search(node->indices_range_.first,
                      node->indices_range_.second,
                      query_indexer_ptr_->begin(),
                      query_indexer_ptr_->end(),
                      query_indexer_ptr_->n_features(),
                      query_feature_first,
                      query_feature_last);

        // continue to recurse down the tree if the current node is not leaf until we reach a terminal node
        if (!node->is_leaf()) {
            // get the pivot sample index in the dataset
            const auto pivot_index = node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = (*query_indexer_ptr_)[pivot_index][node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = query_feature_first[node->cut_feature_index_];

            // traverse either the left or right child node depending on where the target sample is located relatively
            // to the cut value
            if (query_split_value < pivot_split_value) {
                node = recurse_to_closest_leaf_node(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ buffer,
                    /**/ node->left_);
            } else {
                node = recurse_to_closest_leaf_node(
                    /**/ query_feature_first,
                    /**/ query_feature_last,
                    /**/ buffer,
                    /**/ node->right_);
            }
        }
        return node;
    }

    template <typename BufferType>
    auto get_parent_node_after_sibling_traversal(const SamplesIteratorType& query_feature_first,
                                                 const SamplesIteratorType& query_feature_last,
                                                 BufferType&                buffer,
                                                 const KDNodeViewPtr&       node) {
        auto parent_node = node->parent_.lock();
        // if node has a parent
        if (parent_node) {
            // get the pivot sample index in the dataset
            const auto pivot_index = parent_node->indices_range_.first[0];
            // get the split value according to the current split dimension
            const auto pivot_split_value = (*query_indexer_ptr_)[pivot_index][parent_node->cut_feature_index_];
            // get the value of the query according to the split dimension
            const auto query_split_value = query_feature_first[parent_node->cut_feature_index_];
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
                        /**/ query_feature_first,
                        /**/ query_feature_last,
                        /**/ buffer,
                        /**/ sibling_node);
                }
            }
        }
        // returns nullptr if node doesnt have parent (or is root)
        return parent_node;
    }

    IndexerPtr query_indexer_ptr_;
};

template <typename IndexerPtr, typename BufferType>
class Searcher {
  public:
    using IndexType           = typename IndexerPtr::element_type::IndexType;
    using DataType            = typename IndexerPtr::element_type::DataType;
    using IndicesIteratorType = typename IndexerPtr::element_type::IndicesIteratorType;
    using SamplesIteratorType = typename IndexerPtr::element_type::SamplesIteratorType;

  private:
    static_assert(std::is_base_of_v<knn::buffer::Base<IndicesIteratorType, SamplesIteratorType>, BufferType> ||
                      std::is_base_of_v<knn::count::Base<IndicesIteratorType, SamplesIteratorType>, BufferType>,
                  "BufferType must inherit from knn::buffer::Base<IndicesIteratorType, SamplesIteratorType> or "
                  "knn::count::Base<IndicesIteratorType, SamplesIteratorType>");

  public:
    Searcher(IndexerPtr query_indexer_ptr, const BufferType& buffer)
      : query_indexer_ptr_{query_indexer_ptr}
      , buffer_{buffer} {}

    BufferType operator()(std::size_t query_index) {
        return SingleTreeTraverser(query_indexer_ptr_)(query_index, buffer_);
    }

    BufferType operator()(const SamplesIteratorType& query_feature_first,
                          const SamplesIteratorType& query_feature_last) {
        return SingleTreeTraverser(query_indexer_ptr_)(query_feature_first, query_feature_last, buffer_);
    }

  private:
    IndexerPtr query_indexer_ptr_;
    BufferType buffer_;
};

}  // namespace ffcl::knn::count