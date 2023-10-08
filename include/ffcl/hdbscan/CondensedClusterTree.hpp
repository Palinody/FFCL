#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"
#include "ffcl/hdbscan/CondensedClusterNode.hpp"

#include <memory>

namespace ffcl {

namespace fs = std::filesystem;

template <typename IndexType, typename ValueType>
class CondensedClusterTree {
    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

  public:
    using CondensedClusterNodeType = CondensedClusterNode<IndexType, ValueType>;
    using CondensedClusterNodePtr  = std::shared_ptr<CondensedClusterNodeType>;

    using SingleLinkageClusterTreeType = SingleLinkageClusterTree<IndexType, ValueType>;
    using SingleLinkageClusterNodeType = typename SingleLinkageClusterTreeType::SingleLinkageClusterNodeType;
    using SingleLinkageClusterNodePtr  = typename SingleLinkageClusterTreeType::SingleLinkageClusterNodePtr;

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& min_cluster_size(std::size_t min_cluster_size) {
            min_cluster_size_ = min_cluster_size;
            return *this;
        }

        Options& operator=(const Options& options) {
            min_cluster_size_ = options.min_cluster_size_;
            return *this;
        }

        std::size_t min_cluster_size_ = 1;
    };

    CondensedClusterTree(SingleLinkageClusterNodePtr single_linkage_cluster_root);

    CondensedClusterTree(SingleLinkageClusterNodePtr single_linkage_cluster_root, const Options& options);

    CondensedClusterTree(const Options& options);

    CondensedClusterTree<IndexType, ValueType>& set_options(const Options& options);

    auto build(SingleLinkageClusterNodePtr single_linkage_cluster_node);

    void preorder_traversal(SingleLinkageClusterNodePtr single_linkage_cluster_node,
                            CondensedClusterNodePtr     condensed_cluster_node);

  private:
    SingleLinkageClusterNodePtr single_linkage_cluster_root_;

    Options options_;

    CondensedClusterNodePtr root_;
};

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root)
  : single_linkage_cluster_root_{single_linkage_cluster_root}
  , root_{build(single_linkage_cluster_root_)} {}

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root,
    const Options&              options)
  : single_linkage_cluster_root_{single_linkage_cluster_root}
  , root_{build(single_linkage_cluster_root)}
  , options_{options} {}

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>& CondensedClusterTree<IndexType, ValueType>::set_options(
    const Options& options) {
    options_ = options;
    return *this;
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::build(SingleLinkageClusterNodePtr single_linkage_cluster_node) {
    auto condensed_cluster_node = std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node);

    preorder_traversal(single_linkage_cluster_node, condensed_cluster_node);

    return condensed_cluster_node;
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal(
    SingleLinkageClusterNodePtr single_linkage_cluster_node,
    CondensedClusterNodePtr     condensed_cluster_node) {
    if (!single_linkage_cluster_node->is_leaf()) {
        // if the left node has at least min cluster size descendant samples
        if (single_linkage_cluster_node->left_->size() >= options_.min_cluster_size_) {
            // create a new node split
            condensed_cluster_node->left_ = std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node);
            // continue to traverse the tree with the new condensed_cluster_node and the left single linkage node
            preorder_traversal(single_linkage_cluster_node->left_, condensed_cluster_node->left_);

        } else {
            // update the stability of the same condensed cluster node with maybe a few less samples
            condensed_cluster_node->update_stability(single_linkage_cluster_node->left_->level_);
            // continue to traverse the tree with the same condensed_cluster_node and the left single linkage node
            preorder_traversal(single_linkage_cluster_node->left_, condensed_cluster_node);
        }
        // if the right node has at least min cluster size descendant samples
        if (single_linkage_cluster_node->right_->size() >= options_.min_cluster_size_) {
            // create a new node split
            condensed_cluster_node->right_ = std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node);
            // continue to traverse the tree with the new condensed_cluster_node and the right single linkage node
            preorder_traversal(single_linkage_cluster_node->right_, condensed_cluster_node->right_);

        } else {
            // update the stability of the same condensed cluster node with maybe a few less samples
            condensed_cluster_node->update_stability(single_linkage_cluster_node->right_->level_);
            // continue to traverse the tree with the same condensed_cluster_node and the right single linkage node
            preorder_traversal(single_linkage_cluster_node->right_, condensed_cluster_node);
        }
    } else {
        condensed_cluster_node->update_stability(single_linkage_cluster_node->level_);
    }
}

}  // namespace ffcl