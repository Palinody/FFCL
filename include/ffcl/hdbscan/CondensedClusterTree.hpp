#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"
#include "ffcl/hdbscan/CondensedClusterNode.hpp"

#include <map>
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

    using ClusterIndexType = IndexType;

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

  private:
    auto build(SingleLinkageClusterNodePtr single_linkage_cluster_node);

    void preorder_traversal(SingleLinkageClusterNodePtr single_linkage_cluster_node,
                            CondensedClusterNodePtr     condensed_cluster_node);

    void select_condensed_cluster_nodes();

    Options options_;

    SingleLinkageClusterNodePtr single_linkage_cluster_root_;

    std::vector<CondensedClusterNodePtr> leaf_condensed_cluster_nodes_;

    CondensedClusterNodePtr root_;
};

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root)
  : CondensedClusterTree(single_linkage_cluster_root, Options{}) {}

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root,
    const Options&              options)
  : options_{options}
  , single_linkage_cluster_root_{single_linkage_cluster_root}
  , leaf_condensed_cluster_nodes_{}
  , root_{build(single_linkage_cluster_root)} {
    select_condensed_cluster_nodes();
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
        const bool is_left_child_split_candidate =
            single_linkage_cluster_node->left_->size() >= options_.min_cluster_size_;

        const bool is_right_child_split_candidate =
            single_linkage_cluster_node->right_->size() >= options_.min_cluster_size_;

        // if both children are split candidates, we consider the event as a true split and split the condensed cluster
        // node in two new condensed cluster nodes
        if (is_left_child_split_candidate && is_right_child_split_candidate) {
            // create a new left cluster node split
            condensed_cluster_node->left_ =
                std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node->left_);
            // link the new created left branch node to the current node
            condensed_cluster_node->left_->parent_ = condensed_cluster_node;
            // continue to traverse the tree with the new condensed_cluster_node and the left single linkage node that
            // didn't fall out of the cluster
            preorder_traversal(single_linkage_cluster_node->left_, condensed_cluster_node->left_);

            // create a new right cluster node split
            condensed_cluster_node->right_ =
                std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node->right_);
            // link the new created right branch node to the current node
            condensed_cluster_node->right_->parent_ = condensed_cluster_node;
            // continue to traverse the tree with the new condensed_cluster_node and the right single linkage node that
            // didn't fall out of the cluster
            preorder_traversal(single_linkage_cluster_node->right_, condensed_cluster_node->right_);

        } else if (is_left_child_split_candidate) {
            // update the stability of the same condensed cluster node with maybe a few less samples
            condensed_cluster_node->update_stability(single_linkage_cluster_node->left_->level_);
            // continue to traverse the tree with the same condensed_cluster_node and the left single linkage node that
            // didn't fall out of the cluster
            preorder_traversal(single_linkage_cluster_node->left_, condensed_cluster_node);

        } else if (is_right_child_split_candidate) {
            // update the stability of the same condensed cluster node with maybe a few less samples
            condensed_cluster_node->update_stability(single_linkage_cluster_node->right_->level_);
            // continue to traverse the tree with the same condensed_cluster_node and the right single linkage node that
            // didn't fall out of the cluster
            preorder_traversal(single_linkage_cluster_node->right_, condensed_cluster_node);
        }
    }
    // the condensed cluster tree can finally be considered a leaf node if no children are splitting candidates
    condensed_cluster_node->is_selected() = true;
    // add the leaf node to the set of other leaf nodes
    leaf_condensed_cluster_nodes_.emplace_back(condensed_cluster_node);
}

/*
Declare all leaf nodes to be selected clusters. Now work up through the tree (the reverse topological sort order). If
the sum of the stabilities of the child clusters is greater than the stability of the cluster, then we set the cluster
stability to be the sum of the child stabilities. If, on the other hand, the cluster’s stability is greater than the sum
of its children then we declare the cluster to be a selected cluster and unselect all its descendants. Once we reach the
root node we call the current set of selected clusters our flat clustering and return that.
*/
template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::select_condensed_cluster_nodes() {
    // find the condensed node with the deepest single linkage cluster
    auto deepest_leaf_node = *std::min_element(leaf_condensed_cluster_nodes_.begin(),
                                               leaf_condensed_cluster_nodes_.end(),
                                               [&](const auto& node_1, const auto& node_2) {
                                                   return node_1->single_linkage_cluster_node_->level_ <
                                                          node_2->single_linkage_cluster_node_->level_;
                                               });

    const auto deepest_level = deepest_leaf_node->single_linkage_cluster_node_->level_;

    std::cout << "Deepest node level: " << deepest_level << "\n";

    std::cout << "node level:\n";
    for (const auto& leaf_node : leaf_condensed_cluster_nodes_) {
        std::cout << (deepest_level >= leaf_node->single_linkage_cluster_node_->level_) << ",";
    }
    std::cout << "\n";

    std::cout << "leaf_condensed_cluster_nodes_ size: " << leaf_condensed_cluster_nodes_.size() << "\n";
}

}  // namespace ffcl