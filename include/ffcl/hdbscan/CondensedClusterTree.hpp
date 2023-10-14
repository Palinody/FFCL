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

    auto extract_flat_cluster() const;

  private:
    auto build(const SingleLinkageClusterNodePtr& single_linkage_cluster_node);

    void preorder_traversal_assign_cluster_label_to_node(const ClusterIndexType&        cluster_label,
                                                         const CondensedClusterNodePtr& condensed_cluster_node,
                                                         std::vector<ClusterIndexType>& flat_cluster) const;

    void single_linkage_preorder_traversal_clustering(const ClusterIndexType&        cluster_label,
                                                      SingleLinkageClusterNodePtr    single_linkage_cluster_node,
                                                      std::vector<ClusterIndexType>& flat_cluster) const;

    void preorder_traversal_build(SingleLinkageClusterNodePtr single_linkage_cluster_node,
                                  CondensedClusterNodePtr     condensed_cluster_node);

    void select_condensed_cluster_nodes();

    Options options_;

    std::vector<CondensedClusterNodePtr> condensed_cluster_node_leaves_;

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
  , condensed_cluster_node_leaves_{}
  , root_{build(single_linkage_cluster_root)} {
    // select_condensed_cluster_nodes();
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::build(const SingleLinkageClusterNodePtr& single_linkage_cluster_node) {
    auto condensed_cluster_node = std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node);

    preorder_traversal_build(single_linkage_cluster_node, condensed_cluster_node);

    return condensed_cluster_node;
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal_build(
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
            preorder_traversal_build(single_linkage_cluster_node->left_, condensed_cluster_node->left_);

            // create a new right cluster node split
            condensed_cluster_node->right_ =
                std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node->right_);
            // link the new created right branch node to the current node
            condensed_cluster_node->right_->parent_ = condensed_cluster_node;
            // continue to traverse the tree with the new condensed_cluster_node and the right single linkage node that
            // didn't fall out of the cluster
            preorder_traversal_build(single_linkage_cluster_node->right_, condensed_cluster_node->right_);

        } else if (is_left_child_split_candidate) {
            // update the stability of the same condensed cluster node with maybe a few less samples
            condensed_cluster_node->update_stability(single_linkage_cluster_node->left_->level_);
            // continue to traverse the tree with the left single linkage node that didn't fall out of the cluster
            // in the same condensed cluster node
            preorder_traversal_build(single_linkage_cluster_node->left_, condensed_cluster_node);

        } else if (is_right_child_split_candidate) {
            // update the stability of the same condensed cluster node with maybe a few less samples
            condensed_cluster_node->update_stability(single_linkage_cluster_node->right_->level_);
            // continue to traverse the tree with the right single linkage node that didn't fall out of the cluster
            // in the same condensed cluster node
            preorder_traversal_build(single_linkage_cluster_node->right_, condensed_cluster_node);

        } else {
            // The condensed cluster node can finally be considered a leaf node if no children are splitting candidates.
            // All the samples descendant to the node split fall out of the cluster and thus terminate the tree build.
            condensed_cluster_node->is_selected() = true;
            // add the leaf node to the set of other leaf nodes
            condensed_cluster_node_leaves_.emplace_back(condensed_cluster_node);
        }
    }
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal_assign_cluster_label_to_node(
    const ClusterIndexType&        cluster_label,
    const CondensedClusterNodePtr& condensed_cluster_node,
    std::vector<ClusterIndexType>& flat_cluster) const {
    // continue to traverse the tree if the current node is not leaf
    // a single linkage cluster node is guaranteed to have a left and a right child if its not leaf
    if (!condensed_cluster_node->is_leaf()) {
        preorder_traversal_assign_cluster_label_to_node(cluster_label, condensed_cluster_node->left_, flat_cluster);
        preorder_traversal_assign_cluster_label_to_node(cluster_label, condensed_cluster_node->right_, flat_cluster);

    } else {
        // assign the cluster label to the sample index (which is its own node at level 0)
        flat_cluster[condensed_cluster_node->single_linkage_cluster_node->representative_] = cluster_label;
    }
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::extract_flat_cluster() const {
    auto flat_cluster = std::vector<ClusterIndexType>(root_->size());
    // The cluster hierarchy for each condensed cluster node will be assigned consecutive cluster labels beginning from
    // 1, with 0 denoting noise.
    ClusterIndexType cluster_label = 1;
    for (const auto& condensed_cluster_node_leaf : condensed_cluster_node_leaves_) {
        // assign all descendant samples in the node hierarchy with the same cluster label
        single_linkage_preorder_traversal_clustering(
            cluster_label++, condensed_cluster_node_leaf->single_linkage_cluster_node_, flat_cluster);
    }
    return flat_cluster;
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::single_linkage_preorder_traversal_clustering(
    const ClusterIndexType&        cluster_label,
    SingleLinkageClusterNodePtr    single_linkage_cluster_node,
    std::vector<ClusterIndexType>& flat_cluster) const {
    // continue to traverse the tree if the current node is not leaf
    // a single linkage cluster node is guaranteed to have a left and a right child if its not leaf
    if (!single_linkage_cluster_node->is_leaf()) {
        single_linkage_preorder_traversal_clustering(cluster_label, single_linkage_cluster_node->left_, flat_cluster);
        single_linkage_preorder_traversal_clustering(cluster_label, single_linkage_cluster_node->right_, flat_cluster);

    } else {
        // assign the cluster label to the sample index (which is its own node at level 0)
        flat_cluster[single_linkage_cluster_node->representative_] = cluster_label;
    }
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
    /*
    std::map<CondensedClusterNodePtr, ValueType> condensed_cluster_node_candidates;

    for (const auto& condensed_cluster_leaf_node : condensed_cluster_node_leaves_) {
        condensed_cluster_node_candidates.insert(
            std::make_pair(condensed_cluster_leaf_node, condensed_cluster_leaf_node->stability_));
    }
    while (!condensed_cluster_node_candidates.empty()) {
        std::map<CondensedClusterNodePtr, ValueType> next_condensed_cluster_node_candidates;

        for (const auto& node : condensed_cluster_node_candidates) {
            if (!node->is_leaf()) {
                const auto children_total_stability = condensed_cluster_node_candidates[node->left_]->stability_ +
                                                      condensed_cluster_node_candidates[node->right_]->stability_;

                if (node->stability_ < children_total_stability) {
                    next_condensed_cluster_node_candidates.insert(std::make_pair(node, children_total_stability));

                } else {
                    next_condensed_cluster_node_candidates.insert(std::make_pair(node, node->stability_));
                }
            } else {
                next_condensed_cluster_node_candidates.insert(std::make_pair(node->parent_, node->stability_));
            }
        }
        condensed_cluster_node_candidates = std::move(next_condensed_cluster_node_candidates);
    }
    */
}

}  // namespace ffcl