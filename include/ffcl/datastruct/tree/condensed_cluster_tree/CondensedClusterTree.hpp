#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/tree/condensed_cluster_tree/CondensedClusterNode.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"
#include "ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"

#include <map>
#include <memory>
#include <type_traits>

namespace ffcl::datastruct {

namespace fs = std::filesystem;

template <typename Index, typename Value>
class CondensedClusterTree {
  public:
    using IndexType = Index;
    using ValueType = Value;

    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

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

        Options& allow_single_cluster(bool allow_single_cluster) {
            allow_single_cluster_ = allow_single_cluster;
            return *this;
        }

        Options& return_leaf_nodes(bool return_leaf_nodes) {
            return_leaf_nodes_ = return_leaf_nodes;
            return *this;
        }

        Options& operator=(const Options& options) {
            min_cluster_size_     = options.min_cluster_size_;
            allow_single_cluster_ = options.allow_single_cluster_;
            return_leaf_nodes_    = options.return_leaf_nodes_;
            return *this;
        }

        std::size_t min_cluster_size_     = 1;
        bool        allow_single_cluster_ = false;
        bool        return_leaf_nodes_    = false;
    };

    CondensedClusterTree(SingleLinkageClusterNodePtr single_linkage_cluster_root);

    CondensedClusterTree(SingleLinkageClusterNodePtr single_linkage_cluster_root, const Options& options);

    CondensedClusterTree(const Options& options);

    auto extract_flat_clusters() const;

    auto predict() const;

  private:
    auto build(const SingleLinkageClusterNodePtr& slink_root_node);

    void preorder_traversal_single_linkage_clustering(const ClusterIndexType&            cluster_label,
                                                      const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
                                                      std::vector<ClusterIndexType>&     flat_cluster) const;

    void preorder_traversal_build(CondensedClusterNodePtr            condensed_cluster_node,
                                  const SingleLinkageClusterNodePtr& single_linkage_cluster_node);

    auto select_subtree(CondensedClusterNodePtr condensed_cluster_node);

    auto return_shallowest_cluster_selection() const;

    void preorder_traversal_fill_shallowest_selected_nodes(
        const CondensedClusterNodePtr&        condensed_cluster_node,
        std::vector<CondensedClusterNodePtr>& selected_condensed_cluster_nodes) const;

    Options options_;

    CondensedClusterNodePtr root_;
};

// Class template argument deduction (CTAD)
template <typename SingleLinkageClusterNodePtr>
CondensedClusterTree(SingleLinkageClusterNodePtr)
    -> CondensedClusterTree<typename std::remove_pointer_t<SingleLinkageClusterNodePtr>::IndexType,
                            typename std::remove_pointer_t<SingleLinkageClusterNodePtr>::ValueType>;

template <typename SingleLinkageClusterNodePtr, typename Options>
CondensedClusterTree(SingleLinkageClusterNodePtr, const Options&)
    -> CondensedClusterTree<typename std::remove_pointer_t<SingleLinkageClusterNodePtr>::IndexType,
                            typename std::remove_pointer_t<SingleLinkageClusterNodePtr>::ValueType>;

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root)
  : CondensedClusterTree(single_linkage_cluster_root, Options{}) {}

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root,
    const Options&              options)
  : options_{options}
  , root_{build(single_linkage_cluster_root)} {
    // Don't perform cluster selection if the leaf nodes are requested.
    // The build already marks the leaves as selected.
    if (!options_.return_leaf_nodes_) {
        select_subtree(root_);

        // unselect the root condensed cluster node by default if it shouldn't be allowed
        // this won't work if the root node is also a leaf node
        if (!options_.allow_single_cluster_ && !root_->is_leaf()) {
            root_->is_selected() = false;
        }
    }
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::build(const SingleLinkageClusterNodePtr& slink_root_node) {
    auto condensed_cluster_root_node = std::make_shared<CondensedClusterNodeType>(slink_root_node);

    preorder_traversal_build(condensed_cluster_root_node, slink_root_node);

    return condensed_cluster_root_node;
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal_build(
    CondensedClusterNodePtr            condensed_cluster_node,
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node) {
    if (!single_linkage_cluster_node->is_leaf()) {
        const bool is_left_child_split_candidate =
            single_linkage_cluster_node->left_->size() >= options_.min_cluster_size_;

        const bool is_right_child_split_candidate =
            single_linkage_cluster_node->right_->size() >= options_.min_cluster_size_;

        // the lambda value needs to be computed for any of the if/else branches
        const auto lambda_value = common::division(1, single_linkage_cluster_node->level_);

        // if both children are split candidates (they hold enough samples to be considered as their own cluster), we
        // consider the event as a true split and split the condensed cluster node in two new condensed cluster nodes
        if (is_left_child_split_candidate && is_right_child_split_candidate) {
            // update the stability of the current condensed cluster node with the final slink node
            condensed_cluster_node->add_excess_of_mass(single_linkage_cluster_node, lambda_value);
            // make the new left condensed cluster node
            {
                // create a new left cluster node split
                condensed_cluster_node->left_ =
                    std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node->left_);
                // link the new created left branch node to the current node
                condensed_cluster_node->left_->parent_ = condensed_cluster_node;
                // continue to traverse the tree with the new condensed_cluster_node and the left single linkage node
                // that didn't fall out of the cluster
                preorder_traversal_build(condensed_cluster_node->left_, single_linkage_cluster_node->left_);
            }
            // make the new right condensed cluster node
            {
                // create a new right cluster node split
                condensed_cluster_node->right_ =
                    std::make_shared<CondensedClusterNodeType>(single_linkage_cluster_node->right_);
                // link the new created right branch node to the current node
                condensed_cluster_node->right_->parent_ = condensed_cluster_node;
                // continue to traverse the tree with the new condensed_cluster_node and the right single linkage node
                // that didn't fall out of the cluster
                preorder_traversal_build(condensed_cluster_node->right_, single_linkage_cluster_node->right_);
            }
        }
        // if only the left child is a split candidate, it persists in the current condensed cluster node and the right
        // child is simply discarded (it "falls out of the cluster")
        else if (is_left_child_split_candidate) {
            // update the condensed cluster node with the samples that fell out from the cluster at the current level
            condensed_cluster_node->add_excess_of_mass(single_linkage_cluster_node->right_, lambda_value);
            // continue to traverse the tree with the left single linkage node that didn't fall out of the cluster
            // in the same condensed cluster node
            preorder_traversal_build(condensed_cluster_node, single_linkage_cluster_node->left_);
        }
        // if only the right child is a split candidate, it persists in the current condensed cluster node and the left
        // child is simply discarded (it "falls out of the cluster")
        else if (is_right_child_split_candidate) {
            // update the condensed cluster node with the samples that fell out from the cluster at the current level
            condensed_cluster_node->add_excess_of_mass(single_linkage_cluster_node->left_, lambda_value);
            // continue to traverse the tree with the right single linkage node that didn't fall out of the cluster
            // in the same condensed cluster node
            preorder_traversal_build(condensed_cluster_node, single_linkage_cluster_node->right_);
        }
        // if none of the children are split candidates, then the tree branch is terminated
        else {
            // update the condensed cluster node with the samples that fell out from the cluster at the current level
            condensed_cluster_node->add_excess_of_mass(single_linkage_cluster_node, lambda_value);
            // The condensed cluster node can finally be considered a leaf node if no children are splitting
            // candidates. All the samples descendant to the node split fall out of the cluster and thus terminate
            // the tree build.
            condensed_cluster_node->is_selected() = true;
        }
    }
    // if single_linkage_cluster_node is a leaf node,
    else {
        // theres no need to compute the excess of mass since falling in that branch means that either the parent
        // node already added the right excess of mass or the condensed cluster node is root. In the latter case,
        // there's no need to compute the excess of mass so we just flag the condensed cluster as selected
        condensed_cluster_node->is_selected() = true;
    }
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::extract_flat_clusters() const {
    // the flat cluster represented as an array vector of labels mapping each sample to their respective cluster label
    auto flat_cluster = std::vector<ClusterIndexType>(root_->size());
    // The cluster hierarchy for each condensed cluster node will be assigned consecutive cluster labels beginning from
    // 1, with 0 denoting noise.
    ClusterIndexType cluster_label = 1;
    for (const auto& condensed_cluster_node : return_shallowest_cluster_selection()) {
        // assign all descendant samples in the node hierarchy with the same cluster label
        // then increment the cluster label for the next non-overlapping cluster
        preorder_traversal_single_linkage_clustering(cluster_label++,
                                                     /**/ condensed_cluster_node->single_linkage_cluster_node_root_,
                                                     /**/ flat_cluster);
    }
    return flat_cluster;
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::predict() const {
    return extract_flat_clusters();
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal_single_linkage_clustering(
    const ClusterIndexType&            cluster_label,
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    std::vector<ClusterIndexType>&     flat_cluster) const {
    // continue to traverse the tree if the current node is not leaf
    // a single linkage cluster node is guaranteed to have a left and a right child if its not leaf
    if (!single_linkage_cluster_node->is_leaf()) {
        preorder_traversal_single_linkage_clustering(cluster_label, single_linkage_cluster_node->left_, flat_cluster);
        preorder_traversal_single_linkage_clustering(cluster_label, single_linkage_cluster_node->right_, flat_cluster);

    } else {
        // assert that single_linkage_cluster_node is indeed a leaf  by checking if its level in the tree is zero
        assert(common::equality(single_linkage_cluster_node->level_, 0));

        // assign the cluster label to the sample index (which is its own node at level 0)
        flat_cluster[single_linkage_cluster_node->representative_] = cluster_label;
    }
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::select_subtree(CondensedClusterNodePtr condensed_cluster_node) {
    // return the stability of the node directly if the node query is leaf
    if (condensed_cluster_node->is_leaf()) {
        return condensed_cluster_node->stability_;

    } else {
        // compute the stability of the children of the current node, recursively
        const auto children_stability =
            select_subtree(condensed_cluster_node->left_) + select_subtree(condensed_cluster_node->right_);
        // If the sum of the stabilities of the child clusters is greater than the stability of the cluster, then we set
        // the cluster stability to be the sum of the child stabilities and the cluster remains unselected
        if (children_stability > condensed_cluster_node->stability_) {
            return children_stability;

        } else {
            // If, on the other hand, the cluster’s stability is greater than the sum of its children then we declare
            // the cluster to be a selected cluster
            condensed_cluster_node->is_selected() = true;
            // and we return its stability
            return condensed_cluster_node->stability_;
        }
    }
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::return_shallowest_cluster_selection() const {
    // an array vector of CondensedClusterNodePtr containing the shallowest non overlapping nodes marked as selected
    std::vector<CondensedClusterNodePtr> selected_condensed_cluster_nodes;

    preorder_traversal_fill_shallowest_selected_nodes(root_, selected_condensed_cluster_nodes);

    return selected_condensed_cluster_nodes;
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal_fill_shallowest_selected_nodes(
    const CondensedClusterNodePtr&        condensed_cluster_node,
    std::vector<CondensedClusterNodePtr>& selected_condensed_cluster_nodes) const {
    // if condensed_cluster_node is selected, we simply add it to the array vector and exit the function
    if (condensed_cluster_node->is_selected()) {
        selected_condensed_cluster_nodes.emplace_back(condensed_cluster_node);
    }
    // else we still need to recurse the tree until we end up finding the first selected node of the branch
    else {
        preorder_traversal_fill_shallowest_selected_nodes(condensed_cluster_node->left_,
                                                          selected_condensed_cluster_nodes);

        preorder_traversal_fill_shallowest_selected_nodes(condensed_cluster_node->right_,
                                                          selected_condensed_cluster_nodes);
    }
}

}  // namespace ffcl::datastruct