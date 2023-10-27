#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterNode.hpp"

#include <memory>

namespace ffcl {

template <typename IndexType, typename ValueType>
struct CondensedClusterNode {
    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

    using NodeType = CondensedClusterNode<IndexType, ValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using SingleLinkageClusterNodeType = SingleLinkageClusterNode<IndexType, ValueType>;
    using SingleLinkageClusterNodePtr  = typename SingleLinkageClusterNodeType::NodePtr;

    CondensedClusterNode(SingleLinkageClusterNodePtr single_linkage_cluster_node);

    bool is_leaf() const;

    bool& is_selected();

    bool is_selected() const;

    std::size_t size() const;

    void accumulate_excess_stability(const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
                                     const ValueType&                   lambda_max);

    void accumulate_stability_from_node(const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
                                        const ValueType&                   lambda_max);

    void accumulate_fallen_clusters_stability(const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
                                              const ValueType&                   lambda_max);

    // the single linkage cluster node pointer that led to a split of the single linkage tree.
    SingleLinkageClusterNodePtr single_linkage_cluster_node_min_, single_linkage_cluster_node_max_;
    // the initial lambda value: 1 / distance. It results from the creation of the current node after a split.
    ValueType lambda_min_;
    // the total accumulated lambda values for each points persisting in the same cluster.
    // stability: sum(lambda(p) - lambda_init), with p a point in the cluster.
    ValueType stability_;
    // whether this node is marked as selected. Used for flat cluster extraction algorithm and might not be the actual
    // selected cluster if marked as true if shallower cluster nodes are also marked as selected.
    bool is_selected_;
    // parent pointer used to parse from the leaves to the root and the left/right ones for the opposite direction
    NodePtr parent_, left_, right_;
};

template <typename IndexType, typename ValueType>
CondensedClusterNode<IndexType, ValueType>::CondensedClusterNode(
    SingleLinkageClusterNodePtr single_linkage_cluster_node)
  : single_linkage_cluster_node_min_{single_linkage_cluster_node}
  , single_linkage_cluster_node_max_{}
  , lambda_min_{single_linkage_cluster_node_min_->has_parent()
                    ? common::utils::division(1, single_linkage_cluster_node_min_->parent_->level_)
                    : 0}
  , stability_{0}
  , is_selected_{false} {
    // Calculate the stability of the current cluster node. The stability may increase over time
    // as the cluster node persists as we build the tree.
    // accumulate_excess_stability(single_linkage_cluster_node_min_,
    // common::utils::division(1, single_linkage_cluster_node_min_->level_));
}

template <typename IndexType, typename ValueType>
bool CondensedClusterNode<IndexType, ValueType>::is_leaf() const {
    return left_ == nullptr && right_ == nullptr;
}

template <typename IndexType, typename ValueType>
bool& CondensedClusterNode<IndexType, ValueType>::is_selected() {
    return is_selected_;
}

template <typename IndexType, typename ValueType>
bool CondensedClusterNode<IndexType, ValueType>::is_selected() const {
    return is_selected_;
}

template <typename IndexType, typename ValueType>
std::size_t CondensedClusterNode<IndexType, ValueType>::size() const {
    // takes into account only the initiating node of the branch that persists along different lambda values
    return single_linkage_cluster_node_min_->cluster_size_;
}

template <typename IndexType, typename ValueType>
void CondensedClusterNode<IndexType, ValueType>::accumulate_excess_stability(
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    const ValueType&                   lambda_max) {
    // the division performs
    //      1 / level if level != 0,
    //      else it returns lambda_min_
    // so the operation is cancelled out in case we encounter a division by zero
    stability_ += single_linkage_cluster_node->size() * (lambda_max - lambda_min_);
}

/*
template <typename IndexType, typename ValueType>
void CondensedClusterNode<IndexType, ValueType>::accumulate_stability_from_node(
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    const ValueType&                   lambda_max) {
    if (single_linkage_cluster_node == single_linkage_cluster_node_min_) {
        if (!single_linkage_cluster_node->is_leaf()) {
            accumulate_stability_from_node(single_linkage_cluster_node->left_, lambda_max);
            accumulate_stability_from_node(single_linkage_cluster_node->right_, lambda_max);
        }
        // if single_linkage_cluster_node->is_leaf(), the stability will be 0 so we skip the computation
    } else {
        if (single_linkage_cluster_node->is_leaf()) {
            const auto current_lambda = common::utils::division(1, single_linkage_cluster_node->parent_->level_);

            // stability_ += current_lambda < lambda_max ? current_lambda - lambda_min_ : lambda_max - lambda_min_;
            stability_ += current_lambda - lambda_min_;

        } else {
            const auto current_lambda = common::utils::division(1, single_linkage_cluster_node->level_);

            if (current_lambda < lambda_max) {
                accumulate_stability_from_node(single_linkage_cluster_node->left_, lambda_max);
                accumulate_stability_from_node(single_linkage_cluster_node->right_, lambda_max);

            } else {
                accumulate_excess_stability(single_linkage_cluster_node, lambda_max);
            }
        }
    }
}
*/

template <typename IndexType, typename ValueType>
void CondensedClusterNode<IndexType, ValueType>::accumulate_stability_from_node(
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    const ValueType&                   lambda_max) {
    if (single_linkage_cluster_node->is_leaf()) {
        const auto current_lambda = common::utils::division(1, single_linkage_cluster_node->parent_->level_);

        stability_ += current_lambda - lambda_min_;

    } else {
        accumulate_stability_from_node(single_linkage_cluster_node->left_, lambda_max);
        accumulate_stability_from_node(single_linkage_cluster_node->right_, lambda_max);
    }
}

/*
template <typename IndexType, typename ValueType>
void CondensedClusterNode<IndexType, ValueType>::accumulate_fallen_clusters_stability(
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    const ValueType&                   lambda_max) {
    if (single_linkage_cluster_node != single_linkage_cluster_node_min_) {
        accumulate_stability_from_node(single_linkage_cluster_node->get_sibling_node(), lambda_max);
        accumulate_fallen_clusters_stability(single_linkage_cluster_node->parent_, lambda_max);
    }
}
*/

}  // namespace ffcl