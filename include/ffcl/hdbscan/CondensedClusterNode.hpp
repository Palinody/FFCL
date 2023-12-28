#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterNode.hpp"

#include <memory>

namespace ffcl {

template <typename Index, typename Value>
struct CondensedClusterNode {
    static_assert(std::is_fundamental<Index>::value, "Index must be a fundamental type.");
    static_assert(std::is_fundamental<Value>::value, "Value must be a fundamental type.");

    using NodeType = CondensedClusterNode<Index, Value>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using SingleLinkageClusterNodeType = SingleLinkageClusterNode<Index, Value>;
    using SingleLinkageClusterNodePtr  = typename SingleLinkageClusterNodeType::NodePtr;

    CondensedClusterNode(SingleLinkageClusterNodePtr single_linkage_cluster_node);

    bool is_leaf() const;

    bool& is_selected();

    bool is_selected() const;

    std::size_t size() const;

    void add_excess_of_mass(const SingleLinkageClusterNodePtr& single_linkage_cluster_node, const Value& lambda_value);

    // the single linkage cluster node pointer that led to a split of the single linkage tree.
    SingleLinkageClusterNodePtr single_linkage_cluster_node_root_;
    // the initial lambda value: 1 / distance. It results from the creation of the current node after a split.
    Value lambda_min_;
    // the total accumulated lambda values for each points persisting in the same cluster.
    // stability: sum(lambda(p) - lambda_init), with p a point in the cluster.
    Value stability_;
    // whether this node is marked as selected. Used for flat cluster extraction algorithm and might not be the actual
    // selected cluster if marked as true if shallower cluster nodes are also marked as selected.
    bool is_selected_;
    // parent pointer used to parse from the leaves to the root and the left/right ones for the opposite direction
    NodePtr parent_, left_, right_;
};

template <typename Index, typename Value>
CondensedClusterNode<Index, Value>::CondensedClusterNode(SingleLinkageClusterNodePtr single_linkage_cluster_node)
  : single_linkage_cluster_node_root_{single_linkage_cluster_node}
  , lambda_min_{single_linkage_cluster_node_root_->has_parent()
                    ? common::division(1, single_linkage_cluster_node_root_->parent_->level_)
                    : 0}
  , stability_{0}
  , is_selected_{false} {}

template <typename Index, typename Value>
bool CondensedClusterNode<Index, Value>::is_leaf() const {
    return left_ == nullptr && right_ == nullptr;
}

template <typename Index, typename Value>
bool& CondensedClusterNode<Index, Value>::is_selected() {
    return is_selected_;
}

template <typename Index, typename Value>
bool CondensedClusterNode<Index, Value>::is_selected() const {
    return is_selected_;
}

template <typename Index, typename Value>
std::size_t CondensedClusterNode<Index, Value>::size() const {
    // takes into account only the initiating node of the branch that persists along different lambda values
    return single_linkage_cluster_node_root_->size();
}

template <typename Index, typename Value>
void CondensedClusterNode<Index, Value>::add_excess_of_mass(
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    const Value&                       lambda_value) {
    stability_ += single_linkage_cluster_node->size() * (lambda_value - lambda_min_);
}

}  // namespace ffcl