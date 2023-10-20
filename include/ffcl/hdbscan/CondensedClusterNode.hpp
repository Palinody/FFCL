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

    void accumulate_stability(const ValueType& level);

    // the single linkage cluster node pointer that led to a split of the single linkage tree.
    SingleLinkageClusterNodePtr single_linkage_cluster_node_;
    // the initial lambda value: 1 / distance. It results from the creation of the current node after a split.
    ValueType lambda_init_, lambda_final_;
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
  : single_linkage_cluster_node_{single_linkage_cluster_node}
  , lambda_init_{common::utils::division(1,
                                         single_linkage_cluster_node_->level_,
                                         common::utils::infinity<decltype(single_linkage_cluster_node_->level_)>())}
  , lambda_final_{lambda_init_}
  , stability_{0}
  , is_selected_{false} {}

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
    return single_linkage_cluster_node_->cluster_size_;
}

template <typename IndexType, typename ValueType>
void CondensedClusterNode<IndexType, ValueType>::accumulate_stability(const ValueType& level) {
    // the division performs
    //      1 / level if level != 0,
    //      else it returns lambda_init_
    // so the operation is cancelled out in case we encounter a division by zero
    stability_ += common::utils::division(1, level, lambda_init_) - lambda_init_;
}

}  // namespace ffcl