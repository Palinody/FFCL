#pragma once

#include "ffcl/common/Utils.hpp"

#include <memory>

namespace ffcl {

template <typename IndexType, typename ValueType>
struct CondensedClusterNode {
    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

    using NodeType = CondensedClusterNode<IndexType, ValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    CondensedClusterNode(const IndexType& representative, const ValueType& level = 0, std::size_t cluster_size = 1);

    bool is_leaf() const;

    std::size_t size() const;

    // the number of nodes that this node is an ancestor of (counting itself)
    std::size_t cluster_size_;
    // parent pointer used to parse from the leaves to the root and the left/right ones for the opposite direction
    NodePtr parent_, left_, right_;
};

}  // namespace ffcl