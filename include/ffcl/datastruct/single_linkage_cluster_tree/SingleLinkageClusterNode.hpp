#pragma once

#include "ffcl/common/Utils.hpp"

#include <cstdio>
#include <memory>
#include <numeric>

#include "rapidjson/writer.h"

namespace ffcl {

template <typename IndexType, typename ValueType>
struct SingleLinkageClusterNode {
    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

    using NodeType = SingleLinkageClusterNode<IndexType, ValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    SingleLinkageClusterNode(const IndexType& representative, const ValueType& level = 0, std::size_t cluster_size = 1);

    bool is_leaf() const;

    std::size_t size() const;

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    // a node indexes itself if the node is leaf or the sample's index used as a cluster index representative otherwise
    IndexType representative_;
    // the distance that separates this node's left and right children and which represents its hight in the tree
    ValueType level_;
    // the number of nodes that this node is an ancestor of (counting itself)
    std::size_t cluster_size_;
    // parent pointer used to parse from the leaves to the root and the left/right ones for the opposite direction
    NodePtr parent_, left_, right_;
};

template <typename IndexType, typename ValueType>
SingleLinkageClusterNode<IndexType, ValueType>::SingleLinkageClusterNode(const IndexType& representative,
                                                                         const ValueType& level,
                                                                         std::size_t      cluster_size)
  : representative_{representative}
  , level_{level}
  , cluster_size_{cluster_size} {}

template <typename IndexType, typename ValueType>
bool SingleLinkageClusterNode<IndexType, ValueType>::is_leaf() const {
    // could do level_ == 0 but that might require performing float equality
    return left_ == nullptr && right_ == nullptr;
}

template <typename IndexType, typename ValueType>
std::size_t SingleLinkageClusterNode<IndexType, ValueType>::size() const {
    return cluster_size_;
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterNode<IndexType, ValueType>::serialize(
    rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    static_assert(std::is_floating_point_v<ValueType> || std::is_integral_v<ValueType>,
                  "Unsupported type during kdnode serialization");

    writer.String("representative");
    writer.Int64(representative_);

    writer.String("level");
    if constexpr (std::is_integral_v<ValueType>) {
        writer.Int64(level_);

    } else if constexpr (std::is_floating_point_v<ValueType>) {
        writer.Double(level_);
    }
    writer.String("cluster_size");
    writer.Int64(cluster_size_);
}

}  // namespace ffcl