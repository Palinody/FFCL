#pragma once

#include "ffcl/common/Utils.hpp"

#include <cstdio>
#include <memory>
#include <numeric>

#include "rapidjson/writer.h"

namespace ffcl::datastruct {

template <typename Index, typename Value>
struct SingleLinkageClusterNode {
    static_assert(std::is_fundamental<Index>::value, "Index must be a fundamental type.");
    static_assert(std::is_fundamental<Value>::value, "Value must be a fundamental type.");

    using NodeType = SingleLinkageClusterNode<Index, Value>;
    using NodePtr  = std::shared_ptr<NodeType>;

    SingleLinkageClusterNode(const Index& representative, const Value& level = 0, std::size_t cluster_size = 1);

    bool is_leaf() const;

    std::size_t size() const;

    bool has_parent() const;

    NodePtr get_sibling_node() const;

    void serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    // a node indexes itself if the node is leaf or the sample's index used as a cluster index representative otherwise
    Index representative_;
    // the distance that separates this node's left and right children and which represents its hight in the tree
    Value level_;
    // the number of nodes that this node is an ancestor of (counting itself)
    std::size_t cluster_size_;
    // parent pointer used to parse from the leaves to the root and the left/right ones for the opposite direction
    NodePtr parent_, left_, right_;
};

template <typename Index, typename Value>
SingleLinkageClusterNode<Index, Value>::SingleLinkageClusterNode(const Index& representative,
                                                                 const Value& level,
                                                                 std::size_t  cluster_size)
  : representative_{representative}
  , level_{level}
  , cluster_size_{cluster_size} {}

template <typename Index, typename Value>
bool SingleLinkageClusterNode<Index, Value>::is_leaf() const {
    // could do level_ == 0 but that might require performing float equality
    return left_ == nullptr && right_ == nullptr;
}

template <typename Index, typename Value>
std::size_t SingleLinkageClusterNode<Index, Value>::size() const {
    return cluster_size_;
}

template <typename Index, typename Value>
bool SingleLinkageClusterNode<Index, Value>::has_parent() const {
    return parent_ != nullptr;
}

template <typename Index, typename Value>
typename SingleLinkageClusterNode<Index, Value>::NodePtr SingleLinkageClusterNode<Index, Value>::get_sibling_node()
    const {
    if (has_parent()) {
        if (this == parent_->left_.get()) {
            return parent_->right_;

        } else if (this == parent_->right_.get()) {
            return parent_->left_;
        }
    }
    return nullptr;
}

template <typename Index, typename Value>
void SingleLinkageClusterNode<Index, Value>::serialize(rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    static_assert(std::is_floating_point_v<Value> || std::is_integral_v<Value>,
                  "Unsupported type during kdnode serialization");

    writer.String("representative");
    writer.Int64(representative_);

    writer.String("level");
    if constexpr (std::is_integral_v<Value>) {
        writer.Int64(level_);

    } else if constexpr (std::is_floating_point_v<Value>) {
        writer.Double(level_);
    }
    writer.String("cluster_size");
    writer.Int64(cluster_size_);
}

}  // namespace ffcl::datastruct