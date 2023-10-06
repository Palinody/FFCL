#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"

#include <memory>

namespace ffcl {

namespace fs = std::filesystem;

template <typename IndexType, typename ValueType>
class CondensedClusterTree {
    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

  public:
    using NodeType = CondensedClusterTree<IndexType, ValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using SingleLinkageClusterTreeType = SingleLinkageClusterTree<IndexType, ValueType>;
    using SingleLinkageClusterNodeType = typename SingleLinkageClusterTreeType::NodeType;
    using SingleLinkageClusterNodePtr  = typename SingleLinkageClusterTreeType::NodePtr;

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

    auto build(const SingleLinkageClusterNodePtr& single_linkage_cluster_node);

    void preorder_traversal(const SingleLinkageClusterNodePtr& single_linkage_cluster_node);

  private:
    SingleLinkageClusterTreeType single_linkage_cluster_root_;

    Options options_;
};

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root)
  : single_linkage_cluster_root_{single_linkage_cluster_root} {}

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>::CondensedClusterTree(
    SingleLinkageClusterNodePtr single_linkage_cluster_root,
    const Options&              options)
  : single_linkage_cluster_root_{single_linkage_cluster_root}
  , options_{options} {}

template <typename IndexType, typename ValueType>
CondensedClusterTree<IndexType, ValueType>& CondensedClusterTree<IndexType, ValueType>::set_options(
    const Options& options) {
    options_ = options;
    return *this;
}

template <typename IndexType, typename ValueType>
auto CondensedClusterTree<IndexType, ValueType>::build(const SingleLinkageClusterNodePtr& single_linkage_cluster_node) {
    auto condensed_cluster_node = std::make_shared<NodeType>(single_linkage_cluster_node);

    if (!single_linkage_cluster_node->is_leaf()) {
        //
    }
    return condensed_cluster_node;
}

template <typename IndexType, typename ValueType>
void CondensedClusterTree<IndexType, ValueType>::preorder_traversal(
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node) {
    if (!single_linkage_cluster_node->is_leaf()) {
        //
    }
}

}  // namespace ffcl