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

    CondensedClusterTree(SingleLinkageClusterNodePtr single_linkage_cluster_root, std::size_t min_cluster_size);

  private:
    auto build(SingleLinkageClusterNodePtr single_linkage_cluster_node);

    SingleLinkageClusterTreeType single_linkage_cluster_tree_;
    std::size_t                  min_cluster_size_;
};

}  // namespace ffcl