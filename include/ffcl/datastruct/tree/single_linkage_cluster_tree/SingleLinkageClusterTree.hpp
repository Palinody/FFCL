#pragma once

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"
#include "ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterNode.hpp"

#include <filesystem>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <cstdio>
#include <fstream>
#include <iostream>

#include "rapidjson/document.h"
#include "rapidjson/ostreamwrapper.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace ffcl::datastruct {

namespace fs = std::filesystem;

template <typename Index, typename Value>
class SingleLinkageClusterTree {
  public:
    using IndexType = Index;
    using ValueType = Value;

    static_assert(std::is_fundamental<IndexType>::value, "IndexType must be a fundamental type.");
    static_assert(std::is_fundamental<ValueType>::value, "ValueType must be a fundamental type.");

    using MinimumSpanningTreeType = mst::EdgesList<IndexType, ValueType>;
    using UnionFindType           = UnionFind<IndexType>;

    using SingleLinkageClusterNodeType = SingleLinkageClusterNode<IndexType, ValueType>;
    using NodeStorageType              = std::vector<SingleLinkageClusterNodeType>;
    using SingleLinkageClusterNodePtr  = typename SingleLinkageClusterNodeType::NodePtr;

    using ClusterIndexType = IndexType;

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& cut_level(const ValueType& cut_level) {
            cut_level_ = cut_level;
            return *this;
        }

        Options& min_cluster_size(std::size_t min_cluster_size) {
            min_cluster_size_ = min_cluster_size;
            return *this;
        }

        Options& operator=(const Options& options) {
            cut_level_        = options.cut_level_;
            min_cluster_size_ = options.min_cluster_size_;
            return *this;
        }

        ValueType   cut_level_        = common::infinity<ValueType>();
        std::size_t min_cluster_size_ = 1;
    };

    SingleLinkageClusterTree(const MinimumSpanningTreeType& mst);

    SingleLinkageClusterTree(const MinimumSpanningTreeType& mst, const Options& options);

    SingleLinkageClusterTree(MinimumSpanningTreeType&& mst);

    SingleLinkageClusterTree(MinimumSpanningTreeType&& mst, const Options& options);

    SingleLinkageClusterTree<IndexType, ValueType>& set_options(const Options& options);

    constexpr auto root() const;

    auto extract_flat_clusters() const;

    auto predict() const;

    void print() const;

    void serialize(const fs::path& filepath) const;

  private:
    auto build() -> SingleLinkageClusterNodePtr;

    void preorder_traversal_single_linkage_clustering(ClusterIndexType                   cluster_label,
                                                      const SingleLinkageClusterNodePtr& kdnode,
                                                      std::vector<ClusterIndexType>&     flat_cluster) const;

    void serialize(const SingleLinkageClusterNodePtr& node, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void print_node(const SingleLinkageClusterNodePtr& node) const;

    MinimumSpanningTreeType sorted_mst_;

    // Every node of the tree, in one array. The root handle aliases it, so a condensed tree built from the
    // root keeps the nodes alive even if this tree goes away first. Declared before the root: `build()` runs
    // in the root's initialiser and assigns this member, so it has to be constructed first.
    std::shared_ptr<NodeStorageType> nodes_;

    SingleLinkageClusterNodePtr root_;

    Options options_;
};

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(const MinimumSpanningTreeType& mst)
  : sorted_mst_{mst::sort_copy(mst)}
  , root_{build()} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(const MinimumSpanningTreeType& mst,
                                                                         const Options&                 options)
  : sorted_mst_{mst::sort_copy(mst)}
  , root_{build()}
  , options_{options} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(MinimumSpanningTreeType&& mst)
  : sorted_mst_{mst::sort(std::move(mst))}
  , root_{build()} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(MinimumSpanningTreeType&& mst,
                                                                         const Options&            options)
  : sorted_mst_{mst::sort(std::move(mst))}
  , root_{build()}
  , options_{options} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>& SingleLinkageClusterTree<IndexType, ValueType>::set_options(
    const Options& options) {
    options_ = options;
    return *this;
}

template <typename IndexType, typename ValueType>
constexpr auto SingleLinkageClusterTree<IndexType, ValueType>::root() const {
    return root_;
}

template <typename IndexType, typename ValueType>
auto SingleLinkageClusterTree<IndexType, ValueType>::extract_flat_clusters() const {
    auto flat_cluster = std::vector<ClusterIndexType>(root_->size());

    preorder_traversal_single_linkage_clustering(root_->representative_, root_, flat_cluster);

    return flat_cluster;
}

template <typename IndexType, typename ValueType>
auto SingleLinkageClusterTree<IndexType, ValueType>::predict() const {
    return extract_flat_clusters();
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::preorder_traversal_single_linkage_clustering(
    ClusterIndexType                   cluster_label,
    const SingleLinkageClusterNodePtr& single_linkage_cluster_node,
    std::vector<ClusterIndexType>&     flat_cluster) const {
    // consider updating the cluster label for the descendant nodes only if we are above or at the desired cut level
    // else the same cluster label will be used for all the descendant nodes from the current node
    if (single_linkage_cluster_node->level_ >= options_.cut_level_) {
        // set the cluster label as noise if the current node is parent of less than 'min cluster size' nodes
        // else set the cluster label as the one at the current level
        cluster_label = single_linkage_cluster_node->size() < options_.min_cluster_size_
                            ? 0
                            : single_linkage_cluster_node->representative_;
    }
    // continue to traverse the tree if the current node is not leaf
    // a single linkage cluster node is guaranteed to have a left and a right child if its not leaf
    if (!single_linkage_cluster_node->is_leaf()) {
        preorder_traversal_single_linkage_clustering(cluster_label, single_linkage_cluster_node->left_, flat_cluster);
        preorder_traversal_single_linkage_clustering(cluster_label, single_linkage_cluster_node->right_, flat_cluster);

    } else {
        // assign the cluster label to the sample index (which is its own node at level 0)
        flat_cluster[single_linkage_cluster_node->representative_] = cluster_label;
    }
}

template <typename IndexType, typename ValueType>
auto SingleLinkageClusterTree<IndexType, ValueType>::build() -> SingleLinkageClusterNodePtr {
    const std::size_t n_samples = sorted_mst_.size() + 1;

    // Every merge creates one internal node over the n leaves, so the tree has exactly 2n - 1 nodes. They are
    // all allocated once, in one array reserved to that size, and linked with plain pointers: no per-node
    // allocation and no reference counting anywhere in the tree. The children handles alias the array, so
    // they share ownership of it instead of owning their targets.
    auto nodes = std::make_shared<NodeStorageType>();
    nodes->reserve(2 * n_samples - 1);

    UnionFindType union_find(n_samples);

    // the current node of every live component, indexed by its union-find representative
    auto component_to_node = std::vector<SingleLinkageClusterNodeType*>(n_samples, nullptr);

    for (std::size_t cluster_index = 0; cluster_index < n_samples; ++cluster_index) {
        component_to_node[cluster_index] = &nodes->emplace_back(static_cast<ClusterIndexType>(cluster_index));
    }
    for (const auto& [sample_index_1, sample_index_2, samples_edge_weight] : sorted_mst_) {
        const auto representative_1 = union_find.find(sample_index_1);
        const auto representative_2 = union_find.find(sample_index_2);

        const auto common_representative = union_find.merge(sample_index_1, sample_index_2);

        if (common_representative != representative_1 || common_representative != representative_2) {
            auto* const left_node  = component_to_node[representative_1];
            auto* const right_node = component_to_node[representative_2];

            auto& cluster_node =
                nodes->emplace_back(common_representative, samples_edge_weight, left_node->size() + right_node->size());

            left_node->parent_  = &cluster_node;
            right_node->parent_ = &cluster_node;

            // Non-owning handles: the arena owns every node, so a child must not also own the arena, or the
            // arena would keep itself alive through its own nodes.
            cluster_node.left_  = SingleLinkageClusterNodePtr(left_node, [](auto*) {});
            cluster_node.right_ = SingleLinkageClusterNodePtr(right_node, [](auto*) {});

            // the common representative is one of the two, so clear them before setting it
            component_to_node[representative_1]      = nullptr;
            component_to_node[representative_2]      = nullptr;
            component_to_node[common_representative] = &cluster_node;
        }
    }
    auto* const root_node = &nodes->back();

    nodes_ = std::move(nodes);

    // the root handle aliases the array, so it keeps every node alive
    return SingleLinkageClusterNodePtr(nodes_, root_node);
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::print() const {
    print_node(root_);
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::print_node(const SingleLinkageClusterNodePtr& node) const {
    std::cout << "representative "
              << "(" << node->level_ << "): " << node->representative_ << "\n---\n";

    if (!node->is_leaf()) {
        print_node(node->left_);
        print_node(node->right_);
    }
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::serialize(
    const SingleLinkageClusterNodePtr&          node,
    rapidjson::Writer<rapidjson::StringBuffer>& writer) const {
    writer.StartObject();
    {
        node->serialize(writer);

        // continue the recursion if the current node is not leaf
        if (!node->is_leaf()) {
            writer.String("left");
            serialize(node->left_, writer);

            writer.String("right");
            serialize(node->right_, writer);
        }
    }
    writer.EndObject();
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::serialize(const fs::path& filepath) const {
    rapidjson::Document document;

    rapidjson::StringBuffer buffer;

    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

    writer.StartObject();
    {
        writer.String("root");
        serialize(root_, writer);
    }
    writer.EndObject();

    document.Parse(buffer.GetString());

    std::ofstream output_file(filepath);

    rapidjson::OStreamWrapper output_stream_wrapper(output_file);

    rapidjson::Writer<rapidjson::OStreamWrapper> filewriter(output_stream_wrapper);

    document.Accept(filewriter);

    output_file.close();
}

}  // namespace ffcl::datastruct