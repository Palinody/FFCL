#pragma once

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterNode.hpp"
#include "ffcl/datastruct/spanning_tree/MinimumSpanningTree.hpp"

#include <cstdio>
#include <numeric>

#include <iostream>

#include "rapidjson/writer.h"

namespace ffcl {

namespace fs = std::filesystem;

template <typename IndexType, typename ValueType>
class SingleLinkageClusterTree {
  public:
    using MinimumSpanningTreeType = mst::MinimumSpanningTree<IndexType, ValueType>;
    using UnionFindType           = datastruct::UnionFind<IndexType>;

    using NodeType = SingleLinkageClusterNode<IndexType, ValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using ClusterIndexType = IndexType;

    SingleLinkageClusterTree(const MinimumSpanningTreeType& mst);

    SingleLinkageClusterTree(MinimumSpanningTreeType&& mst);

    void print() const;

    void serialize(const fs::path& filepath) const;

  private:
    void serialize(const NodePtr& node, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    auto build();

    void print_node(const NodePtr& node) const;

    MinimumSpanningTreeType sorted_mst_;

    NodePtr root_;
};

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(const MinimumSpanningTreeType& mst)
  : sorted_mst_{ffcl::mst::sort_copy(mst)}
  , root_{build()} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(MinimumSpanningTreeType&& mst)
  : sorted_mst_{ffcl::mst::sort(std::move(mst))}
  , root_{build()} {}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::print() const {
    print_node(root_);
}

template <typename IndexType, typename ValueType>
auto SingleLinkageClusterTree<IndexType, ValueType>::build() {
    const std::size_t n_samples = sorted_mst_.size() + 1;

    // union find data structure to keep track of the cluster id
    UnionFindType union_find(n_samples);

    // cluster indices mapped to their node that may contain descendants
    auto nodes = std::map<ClusterIndexType, NodePtr>{};

    // init each sample as its own cluster/component
    for (std::size_t cluster_index = 0; cluster_index < n_samples; ++cluster_index) {
        nodes[cluster_index] = std::make_shared<NodeType>(cluster_index);
    }
    for (const auto& [sample_index_1, sample_index_2, distance] : sorted_mst_) {
        // find in which cluster index the samples are currently in
        const auto representative_1 = union_find.find(sample_index_1);
        const auto representative_2 = union_find.find(sample_index_2);

        if (union_find.merge(sample_index_1, sample_index_2)) {
            const auto new_representative = union_find.find(sample_index_1);

            // create a new node, if we have reached a new level, that will agglomerate its children
            auto cluster_node = std::make_shared<NodeType>(
                new_representative, distance, nodes[representative_1]->size() + nodes[representative_2]->size());

            // link the nodes to their new parent node
            nodes[representative_1]->parent_ = nodes[new_representative];
            nodes[representative_2]->parent_ = nodes[new_representative];

            // add them as children of their common parent node
            cluster_node->left_  = std::move(nodes[representative_1]);
            cluster_node->right_ = std::move(nodes[representative_2]);

            // erase the representatives that are now children
            nodes.erase(representative_1);
            nodes.erase(representative_2);
            // place the new cluster node. It might be created at one of the prevous nodes' emplacement
            nodes[new_representative] = std::move(cluster_node);
        }
    }
    return nodes.begin()->second;
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::print_node(const NodePtr& node) const {
    std::cout << "representative "
              << "(" << node->level_ << "): " << node->representative_ << "\n---\n";

    if (!node->is_leaf()) {
        print_node(node->left_);
        print_node(node->right_);
    }
}

template <typename IndexType, typename ValueType>
void SingleLinkageClusterTree<IndexType, ValueType>::serialize(
    const NodePtr&                              node,
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

}  // namespace ffcl