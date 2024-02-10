#pragma once

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"
#include "ffcl/datastruct/tree/single_linkage_cluster_tree/SingleLinkageClusterNode.hpp"

#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <cstdio>
#include <iostream>

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

    using MinimumSpanningTreeType = datastruct::mst::EdgesList<IndexType, ValueType>;
    using UnionFindType           = datastruct::UnionFind<IndexType>;

    using SingleLinkageClusterNodeType = SingleLinkageClusterNode<IndexType, ValueType>;
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
    auto build();

    void preorder_traversal_single_linkage_clustering(ClusterIndexType                   cluster_label,
                                                      const SingleLinkageClusterNodePtr& kdnode,
                                                      std::vector<ClusterIndexType>&     flat_cluster) const;

    void serialize(const SingleLinkageClusterNodePtr& node, rapidjson::Writer<rapidjson::StringBuffer>& writer) const;

    void print_node(const SingleLinkageClusterNodePtr& node) const;

    MinimumSpanningTreeType sorted_mst_;

    SingleLinkageClusterNodePtr root_;

    Options options_;
};

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(const MinimumSpanningTreeType& mst)
  : sorted_mst_{datastruct::mst::sort_copy(mst)}
  , root_{build()} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(const MinimumSpanningTreeType& mst,
                                                                         const Options&                 options)
  : sorted_mst_{datastruct::mst::sort_copy(mst)}
  , root_{build()}
  , options_{options} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(MinimumSpanningTreeType&& mst)
  : sorted_mst_{datastruct::mst::sort(std::move(mst))}
  , root_{build()} {}

template <typename IndexType, typename ValueType>
SingleLinkageClusterTree<IndexType, ValueType>::SingleLinkageClusterTree(MinimumSpanningTreeType&& mst,
                                                                         const Options&            options)
  : sorted_mst_{datastruct::mst::sort(std::move(mst))}
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
        if (single_linkage_cluster_node->size() < options_.min_cluster_size_) {
            cluster_label = 0;

        } else {
            // set the cluster label as the one at the current level
            cluster_label = single_linkage_cluster_node->representative_;
        }
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
auto SingleLinkageClusterTree<IndexType, ValueType>::build() {
    const std::size_t n_samples = sorted_mst_.size() + 1;

    // union find data structure to keep track of the cluster id
    UnionFindType union_find(n_samples);

    // cluster indices mapped to their node that may contain descendants
    auto nodes = std::unordered_map<ClusterIndexType, SingleLinkageClusterNodePtr>{};

    // init each sample as its own cluster/component
    for (std::size_t cluster_index = 0; cluster_index < n_samples; ++cluster_index) {
        nodes[cluster_index] = std::make_shared<SingleLinkageClusterNodeType>(cluster_index);
    }
    for (const auto& [sample_index_1, sample_index_2, samples_edge_weight] : sorted_mst_) {
        // find in which cluster index the samples are currently in, before being merged
        const auto representative_1 = union_find.find(sample_index_1);
        const auto representative_2 = union_find.find(sample_index_2);
        // merge the disjoints sets based on the 2 samples and return the common representative of the newly formed set
        const auto common_representative = union_find.merge(sample_index_1, sample_index_2);
        // this condition checks if a merge happened based on the old representative values
        if (common_representative != representative_1 || common_representative != representative_2) {
            // create a new node, if we have reached a new level, that will agglomerate its children
            auto cluster_node = std::make_shared<SingleLinkageClusterNodeType>(
                common_representative,
                samples_edge_weight,
                nodes[representative_1]->size() + nodes[representative_2]->size());

            // link the nodes to their new parent node
            nodes[representative_1]->parent_ = cluster_node;
            nodes[representative_2]->parent_ = cluster_node;

            // add them as children of their common parent node
            cluster_node->left_  = std::move(nodes[representative_1]);
            cluster_node->right_ = std::move(nodes[representative_2]);

            // erase the representatives that are now children
            nodes.erase(representative_1);
            nodes.erase(representative_2);
            // place the new cluster node. It might be created at one of the prevous nodes' emplacement
            nodes[common_representative] = std::move(cluster_node);
        }
    }
    return nodes.begin()->second;
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