#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/containers/spanning_tree/MinimumSpanningTree.hpp"

#include <numeric>

#include <iostream>

namespace ffcl {

template <typename SampleIndexType, typename SampleValueType>
class SingleLinkageClusterNode {
  public:
    using NodeType = SingleLinkageClusterNode<SampleIndexType, SampleValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    SingleLinkageClusterNode(const SampleIndexType& representative, const SampleValueType& level = 0)
      : representative_{representative}
      , level_{level} {
        //
    }

    bool is_leaf() const {
        // could do level_ == 0 but that might require performing float equality
        return (left_ == nullptr && right_ == nullptr);
    }

  public:
    SampleIndexType representative_;

    SampleValueType level_;

    NodePtr parent_, left_, right_;
};

template <typename SampleIndexType>
class UnionFind {
  public:
    UnionFind(std::size_t n_samples)
      : n_samples_{n_samples}
      , labels_{std::make_unique<SampleIndexType[]>(n_samples)}
      , ranks_{std::make_unique<SampleIndexType[]>(n_samples)} {
        std::iota(labels_.get(), labels_.get() + n_samples, static_cast<SampleIndexType>(0));
    }

    SampleIndexType find(SampleIndexType index) {
        while (index != labels_[index]) {
            // set the label of each examined node to the root
            const auto temp = labels_[index];
            labels_[index]  = labels_[temp];
            index           = temp;
        }
        return index;
    }

    bool merge(const SampleIndexType& index_1, const SampleIndexType& index_2) {
        const auto representative_1 = find(index_1);
        const auto representative_2 = find(index_2);

        if (representative_1 == representative_2) {
            return false;
        }
        if (ranks_[representative_1] < ranks_[representative_2]) {
            labels_[representative_1] = representative_2;
            ++ranks_[representative_2];

        } else {
            labels_[representative_2] = representative_1;
            ++ranks_[representative_1];
        }
        return true;
    }

    void print() const {
        std::cout << "Indices:\n";
        for (std::size_t index = 0; index < n_samples_; ++index) {
            std::cout << index << " ";
        }
        std::cout << "\n";

        std::cout << "Parents:\n";
        for (std::size_t index = 0; index < n_samples_; ++index) {
            std::cout << labels_[index] << " ";
        }
        std::cout << "\n";

        std::cout << "Ranks:\n";
        for (std::size_t index = 0; index < n_samples_; ++index) {
            std::cout << ranks_[index] << " ";
        }
        std::cout << "\n---\n";
    }

  private:
    std::size_t                        n_samples_;
    std::unique_ptr<SampleIndexType[]> labels_;
    std::unique_ptr<SampleIndexType[]> ranks_;
};

template <typename SampleIndexType, typename SampleValueType>
class SingleLinkageClusterTree {
  public:
    using MinimumSpanningTreeType = mst::MinimumSpanningTreeType<SampleIndexType, SampleValueType>;
    using UnionFindType           = UnionFind<SampleIndexType>;

    using NodeType = SingleLinkageClusterNode<SampleIndexType, SampleValueType>;
    using NodePtr  = std::shared_ptr<NodeType>;

    using ClusterIndexType = SampleIndexType;

    SingleLinkageClusterTree(const MinimumSpanningTreeType& mst)
      : sorted_mst_{ffcl::mst::sort_copy(mst)}
      , root_{build()} {}

    SingleLinkageClusterTree(MinimumSpanningTreeType&& mst)
      : sorted_mst_{ffcl::mst::sort(std::move(mst))}
      , root_{build()} {}

    void print() const {
        //
    }

  private:
    auto build() {
        // ffcl::mst::print(sorted_mst_);

        const std::size_t n_samples = sorted_mst_.size() + 1;

        // union find data structure to keep track of the cluster id
        UnionFindType union_find(n_samples);

        // cluster indices mapped to their node
        auto nodes = std::map<ClusterIndexType, NodePtr>{};

        // init each sample as its own cluster/component
        for (std::size_t component_index = 0; component_index < n_samples; ++component_index) {
            nodes[component_index] = std::make_shared<NodeType>(component_index);
        }

        for (const auto& [sample_index_1, sample_index_2, distance] : sorted_mst_) {
            // find in which cluster index the following samples are currently in
            const auto representative_1 = union_find.find(sample_index_1);
            const auto representative_2 = union_find.find(sample_index_2);

            if (union_find.merge(sample_index_1, sample_index_2)) {
                const auto new_representative = union_find.find(sample_index_1);

                // create a new node, if we have reached a new level, that will agglomerate its children
                auto cluster_node = std::make_shared<NodeType>(new_representative, distance);

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
                nodes[new_representative] = cluster_node;
            }
            /*
            for (const auto& [key, node] : nodes) {
                std::cout << "key: " << key << ", ";
            }
            std::cout << "\n";
            */
        }
        /*
        std::cout << "nodes.size(): " << nodes.size() << "\n";

        std::cout << "---\n";

        for (const auto& [key, node] : nodes) {
            std::cout << "key: " << key << "\n";
        }
        */

        return (*nodes.begin()).second;
    }

    MinimumSpanningTreeType sorted_mst_;

    NodePtr root_;
};  // namespace ffcl

}  // namespace ffcl