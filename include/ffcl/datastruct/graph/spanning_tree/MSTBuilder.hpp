#pragma once

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

namespace ffcl {

// Accumulates the edges of a Boruvka step and keeps track of the components that are still disjoint.
template <typename Representative, typename Distance>
class MSTBuilder {
  public:
    using RepresentativeType = Representative;
    using DistanceType       = Distance;

    using UnionFindType = datastruct::UnionFind<RepresentativeType>;

    using EdgeType = datastruct::mst::Edge<RepresentativeType, DistanceType>;

    using MinimumSpanningTreeType = datastruct::mst::EdgesList<RepresentativeType, DistanceType>;

    MSTBuilder(std::size_t n_samples)
      : minimum_spanning_tree_{}
      , union_find_{UnionFindType(n_samples)}
      , n_components_{n_samples} {
        minimum_spanning_tree_.reserve(n_samples - 1);
    }

    constexpr std::size_t n_components() const {
        return n_components_;
    }

    const auto& get_union_find_const_ref() const {
        return union_find_;
    }

    auto&& minimum_spanning_tree() && {
        return std::move(minimum_spanning_tree_);
    }

    constexpr RepresentativeType find(std::size_t index) const {
        return union_find_.find(index);
    }

    void merge_components(const EdgeType& edge) {
        // get the indices of the samples that form an edge
        const auto sample_index_1 = std::get<0>(edge);
        const auto sample_index_2 = std::get<1>(edge);

        // Won't merge if both samples belong to the same component.
        const bool merge_occured = union_find_.try_merge(sample_index_1, sample_index_2);

        if (merge_occured) {
            // One component has been merged into the other so the overall number of components can be updated
            // accordingly.
            --n_components_;

            // update the minimum spanning tree
            minimum_spanning_tree_.emplace_back(edge);
        }
    }

    void print() const {
        std::cout << "Minimum Spanning Tree (MST):\n";
        for (const auto& edge : minimum_spanning_tree_) {
            std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "), \n";
        }
        std::cout << "\n";
    }

  private:
    // the container that accumulates the edges for the minimum spanning tree
    MinimumSpanningTreeType minimum_spanning_tree_;
    // a union find data structure used to merge clusters based on sample indices from distinct clusters
    UnionFindType union_find_;

    std::size_t n_components_;
};

}  // namespace ffcl
