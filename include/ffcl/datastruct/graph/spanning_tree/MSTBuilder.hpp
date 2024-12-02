#pragma once

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"

#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace ffcl {

template <typename Representative, typename Distance>
class ClusteredMTSBuilder {
  public:
    using RepresentativeType = Representative;
    using DistanceType       = Distance;

    using ComponentType = std::vector<RepresentativeType>;

    using RepresentativeToComponentMapType = std::unordered_map<RepresentativeType, ComponentType>;

    using UnionFindType = datastruct::UnionFind<RepresentativeType>;

    using EdgeType = datastruct::mst::Edge<RepresentativeType, DistanceType>;

    using MinimumSpanningTreeType = datastruct::mst::EdgesList<RepresentativeType, DistanceType>;

    ClusteredMTSBuilder(std::size_t n_samples)
      : minimum_spanning_tree_{}
      , representatives_to_components_map_{}
      , union_find_{UnionFindType(n_samples)} {
        minimum_spanning_tree_.reserve(n_samples - 1);
        // each sample starts as its own component representative
        for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
            representatives_to_components_map_[sample_index] = ComponentType{sample_index};
        }
    }

    constexpr std::size_t n_components() const {
        return std::distance(representatives_to_components_map_.begin(), representatives_to_components_map_.end());
    }

    const auto& get_union_find_const_ref() const {
        return union_find_;
    }

    constexpr auto begin() {
        return representatives_to_components_map_.begin();
    }

    constexpr auto end() {
        return representatives_to_components_map_.end();
    }

    constexpr auto begin() const {
        return representatives_to_components_map_.begin();
    }

    constexpr auto end() const {
        return representatives_to_components_map_.end();
    }

    constexpr auto cbegin() const {
        return representatives_to_components_map_.cbegin();
    }

    constexpr auto cend() const {
        return representatives_to_components_map_.cend();
    }

    auto&& minimum_spanning_tree() && {
        return std::move(minimum_spanning_tree_);
    }

    void merge_components(const EdgeType& edge) {
        // get the indices of the samples that form an edge
        const auto sample_index_1 = std::get<0>(edge);
        const auto sample_index_2 = std::get<1>(edge);

        // get which component belongs to which representative before the merge
        const auto representative_1 = union_find_.find(sample_index_1);
        const auto representative_2 = union_find_.find(sample_index_2);

        // return if both samples belong to the same component
        if (representative_1 == representative_2) {
            return;
        }
        // merge the sets based on the 2 samples and return the common representative of the newly formed set
        const auto common_representative = union_find_.merge(sample_index_1, sample_index_2);

        // Determine which component will be retained and which will be discarded.
        // The Union-Find structure selects the final representative based on rank comparison,
        // ensuring that the component with the smaller size is the one that gets moved.
        // This decision is made by comparing the representatives before the merge to the common representative
        // determined by UnionFind after the merge.
        const auto [retained_representative, discarded_representative] =
            (representative_1 == common_representative) ? std::make_pair(representative_1, representative_2)
                                                        : std::make_pair(representative_2, representative_1);

        // move the indices from the component that will be discarded to the final one
        representatives_to_components_map_[retained_representative].insert(
            representatives_to_components_map_[retained_representative].end(),
            std::make_move_iterator(representatives_to_components_map_[discarded_representative].cbegin()),
            std::make_move_iterator(representatives_to_components_map_[discarded_representative].cend()));

        // now that the old component has been merged with the final one, clear it
        representatives_to_components_map_.erase(discarded_representative);

        // update the minimum spanning tree
        minimum_spanning_tree_.emplace_back(edge);
    }

    void print() const {
        std::cout << "components:\n";
        for (const auto& [component_index, component] : representatives_to_components_map_) {
            std::cout << component_index << ": ";

            for (const auto& sample_index : component) {
                std::cout << sample_index << ", ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "Minimum Spanning Tree (MST):\n";
        for (const auto& edge : minimum_spanning_tree_) {
            std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge) << "), \n";
        }
        std::cout << "\n";
    }

    // the container that accumulates the edges for the minimum spanning tree
    MinimumSpanningTreeType minimum_spanning_tree_;
    // the container mapping each component representative to the set of actual sample indices
    RepresentativeToComponentMapType representatives_to_components_map_;
    // a union find data structure used to merge clusters based on sample indices from distinct clusters
    UnionFindType union_find_;
};

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

    // the container that accumulates the edges for the minimum spanning tree
    MinimumSpanningTreeType minimum_spanning_tree_;
    // a union find data structure used to merge clusters based on sample indices from distinct clusters
    UnionFindType union_find_;

    std::size_t n_components_;
};

}  // namespace ffcl