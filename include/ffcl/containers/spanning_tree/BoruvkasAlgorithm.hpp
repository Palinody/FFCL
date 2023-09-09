#pragma once

#include "ffcl/common/Utils.hpp"

#include <cassert>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using SampleIndexType = std::size_t;
    using SampleValueType = typename Indexer::DataType;

    using ComponentType = std::vector<SampleIndexType>;

    using EdgeType                = std::tuple<SampleIndexType, SampleIndexType, SampleValueType>;
    using MinimumSpanningTreeType = std::vector<EdgeType>;

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& k_nearest_neighbors(std::size_t k_nearest_neighbors) {
            k_nearest_neighbors_ = k_nearest_neighbors;
            return *this;
        }

        Options& operator=(const Options& options) {
            k_nearest_neighbors_ = options.k_nearest_neighbors_;
            return *this;
        }

        std::size_t k_nearest_neighbors_ = 3;
    };

    class ComponentsPartition {
      public:
        ComponentsPartition(std::size_t n_samples)
          : n_samples_{n_samples}
          , minimum_spanning_tree_{}
          , components_{std::vector<std::vector<SampleIndexType>>(n_samples)}
          , component_labels_{std::vector<SampleIndexType>(n_samples)} {
            minimum_spanning_tree_.reserve(n_samples_ - 1);

            SampleIndexType sample_index = 0;
            for (auto& component : components_) {
                component = std::vector<SampleIndexType>(1, sample_index++);
            }
            std::iota(component_labels_.begin(), component_labels_.end(), static_cast<SampleIndexType>(0));

            valid_component_indices_ = std::unordered_set(component_labels_.begin(), component_labels_.end());
        }

        auto n_elements() const {
            return n_samples_;
        }

        auto n_components() const {
            return valid_component_indices_.size();
        }

        const auto& components_indices() const {
            return valid_component_indices_;
        }

        auto component_indices_range(const SampleIndexType& component_index) const {
            return std::make_pair(components_[component_index].begin(), components_[component_index].end());
        }

        auto get_sample_index_component(const SampleIndexType& sample_index) const {
            return component_labels_[sample_index];
        }

        auto minimum_spanning_tree() const {
            return minimum_spanning_tree_;
        }

        void merge_components(const EdgeType& edge) {
            // get the indices of the samples that form an edge
            const auto sample_index_1 = std::get<0>(edge);
            const auto sample_index_2 = std::get<1>(edge);

            // get which components they belong to
            const auto component_label_1 = component_labels_[sample_index_1];
            const auto component_label_2 = component_labels_[sample_index_2];

            // return if both samples belong to the same component
            if (component_label_1 == component_label_2) {
                return;
            }
            // theres nothing to move if one of the components is empty
            if (components_[component_label_1].empty() || components_[component_label_2].empty()) {
                return;
            }
            // calculate the sizes of the two components
            std::size_t component_size_1 = components_[component_label_1].size();
            std::size_t component_size_2 = components_[component_label_2].size();

            // select the final component that will merge both components in order to move the least amount of data
            const auto [final_component, discarded_component] =
                (component_size_1 > component_size_2) ? std::make_pair(component_label_1, component_label_2)
                                                      : std::make_pair(component_label_2, component_label_1);

            components_[final_component].insert(components_[final_component].end(),
                                                std::make_move_iterator(components_[discarded_component].begin()),
                                                std::make_move_iterator(components_[discarded_component].end()));

            // update the label of each moved sample index
            assign_sample_index_to_component(discarded_component, final_component);

            // now that the old component has been merged with the final one, clear it
            components_[discarded_component].clear();

            // remove the cleared component from the valid components buffer
            valid_component_indices_.erase(discarded_component);

            // update the minimum spanning tree
            minimum_spanning_tree_.emplace_back(edge);
        }

        void print() const {
            std::cout << "component_label:\n";
            for (const auto& component_label : component_labels_) {
                std::cout << component_label << ", ";
            }
            std::cout << "\n";

            std::cout << "components:\n";
            for (std::size_t component_index = 0; component_index < components_.size(); ++component_index) {
                std::cout << component_index << ": ";

                for (const auto& sample_index : components_[component_index]) {
                    std::cout << sample_index << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";

            std::cout << "Minimum Spanning Tree (MST):\n";
            for (const auto& edge : minimum_spanning_tree_) {
                std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge) << ", " << std::get<2>(edge)
                          << "), \n";
            }
            std::cout << "\n";
        }

      private:
        void assign_sample_index_to_component(const SampleIndexType& prev_component,
                                              const SampleIndexType& new_component) {
            // iterate over the component that contains the sample indices that should be updated
            for (const auto& sample_index : components_[prev_component]) {
                // update the corresponding label with the new one
                component_labels_[sample_index] = new_component;
            }
        }

        std::size_t n_samples_;
        // the container that accumulates the edges for the minimum spanning tree
        MinimumSpanningTreeType minimum_spanning_tree_;
        // the vector containing each components, represented as vectors of indices
        std::vector<ComponentType> components_;
        // the component class/label that can range in [0, n_samples) each sample indices is mapped to
        std::vector<SampleIndexType> component_labels_;
        // the components_indices that still contain data
        std::unordered_set<SampleIndexType> valid_component_indices_;
    };

  public:
    BoruvkasAlgorithm() = default;

    BoruvkasAlgorithm(const Options& options);

    BoruvkasAlgorithm(const BoruvkasAlgorithm&) = delete;

    BoruvkasAlgorithm<Indexer>& set_options(const Options& options);

    auto make_tree(const Indexer& indexer) const;

  private:
    Options options_;
};

template <typename Indexer>
BoruvkasAlgorithm<Indexer>::BoruvkasAlgorithm(const Options& options)
  : options_{options} {}

template <typename Indexer>
BoruvkasAlgorithm<Indexer>& BoruvkasAlgorithm<Indexer>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename Indexer>
auto BoruvkasAlgorithm<Indexer>::make_tree(const Indexer& indexer) const {
    ComponentsPartition forest(indexer.n_samples());

    // forest.print();

    while (forest.n_components() > 1) {
        std::cout << "forest.n_components(): " << forest.n_components() << "\n";
        // keep track of the shortest edge from a component's sample index to a sample index thats not within the
        // same component
        std::map<SampleIndexType, EdgeType> closest_edges;

        for (const auto& component_index : forest.components_indices()) {
            // get the iterator to the first element of the current component and the last
            const auto [component_range_first, component_range_last] = forest.component_indices_range(component_index);

            auto nn_buffer_with_memory =
                NearestNeighborsBufferWithMemory<typename std::vector<SampleValueType>::iterator>(
                    component_range_first, component_range_last, 1);

            // initialize the closest edge from the current comonent to infinity
            closest_edges[component_index] = EdgeType{0, 0, common::utils::infinity<SampleValueType>()};

            for (auto component_range_it = component_range_first; component_range_it != component_range_last;
                 ++component_range_it) {
                indexer.buffered_k_nearest_neighbors_around_query_index(*component_range_it, nn_buffer_with_memory);

                if (!nn_buffer_with_memory.empty()) {
                    const auto nearest_neighbor_index    = nn_buffer_with_memory.furthest_k_nearest_neighbor_index();
                    const auto nearest_neighbor_distance = nn_buffer_with_memory.furthest_k_nearest_neighbor_distance();

                    if (nearest_neighbor_distance < std::get<2>(closest_edges[component_index])) {
                        closest_edges[component_index] =
                            EdgeType{*component_range_it, nearest_neighbor_index, nearest_neighbor_distance};
                    }
                }
            }
        }
        // merge components based on the best edges found in each component so far
        for (const auto& [component_index, edge] : closest_edges) {
            assert(std::get<2>(edge) < common::utils::infinity<SampleValueType>());
            common::utils::ignore_parameters(component_index);
            forest.merge_components(edge);
        }
        // forest.print();
    }
    // forest.print();
    return forest.minimum_spanning_tree();
}

}  // namespace ffcl