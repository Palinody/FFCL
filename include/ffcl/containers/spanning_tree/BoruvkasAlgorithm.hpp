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
    using ForestType    = std::map<SampleIndexType, ComponentType>;

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

    class Forest {
      public:
        using Iterator      = typename ForestType::iterator;
        using ConstIterator = typename ForestType::const_iterator;

        Forest(std::size_t n_samples)
          : n_samples_{n_samples}
          , minimum_spanning_tree_{}
          , components_{}
          , component_labels_{std::vector<SampleIndexType>(n_samples)} {
            minimum_spanning_tree_.reserve(n_samples_ - 1);

            for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
                components_[sample_index] = ComponentType(1, sample_index);
            }
            std::iota(component_labels_.begin(), component_labels_.end(), static_cast<SampleIndexType>(0));
        }

        auto n_elements() const {
            return n_samples_;
        }

        std::size_t n_components() const {
            return std::distance(components_.begin(), components_.end());
        }

        Iterator begin() {
            return components_.begin();
        }

        Iterator end() {
            return components_.end();
        }

        ConstIterator begin() const {
            return components_.begin();
        }

        ConstIterator end() const {
            return components_.end();
        }

        ConstIterator cbegin() const {
            return components_.cbegin();
        }

        ConstIterator cend() const {
            return components_.cend();
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
            // theres nothing to merge if one of the components is empty
            if (components_[component_label_1].empty() || components_[component_label_2].empty()) {
                return;
            }
            // calculate the sizes of the two components
            const auto component_size_1 = components_[component_label_1].size();
            const auto component_size_2 = components_[component_label_2].size();

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
            components_.erase(discarded_component);

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
            for (const auto& [component_index, component] : components_) {
                std::cout << component_index << ": ";

                for (const auto& sample_index : component) {
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
        ForestType components_;
        // the component class/label that can range in [0, n_samples) each sample indices is mapped to
        std::vector<SampleIndexType> component_labels_;
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
    Forest forest(indexer.n_samples());

    while (forest.n_components() > 1) {
        // keep track of the shortest edge from a component's sample index to a sample index thats not within the
        // same component
        auto closest_edges = std::map<SampleIndexType, EdgeType>();

        for (const auto& [component_index, component] : forest) {
            auto nn_buffer_with_memory =
                NearestNeighborsBufferWithMemory<typename std::vector<SampleValueType>::iterator>(
                    component.begin(), component.end(), 1);

            // initialize the closest edge from the current comonent to infinity
            closest_edges[component_index] = EdgeType{common::utils::infinity<SampleIndexType>(),
                                                      common::utils::infinity<SampleIndexType>(),
                                                      common::utils::infinity<SampleValueType>()};

            for (const auto& sample_index : component) {
                indexer.buffered_k_nearest_neighbors_around_query_index(sample_index, nn_buffer_with_memory);

                if (!nn_buffer_with_memory.empty()) {
                    const auto nearest_neighbor_index    = nn_buffer_with_memory.furthest_k_nearest_neighbor_index();
                    const auto nearest_neighbor_distance = nn_buffer_with_memory.furthest_k_nearest_neighbor_distance();

                    if (nearest_neighbor_distance < std::get<2>(closest_edges[component_index])) {
                        closest_edges[component_index] =
                            EdgeType{sample_index, nearest_neighbor_index, nearest_neighbor_distance};
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
    }
    return forest.minimum_spanning_tree();
}

}  // namespace ffcl