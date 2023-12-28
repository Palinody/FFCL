#pragma once

#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"
#include "ffcl/datastruct/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/hdbscan/CondensedClusterTree.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ffcl {

template <typename Indexer>
class HDBSCAN {
  public:
    using IndexType = typename Indexer::IndexType;
    using ValueType = typename Indexer::DataType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    struct Options {
        Options() = default;

        Options(const Options& other) = default;

        Options& k_nearest_neighbors(std::size_t k_nearest_neighbors) {
            k_nearest_neighbors_ = k_nearest_neighbors;
            return *this;
        }

        Options& min_cluster_size(std::size_t min_cluster_size) {
            min_cluster_size_ = min_cluster_size;
            return *this;
        }

        Options& return_leaf_nodes(bool return_leaf_nodes) {
            return_leaf_nodes_ = return_leaf_nodes;
            return *this;
        }

        Options& allow_single_cluster(bool allow_single_cluster) {
            allow_single_cluster_ = allow_single_cluster;
            return *this;
        }

        Options& operator=(const Options& options) {
            k_nearest_neighbors_  = options.k_nearest_neighbors_;
            min_cluster_size_     = options.min_cluster_size_;
            return_leaf_nodes_    = options.return_leaf_nodes_;
            allow_single_cluster_ = options.allow_single_cluster_;
            return *this;
        }

        std::size_t k_nearest_neighbors_  = 1;
        std::size_t min_cluster_size_     = 5;
        bool        return_leaf_nodes_    = false;
        bool        allow_single_cluster_ = true;
    };

  public:
    HDBSCAN() = default;

    HDBSCAN(const Options& options);

    HDBSCAN(const HDBSCAN&) = delete;

    HDBSCAN<Indexer>& set_options(const Options& options);

    template <typename ForwardedIndexer>
    auto predict(ForwardedIndexer&& indexer) const;

  private:
    Options options_;
};

template <typename Indexer>
HDBSCAN<Indexer>::HDBSCAN(const Options& options)
  : options_{options} {}

template <typename Indexer>
HDBSCAN<Indexer>& HDBSCAN<Indexer>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename Indexer>
template <typename ForwardedIndexer>
auto HDBSCAN<Indexer>::predict(ForwardedIndexer&& indexer) const {
    using BoruvkasAlgorithmOptionsType    = typename BoruvkasAlgorithm<Indexer>::Options;
    using CondensedClusterTreeOptionsType = typename CondensedClusterTree<IndexType, ValueType>::Options;

    const auto boruvkas_algorithm =
        BoruvkasAlgorithm<Indexer>(BoruvkasAlgorithmOptionsType().k_nearest_neighbors(options_.k_nearest_neighbors_));

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::forward<ForwardedIndexer>(indexer));

    const auto single_linkage_cluster_tree = SingleLinkageClusterTree(std::move(minimum_spanning_tree));

    const auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    const auto condensed_cluster_tree =
        CondensedClusterTree<IndexType, ValueType>(single_linkage_cluster_tree_root,
                                                   CondensedClusterTreeOptionsType()
                                                       .min_cluster_size(options_.min_cluster_size_)
                                                       .return_leaf_nodes(options_.return_leaf_nodes_)
                                                       .allow_single_cluster(options_.allow_single_cluster_));

    return condensed_cluster_tree.predict();
}

}  // namespace ffcl
