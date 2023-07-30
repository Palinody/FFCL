#pragma once

#include "ffcl/common/Utils.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace ffcl {

template <typename Indexer>
class BoruvkasAlgorithm {
  public:
    using DataType = typename Indexer::DataType;

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

  public:
    BoruvkasAlgorithm() = default;

    BoruvkasAlgorithm(const Options& options);

    BoruvkasAlgorithm(const BoruvkasAlgorithm&) = delete;

    BoruvkasAlgorithm<Indexer>& set_options(const Options& options);

    template <typename IndexerFunction, typename... Args>
    auto make_tree(const Indexer& indexer, IndexerFunction&& indexer_function, Args&&... args) const;

  private:
    Options options_;

    std::vector<std::vector<DataType>> graph_;
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
template <typename IndexerFunction, typename... Args>
auto BoruvkasAlgorithm<Indexer>::make_tree(const Indexer&    indexer,
                                           IndexerFunction&& indexer_function,
                                           Args&&... args) const {
    // the query function that should be a member of the indexer
    auto distance_function = [&indexer, indexer_function = std::forward<IndexerFunction>(indexer_function)](
                                 std::size_t sample_index, auto&&... funcArgs) mutable {
        return std::invoke(indexer_function, indexer, sample_index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

    common::utils::ignore_parameters(distance_function, args...);

    return graph_;
}

}  // namespace ffcl