#pragma once

#include <cstdio>
#include <memory>
#include <numeric>

#include <iostream>

namespace ffcl::datastruct {

template <typename Index>
class UnionFind {
  public:
    static_assert(std::is_trivial_v<Index>, "Index must be trivial.");

    explicit UnionFind(std::size_t n_samples);

    UnionFind(std::size_t n_samples, std::unique_ptr<Index[]> labels);

    Index find(Index index) const;

    Index merge(const Index& index_1, const Index& index_2);

    void print() const;

  private:
    std::size_t              n_samples_;
    std::unique_ptr<Index[]> parents_;
    std::unique_ptr<Index[]> ranks_;
};

template <typename Index>
UnionFind<Index>::UnionFind(std::size_t n_samples)
  : n_samples_{n_samples}
  , parents_{std::make_unique<Index[]>(n_samples)}
  , ranks_{std::make_unique<Index[]>(n_samples)} {
    // set each element as its own parent
    std::iota(parents_.get(), parents_.get() + n_samples, static_cast<Index>(0));
}

template <typename Index>
UnionFind<Index>::UnionFind(std::size_t n_samples, std::unique_ptr<Index[]> labels)
  : n_samples_{n_samples}
  , parents_{std::move(labels)}
  , ranks_{std::make_unique<Index[]>(n_samples)} {}

template <typename Index>
Index UnionFind<Index>::find(Index index) const {
    while (index != parents_[index]) {
        // set the label of each examined node to the representative
        const auto temp = parents_[index];
        parents_[index] = parents_[temp];
        index           = temp;
    }
    return index;
}

template <typename Index>
Index UnionFind<Index>::merge(const Index& index_1, const Index& index_2) {
    const auto representative_1 = find(index_1);
    const auto representative_2 = find(index_2);

    if (representative_1 == representative_2) {
        return representative_1;

    } else if (ranks_[representative_1] == ranks_[representative_2]) {
        parents_[representative_2] = parents_[representative_1];
        ++ranks_[representative_1];
        return parents_[representative_1];

    } else if (ranks_[representative_1] > ranks_[representative_2]) {
        parents_[representative_2] = representative_1;
        return representative_1;

    } else {
        parents_[representative_1] = representative_2;
        return representative_2;
    }
}

template <typename Index>
void UnionFind<Index>::print() const {
    std::cout << "Indices:\n";
    for (std::size_t index = 0; index < n_samples_; ++index) {
        std::cout << index << " ";
    }
    std::cout << "\n";

    std::cout << "Parents:\n";
    for (std::size_t index = 0; index < n_samples_; ++index) {
        std::cout << parents_[index] << " ";
    }
    std::cout << "\n";

    std::cout << "Ranks:\n";
    for (std::size_t index = 0; index < n_samples_; ++index) {
        std::cout << ranks_[index] << " ";
    }
    std::cout << "\n---\n";
}

}  // namespace ffcl::datastruct