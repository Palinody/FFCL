#pragma once

#include <cstdio>
#include <memory>
#include <numeric>

#include <iostream>

namespace ffcl::datastruct {

template <typename IndexType>
class UnionFind {
  public:
    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");

    explicit UnionFind(std::size_t n_samples);

    UnionFind(std::size_t n_samples, std::unique_ptr<IndexType[]> labels);

    IndexType find(IndexType index) const;

    IndexType merge(const IndexType& index_1, const IndexType& index_2);

    void print() const;

  private:
    std::size_t                  n_samples_;
    std::unique_ptr<IndexType[]> parents_;
    std::unique_ptr<IndexType[]> ranks_;
};

template <typename IndexType>
UnionFind<IndexType>::UnionFind(std::size_t n_samples)
  : n_samples_{n_samples}
  , parents_{std::make_unique<IndexType[]>(n_samples)}
  , ranks_{std::make_unique<IndexType[]>(n_samples)} {
    // set each element as its own parent
    std::iota(parents_.get(), parents_.get() + n_samples, static_cast<IndexType>(0));
}

template <typename IndexType>
UnionFind<IndexType>::UnionFind(std::size_t n_samples, std::unique_ptr<IndexType[]> labels)
  : n_samples_{n_samples}
  , parents_{std::move(labels)}
  , ranks_{std::make_unique<IndexType[]>(n_samples)} {}

template <typename IndexType>
IndexType UnionFind<IndexType>::find(IndexType index) const {
    while (index != parents_[index]) {
        // set the label of each examined node to the representative
        const auto temp = parents_[index];
        parents_[index] = parents_[temp];
        index           = temp;
    }
    return index;
}

template <typename IndexType>
IndexType UnionFind<IndexType>::merge(const IndexType& index_1, const IndexType& index_2) {
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

template <typename IndexType>
void UnionFind<IndexType>::print() const {
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