#pragma once

#include <cstdio>
#include <memory>
#include <numeric>

#include <iostream>

namespace ffcl {

template <typename IndexType>
class UnionFind {
  public:
    UnionFind(std::size_t n_samples);

    IndexType find(IndexType index) const;

    bool merge(const IndexType& index_1, const IndexType& index_2);

    void print() const;

  private:
    std::size_t                  n_samples_;
    std::unique_ptr<IndexType[]> labels_;
    std::unique_ptr<IndexType[]> ranks_;
};

template <typename IndexType>
UnionFind<IndexType>::UnionFind(std::size_t n_samples)
  : n_samples_{n_samples}
  , labels_{std::make_unique<IndexType[]>(n_samples)}
  , ranks_{std::make_unique<IndexType[]>(n_samples)} {
    std::iota(labels_.get(), labels_.get() + n_samples, static_cast<IndexType>(0));
}

template <typename IndexType>
IndexType UnionFind<IndexType>::find(IndexType index) const {
    while (index != labels_[index]) {
        // set the label of each examined node to the root
        const auto temp = labels_[index];
        labels_[index]  = labels_[temp];
        index           = temp;
    }
    return index;
}

template <typename IndexType>
bool UnionFind<IndexType>::merge(const IndexType& index_1, const IndexType& index_2) {
    const auto representative_1 = find(index_1);
    const auto representative_2 = find(index_2);

    if (representative_1 == representative_2) {
        return false;

    } else if (ranks_[representative_1] == ranks_[representative_2]) {
        labels_[representative_2] = labels_[representative_1];
        ++ranks_[representative_1];

    } else if (ranks_[representative_1] > ranks_[representative_2]) {
        labels_[representative_2] = representative_1;

    } else {
        labels_[representative_1] = representative_2;
    }
    return true;
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
        std::cout << labels_[index] << " ";
    }
    std::cout << "\n";

    std::cout << "Ranks:\n";
    for (std::size_t index = 0; index < n_samples_; ++index) {
        std::cout << ranks_[index] << " ";
    }
    std::cout << "\n---\n";
}

}  // namespace ffcl