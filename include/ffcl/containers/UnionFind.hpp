#pragma once

#include <cstdio>
#include <memory>
#include <numeric>

#include <iostream>

namespace ffcl {

template <typename SampleIndexType>
class UnionFind {
  public:
    UnionFind(std::size_t n_samples);

    SampleIndexType find(SampleIndexType index);

    bool merge(const SampleIndexType& index_1, const SampleIndexType& index_2);

    void print() const;

  private:
    std::size_t                        n_samples_;
    std::unique_ptr<SampleIndexType[]> labels_;
    std::unique_ptr<SampleIndexType[]> ranks_;
};

template <typename SampleIndexType>
UnionFind<SampleIndexType>::UnionFind(std::size_t n_samples)
  : n_samples_{n_samples}
  , labels_{std::make_unique<SampleIndexType[]>(n_samples)}
  , ranks_{std::make_unique<SampleIndexType[]>(n_samples)} {
    std::iota(labels_.get(), labels_.get() + n_samples, static_cast<SampleIndexType>(0));
}

template <typename SampleIndexType>
SampleIndexType UnionFind<SampleIndexType>::find(SampleIndexType index) {
    while (index != labels_[index]) {
        // set the label of each examined node to the root
        const auto temp = labels_[index];
        labels_[index]  = labels_[temp];
        index           = temp;
    }
    return index;
}

template <typename SampleIndexType>
bool UnionFind<SampleIndexType>::merge(const SampleIndexType& index_1, const SampleIndexType& index_2) {
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

template <typename SampleIndexType>
void UnionFind<SampleIndexType>::print() const {
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