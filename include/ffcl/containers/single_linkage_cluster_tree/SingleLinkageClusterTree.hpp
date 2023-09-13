#pragma once

#include "ffcl/containers/spanning_tree/MinimumSpanningTree.hpp"

#include <numeric>

#include <iostream>

namespace ffcl {

template <typename SampleIndexType, typename SampleValueType>
class SingleLinkageClusterTree {
  public:
    using MinimumSpanningTreeType = mst::MinimumSpanningTreeType<SampleIndexType, SampleValueType>;

    class UnionFind {
      public:
        UnionFind(std::size_t n_samples)
          : n_samples_{n_samples}
          , parents_{std::make_unique<SampleIndexType[]>(n_samples)}
          , ranks_{std::make_unique<SampleIndexType[]>(n_samples)} {
            std::iota(parents_.get(), parents_.get() + n_samples, static_cast<SampleIndexType>(0));
        }

        SampleIndexType find(const SampleIndexType& index) {
            if (parents_[index] != index) {
                // Recursively find the representative
                parents_[index] = find(parents_[index]);
            }
            return index;
        }

        bool merge(const SampleIndexType& index_1, const SampleIndexType& index_2) {
            const auto representative_1 = find(index_1);
            const auto representative_2 = find(index_2);

            if (representative_1 == representative_2) {
                return false;
            }
            if (ranks_[representative_1] < ranks_[representative_2]) {
                parents_[representative_1] = representative_2;

            } else if (ranks_[representative_1] > ranks_[representative_2]) {
                parents_[representative_2] = representative_1;

            } else {
                parents_[representative_2] = representative_1;
                ++ranks_[representative_1];
            }
            return true;
        }

        void print() const {
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

      private:
        std::size_t                        n_samples_;
        std::unique_ptr<SampleIndexType[]> parents_;
        std::unique_ptr<SampleIndexType[]> ranks_;
    };

    SingleLinkageClusterTree(const MinimumSpanningTreeType& mst)
      : sorted_mst_{ffcl::mst::sort_copy(mst)} {
        ffcl::mst::print(sorted_mst_);
    }

    SingleLinkageClusterTree(MinimumSpanningTreeType&& mst)
      : sorted_mst_{ffcl::mst::sort(std::move(mst))} {
        ffcl::mst::print(sorted_mst_);

        UnionFind union_find(mst.size() + 1);

        union_find.print();
        for (const auto& edge : sorted_mst_) {
            union_find.merge(std::get<0>(edge), std::get<1>(edge));
            union_find.print();
        }
    }

  private:
    MinimumSpanningTreeType sorted_mst_;
};

}  // namespace ffcl