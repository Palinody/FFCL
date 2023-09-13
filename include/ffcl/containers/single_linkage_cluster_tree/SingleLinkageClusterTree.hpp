#pragma once

#include "ffcl/containers/spanning_tree/MinimumSpanningTree.hpp"

#include <numeric>

namespace ffcl {

template <typename SampleIndexType, typename SampleValueType>
class SingleLinkageClusterTree {
  public:
    using MinimumSpanningTreeType = mst::MinimumSpanningTreeType<SampleIndexType, SampleValueType>;

    class UnionFind {
      public:
        UnionFind(std::size_t n_samples)
          : parents_{std::make_unique<SampleIndexType[]>(n_samples)}
          , ranks_{std::make_unique<SampleIndexType[]>(n_samples)} {
            std::iota(parents_.get(), parents_.get() + n_samples, static_cast<SampleIndexType>(0));
        }

        SampleIndexType find(SampleIndexType index) const {
            while (index != parents_[index]) {
                index = parents_[index];
            }
            return index;
        }

        bool merge(const SampleIndexType& index_1, const SampleIndexType& index_2) {
            const auto parent_1 = find(index_1);
            const auto parent_2 = find(index_2);

            if (parent_1 == parent_2) {
                return false;
            }
            if (ranks_[parent_1] < ranks_[parent_2]) {
                parents_[parent_1] = parent_2;

            } else if (ranks_[parent_1] > ranks_[parent_2]) {
                parents_[parent_2] = parent_1;

            } else {
                parents_[parent_2] = parent_1;
                ++ranks_[parent_1];
            }
            return true;
        }

      private:
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
    }

  private:
    MinimumSpanningTreeType sorted_mst_;
};

}  // namespace ffcl