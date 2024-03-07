#pragma once

#include "ffcl/datastruct/FeaturesVector.hpp"

#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>  // For std::index_sequence and std::make_index_sequence
#include <vector>

namespace ffcl::datastruct {

template <typename Index, std::size_t NFeatures = 0>
class FeatureMaskArray {
  public:
    static_assert(std::is_trivial_v<Index>, "Index must be trivial.");

    using IndexType     = Index;
    using ArrayType     = FeaturesVector<IndexType, NFeatures>;
    using ContainerType = typename ArrayType::ContainerType;
    using Iterator      = typename ArrayType::Iterator;
    using ConstIterator = typename ArrayType::ConstIterator;

    explicit FeatureMaskArray(std::size_t n_features)
      : indices_{generate_dynamic_sequence(n_features)} {}

    explicit FeatureMaskArray()
      : indices_(generate_static_sequence(std::make_index_sequence<NFeatures>{})) {}

    constexpr IndexType& operator[](std::size_t index) {
        return indices_[index];
    }

    constexpr const IndexType& operator[](std::size_t index) const {
        return indices_[index];
    }

    constexpr auto size() const {
        return indices_.size();
    }

    constexpr auto begin() -> Iterator {
        return indices_.begin();
    }

    constexpr auto end() -> Iterator {
        return indices_.end();
    }

    constexpr auto begin() const -> ConstIterator {
        return indices_.begin();
    }

    constexpr auto end() const -> ConstIterator {
        return indices_.end();
    }

    constexpr auto cbegin() const -> ConstIterator {
        return indices_.cbegin();
    }

    constexpr auto cend() const -> ConstIterator {
        return indices_.cend();
    }

  private:
    ArrayType indices_;

    template <std::size_t... Int>
    static constexpr ArrayType generate_static_sequence(std::index_sequence<Int...>) {
        return {{IndexType(Int)...}};
    }

    template <std::size_t... Int>
    ArrayType generate_dynamic_sequence(std::size_t sequence_length) {
        auto indices = std::vector<IndexType>(sequence_length);
        std::iota(indices.begin(), indices.end(), static_cast<IndexType>(0));
        return ContainerType{std::move(indices)};
    }
};

}  // namespace ffcl::datastruct