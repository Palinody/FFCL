#pragma once

#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>  // For std::index_sequence and std::make_index_sequence
#include <vector>

namespace ffcl::datastruct {

template <typename Index, std::size_t NFeatures = 0>
class FeaturesMask {
  public:
    static_assert(std::is_trivial_v<Index>, "Index must be trivial.");

    using IndexType          = Index;
    using FeaturesVectorType = FeaturesVector<IndexType, NFeatures>;
    using ContainerType      = typename FeaturesVectorType::ContainerType;
    using Iterator           = typename FeaturesVectorType::Iterator;
    using ConstIterator      = typename FeaturesVectorType::ConstIterator;

    explicit FeaturesMask(std::size_t n_features)
      : indices_{generate_dynamic_sequence(n_features)} {}

    explicit FeaturesMask(std::initializer_list<IndexType> init_list)
      : indices_{init_list} {}

    constexpr explicit FeaturesMask(const ContainerType& values)
      : indices_{values} {}

    constexpr explicit FeaturesMask(ContainerType&& values) noexcept
      : indices_{std::move(values)} {}

    template <std::size_t NFeaturesConstraint                             = NFeatures,
              typename std::enable_if_t<(NFeaturesConstraint != 0), bool> = true>
    constexpr explicit FeaturesMask()
      : indices_(generate_static_sequence(std::make_index_sequence<NFeatures>{})) {}

    template <typename... Int, std::enable_if_t<sizeof...(Int) == NFeatures, bool> = true>
    constexpr explicit FeaturesMask(Int&&... args)
      : indices_{{static_cast<IndexType>(std::forward<Int>(args))...}} {
        static_assert((std::is_convertible_v<Int, IndexType> && ...),
                      "All arguments must be convertible to IndexType.");
    }

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

    void print() const {
        for (const auto& index : indices_) {
            std::cout << index << " ";
        }
        std::cout << "\n";
    }

  private:
    template <std::size_t... Int>
    static constexpr FeaturesVectorType generate_static_sequence(std::index_sequence<Int...>) {
        return {{IndexType{Int}...}};
    }

    FeaturesVectorType generate_dynamic_sequence(std::size_t sequence_length) {
        auto indices = std::vector<IndexType>(sequence_length);
        std::iota(indices.begin(), indices.end(), static_cast<IndexType>(0));
        return ContainerType{std::move(indices)};
    }

    FeaturesVectorType indices_;
};

}  // namespace ffcl::datastruct