#pragma once

#include <array>
#include <cstddef>  // std::size_t
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace ffcl::datastruct {

template <typename ValueType, std::size_t NFeatures>
class FeaturesVector;

template <typename Value>
class FeaturesVector<Value, 0> {
  public:
    using ValueType     = Value;
    using ContainerType = std::vector<ValueType>;
    using Iterator      = typename ContainerType::iterator;
    using ConstIterator = typename ContainerType::const_iterator;

    FeaturesVector(std::initializer_list<ValueType> init_list)
      : values_(init_list) {}

    FeaturesVector(const ContainerType& values)
      : values_{values} {}

    FeaturesVector(ContainerType&& values) noexcept
      : values_{std::move(values)} {}

    ValueType& operator[](std::size_t index) {
        return values_[index];
    }

    const ValueType& operator[](std::size_t index) const {
        return values_[index];
    }

    constexpr auto size() const {
        return values_.size();
    }

    constexpr auto begin() -> Iterator {
        return values_.begin();
    }

    constexpr auto end() -> Iterator {
        return values_.end();
    }

    constexpr auto begin() const -> ConstIterator {
        return values_.begin();
    }

    constexpr auto end() const -> ConstIterator {
        return values_.end();
    }

    constexpr auto cbegin() const -> ConstIterator {
        return values_.cbegin();
    }

    constexpr auto cend() const -> ConstIterator {
        return values_.cend();
    }

  private:
    ContainerType values_;
};

template <typename Value, std::size_t NFeatures>
class FeaturesVector {
  public:
    using ValueType     = Value;
    using ContainerType = std::array<ValueType, NFeatures>;
    using Iterator      = typename ContainerType::iterator;
    using ConstIterator = typename ContainerType::const_iterator;

    template <typename... Args, std::enable_if_t<sizeof...(Args) == NFeatures, int> = 0>
    constexpr FeaturesVector(Args&&... args)
      : values_{{static_cast<ValueType>(std::forward<Args>(args))...}} {
        static_assert((std::is_convertible_v<Args, ValueType> && ...),
                      "All arguments must be convertible to ValueType.");
    }

    constexpr FeaturesVector(const ContainerType& values)
      : values_{values} {}

    constexpr FeaturesVector(ContainerType&& values) noexcept
      : values_{std::move(values)} {}

    constexpr ValueType& operator[](std::size_t index) {
        return values_[index];
    }

    constexpr const ValueType& operator[](std::size_t index) const {
        return values_[index];
    }

    constexpr auto size() const {
        return values_.size();
    }

    constexpr auto begin() -> Iterator {
        return values_.begin();
    }

    constexpr auto end() -> Iterator {
        return values_.end();
    }

    constexpr auto begin() const -> ConstIterator {
        return values_.begin();
    }

    constexpr auto end() const -> ConstIterator {
        return values_.end();
    }

    constexpr auto cbegin() const -> ConstIterator {
        return values_.cbegin();
    }

    constexpr auto cend() const -> ConstIterator {
        return values_.cend();
    }

  private:
    ContainerType values_;
};

}  // namespace ffcl::datastruct