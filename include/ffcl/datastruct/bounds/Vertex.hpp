#pragma once

#include <array>
#include <cstddef>  // std::size_t
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace ffcl::datastruct::bounds {

template <typename ValueType, std::size_t NFeatures>
class Vertex;

template <typename ValueType>
class Vertex<ValueType, 0> {
  public:
    Vertex(std::initializer_list<ValueType> init_list)
      : values_(init_list) {}

    Vertex(const std::vector<ValueType>& values)
      : values_{values} {}

    Vertex(std::vector<ValueType>&& values) noexcept
      : values_{std::move(values)} {}

    ValueType& operator[](std::size_t index) {
        return values_[index];
    }

    const ValueType& operator[](std::size_t index) const {
        return values_[index];
    }

    std::size_t size() const {
        return values_.size();
    }

    constexpr auto begin() {
        return values_.begin();
    }

    constexpr auto end() {
        return values_.end();
    }

    constexpr auto begin() const {
        return values_.begin();
    }

    constexpr auto end() const {
        return values_.end();
    }

    constexpr auto cbegin() const {
        return values_.cbegin();
    }

    constexpr auto cend() const {
        return values_.cend();
    }

  private:
    std::vector<ValueType> values_;
};

template <typename ValueType, std::size_t NFeatures>
class Vertex {
  public:
    template <typename... Args, std::enable_if_t<sizeof...(Args) == NFeatures, int> = 0>
    constexpr Vertex(Args&&... args)
      : values_{{static_cast<ValueType>(std::forward<Args>(args))...}} {
        static_assert((std::is_convertible_v<Args, ValueType> && ...),
                      "All arguments must be convertible to ValueType.");
    }

    constexpr Vertex(const std::array<ValueType, NFeatures>& values)
      : values_{values} {}

    constexpr Vertex(std::array<ValueType, NFeatures>&& values) noexcept
      : values_{std::move(values)} {}

    constexpr ValueType& operator[](std::size_t index) {
        return values_[index];
    }

    constexpr const ValueType& operator[](std::size_t index) const {
        return values_[index];
    }

    constexpr std::size_t size() const {
        return values_.size();
    }

    constexpr auto begin() {
        return values_.begin();
    }

    constexpr auto end() {
        return values_.end();
    }

    constexpr auto begin() const {
        return values_.begin();
    }

    constexpr auto end() const {
        return values_.end();
    }

    constexpr auto cbegin() const {
        return values_.cbegin();
    }

    constexpr auto cend() const {
        return values_.cend();
    }

  private:
    std::array<ValueType, NFeatures> values_;
};

}  // namespace ffcl::datastruct::bounds