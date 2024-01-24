#pragma once

#include <array>
#include <cstddef>  // std::size_t
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace ffcl::datastruct::bounds {

template <typename ValueType, std::size_t NFeatures>
class Vertex;

template <typename Value>
class Vertex<Value, 0> {
  public:
    using ValueType     = Value;
    using ContainerType = std::vector<ValueType>;
    using IteratorType  = typename ContainerType::iterator;

    Vertex(std::initializer_list<Value> init_list)
      : values_(init_list) {}

    Vertex(const ContainerType& values)
      : values_{values} {}

    Vertex(ContainerType&& values) noexcept
      : values_{std::move(values)} {}

    Value& operator[](std::size_t index) {
        return values_[index];
    }

    const Value& operator[](std::size_t index) const {
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
    ContainerType values_;
};

template <typename Value, std::size_t NFeatures>
class Vertex {
  public:
    using ValueType     = Value;
    using ContainerType = std::array<ValueType, NFeatures>;
    using IteratorType  = typename ContainerType::iterator;

    template <typename... Args, std::enable_if_t<sizeof...(Args) == NFeatures, int> = 0>
    constexpr Vertex(Args&&... args)
      : values_{{static_cast<Value>(std::forward<Args>(args))...}} {
        static_assert((std::is_convertible_v<Args, Value> && ...), "All arguments must be convertible to Value.");
    }

    constexpr Vertex(const ContainerType& values)
      : values_{values} {}

    constexpr Vertex(ContainerType&& values) noexcept
      : values_{std::move(values)} {}

    constexpr Value& operator[](std::size_t index) {
        return values_[index];
    }

    constexpr Value& operator[](std::size_t index) const {
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
    ContainerType values_;
};

}  // namespace ffcl::datastruct::bounds