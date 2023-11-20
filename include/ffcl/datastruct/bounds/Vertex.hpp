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
    using ArrayType = std::vector<ValueType>;

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
    using ArrayType = std::array<ValueType, NFeatures>;

    Vertex(std::initializer_list<ValueType> init_list) {
        if (init_list.size() != NFeatures) {
            throw std::length_error("Initializer list length does not match NFeatures");
        }
        std::copy(init_list.begin(), init_list.end(), values_.begin());
    }

    Vertex(const std::array<ValueType, NFeatures>& values)
      : values_{values} {}

    Vertex(std::array<ValueType, NFeatures>&& values) noexcept
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