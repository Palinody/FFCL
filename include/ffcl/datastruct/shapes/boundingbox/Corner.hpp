#pragma once

#include <array>
#include <cstddef>  // std::size_t
#include <vector>

namespace ffcl::datastruct::boundingbox {

template <typename ValueType, std::size_t Size>
class Corner;

template <typename ValueType>
class Corner<ValueType, 0> {
  public:
    Corner(const std::vector<ValueType>& values)
      : values_{values} {}

    Corner(std::vector<ValueType>&& values) noexcept
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

  private:
    std::vector<ValueType> values_;
};

template <typename ValueType, std::size_t Size>
class Corner {
  public:
    Corner(const std::array<ValueType, Size>& values)
      : values_{values} {}

    Corner(std::array<ValueType, Size>&& values) noexcept
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

  private:
    std::array<ValueType, Size> values_;
};

}  // namespace ffcl::datastruct::boundingbox