#pragma once

#include <array>
#include <cstddef>  // std::size_t
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ffcl::datastruct {

template <typename ValueType, std::size_t NFeatures>
class FeaturesVector;

// std::unique_ptr<T[]> attempt:
/*
// Specialization for dynamic arrays
template <typename ValueType>
class FeaturesVector<ValueType, 0> {
  public:
    using ContainerType = std::unique_ptr<ValueType[]>;
    using Iterator      = ValueType*;
    using ConstIterator = const ValueType*;

    FeaturesVector(std::initializer_list<ValueType> init_list)
      : values_(make_unique_ptr_from_initializer_list(init_list))
      , size_(init_list.size()) {}

    FeaturesVector(FeaturesVector&& features_vector) noexcept
      : values_{std::move(features_vector.values_)}
      , size_{std::move(features_vector.size_)} {}

    ValueType& operator[](std::size_t index) {
        return values_[index];
    }

    const ValueType& operator[](std::size_t index) const {
        return values_[index];
    }

    FeaturesVector& operator=(FeaturesVector&& features_vector) noexcept {
        if (this != &features_vector) {
            values_ = std::move(features_vector.values_);
            size_   = std::move(features_vector.size_);
        }
        return *this;
    }

    std::size_t size() const {
        return size_;
    }

    ValueType* begin() {
        return values_.get();
    }

    ValueType* end() {
        return values_.get() + size_;
    }

    const ValueType* begin() const {
        return values_.get();
    }

    const ValueType* end() const {
        return values_.get() + size_;
    }

    const ValueType* cbegin() const {
        return values_.get();
    }

    const ValueType* cend() const {
        return values_.get() + size_;
    }

  private:
    ContainerType values_;
    std::size_t   size_;

    static std::unique_ptr<ValueType[]> make_unique_ptr_from_initializer_list(std::initializer_list<ValueType> list) {
        auto array = std::make_unique<ValueType[]>(list.size());
        std::copy(list.begin(), list.end(), array.get());
        return array;
    }
};
*/

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