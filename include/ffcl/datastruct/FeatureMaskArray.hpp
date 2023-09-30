#pragma once

#include <array>
#include <variant>
#include <vector>

namespace ffcl::datastruct {

/**
 * @brief A read-only feature mask array
 *
 * @tparam DataType
 * @tparam NFeatures
 */
template <typename DataType, std::size_t NFeatures>
class FeatureMaskArray {
  public:
    using VariantType   = std::variant<std::vector<DataType>, std::array<DataType, NFeatures>>;
    using ConstIterator = typename VariantType::value_type::const_iterator;

    FeatureMaskArray(const std::vector<DataType>& vector)
      : data_{vector} {}

    FeatureMaskArray(const std::array<DataType, NFeatures>& array)
      : data_{array} {}

    constexpr DataType operator[](std::size_t index) const {
        return data_[index];
    }

    constexpr std::size_t size() const {
        return data_.size();
    }

    constexpr ConstIterator begin() const {
        return std::visit([](const auto& data) { return data.begin(); }, data_);
    }

    constexpr ConstIterator end() const {
        return std::visit([](const auto& data) { return data.end(); }, data_);
    }

  private:
    VariantType data_;
};

}  // namespace ffcl::datastruct