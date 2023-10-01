#pragma once

#include <array>
#include <variant>
#include <vector>

namespace ffcl::datastruct {

/**
 * @brief A read-only feature mask array
 *
 * @tparam IndexType
 * @tparam NFeatures
 */
template <typename IndexType, std::size_t NFeatures>
class FeatureMaskArray {
  public:
    using VariantType          = std::variant<std::vector<IndexType>, std::array<IndexType, NFeatures>>;
    using ConstIndicesIterator = typename VariantType::value_type::const_iterator;

    FeatureMaskArray(const std::vector<IndexType>& vector)
      : data_{vector} {}

    FeatureMaskArray(const std::array<IndexType, NFeatures>& array)
      : data_{array} {}

    constexpr IndexType operator[](std::size_t index) const {
        return data_[index];
    }

    constexpr std::size_t size() const {
        return data_.size();
    }

    constexpr ConstIndicesIterator begin() const {
        return std::visit([](const auto& data) { return data.begin(); }, data_);
    }

    constexpr ConstIndicesIterator end() const {
        return std::visit([](const auto& data) { return data.end(); }, data_);
    }

  private:
    VariantType data_;
};

}  // namespace ffcl::datastruct