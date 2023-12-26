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
    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");

    using VariantType          = std::variant<std::vector<IndexType>, std::array<IndexType, NFeatures>>;
    using ConstIndicesIterator = typename VariantType::value_type::const_iterator;

    FeatureMaskArray(const std::vector<IndexType>& vector)
      : indices_{vector} {}

    FeatureMaskArray(const std::array<IndexType, NFeatures>& array)
      : indices_{array} {}

    constexpr IndexType operator[](std::size_t index) const {
        return indices_[index];
    }

    constexpr std::size_t size() const {
        return indices_.size();
    }

    constexpr ConstIndicesIterator begin() const {
        return std::visit([](const auto& data) { return data.begin(); }, indices_);
    }

    constexpr ConstIndicesIterator end() const {
        return std::visit([](const auto& data) { return data.end(); }, indices_);
    }

  private:
    VariantType indices_;
};

}  // namespace ffcl::datastruct