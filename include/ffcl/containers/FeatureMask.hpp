#pragma once

#include <sys/types.h>
#include <array>
#include <variant>
#include <vector>

namespace ffcl::containers {

/**
 * @brief
 *
 * @tparam NStaticFeatures
 */
template <std::size_t NStaticFeatures = 0>
class FeatureMask {
  public:
    using FeatureIndexType = std::size_t;
    using VariantType      = std::variant<std::vector<FeatureIndexType>, std::array<FeatureIndexType, NStaticFeatures>>;
    using ConstIterator    = typename std::conditional_t<NStaticFeatures == 0,
                                                      std::variant_alternative_t<0, VariantType>,
                                                      std::variant_alternative_t<1, VariantType>>::const_iterator;

    FeatureMask(const std::vector<FeatureIndexType>& vector)
      : data_{vector} {}

    FeatureMask(const std::array<FeatureIndexType, NStaticFeatures>& array)
      : data_{array} {}

    FeatureMask(FeatureMask&& feature_mask) noexcept
      : data_{std::move(feature_mask.data_)} {}

    constexpr FeatureIndexType operator[](std::size_t index) const {
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

}  // namespace ffcl::containers