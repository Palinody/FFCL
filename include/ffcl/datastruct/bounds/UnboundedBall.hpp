#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/StaticCentroidBasedBound.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <vector>

namespace ffcl::datastruct::bounds {

template <typename Value, std::size_t NFeatures = 0>
class UnboundedBall : public StaticCentroidBasedBound<UnboundedBall<Value, NFeatures>> {
  public:
    using ValueType = Value;

    using CentroidType = FeaturesVector<ValueType, NFeatures>;

    using IteratorType = typename CentroidType::IteratorType;

    UnboundedBall(const CentroidType& centroid)
      : centroid_{centroid} {}

    UnboundedBall(CentroidType&& centroid) noexcept
      : centroid_{std::move(centroid)} {}

    std::size_t n_features_impl() const {
        return centroid_.size();
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& features_range_first,
                                     const OtherFeaturesIterator& features_range_last) const {
        common::ignore_parameters(features_range_first, features_range_last);
        return true;
    }

    template <typename OtherFeaturesIterator>
    constexpr auto distance_to_centroid_impl(const OtherFeaturesIterator& features_range_first,
                                             const OtherFeaturesIterator& features_range_last) const {
        assert(centroid_.size() == std::distance(features_range_first, features_range_last));

        return common::math::heuristics::auto_distance(
            features_range_first, features_range_last, centroid_.begin(), centroid_.end());
    }

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_to_centroid_if_within_bounds_impl(
        const OtherFeaturesIterator& features_range_first,
        const OtherFeaturesIterator& features_range_last) const {
        assert(centroid_.size() == std::distance(features_range_first, features_range_last));

        return std::optional<ValueType>(distance_to_centroid_impl(features_range_first, features_range_last));
    }

    constexpr auto length_from_centroid_impl() const {
        return common::infinity<ValueType>();
    }

    constexpr auto length_from_centroid_impl(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return common::infinity<ValueType>();
    }

    constexpr auto centroid_begin_impl() const {
        return centroid_.begin();
    }

    constexpr auto centroid_end_impl() const {
        return centroid_.end();
    }

  private:
    // an unbounded ball represented as a single point and an infinite radius
    CentroidType centroid_;
};

template <typename FeaturesIterator>
class UnboundedBallView : public StaticCentroidBasedBound<UnboundedBallView<FeaturesIterator>> {
  public:
    static_assert(common::is_iterator<FeaturesIterator>::value, "FeaturesIterator is not an iterator");

    using ValueType = typename std::iterator_traits<FeaturesIterator>::value_type;

    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    using IteratorType = FeaturesIterator;

    UnboundedBallView(const FeaturesIterator& centroid_features_range_first,
                      const FeaturesIterator& centroid_features_range_last)
      : centroid_features_range_first_{centroid_features_range_first}
      , centroid_features_range_last_{centroid_features_range_last} {}

    std::size_t n_features_impl() const {
        return std::distance(centroid_features_range_first_, centroid_features_range_last_);
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& other_features_range_first,
                                     const OtherFeaturesIterator& other_features_range_last) const {
        common::ignore_parameters(other_features_range_first, other_features_range_last);
        return true;
    }

    template <typename OtherFeaturesIterator>
    constexpr auto distance_to_centroid_impl(const OtherFeaturesIterator& other_features_range_first,
                                             const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == static_cast<decltype(n_features_impl())>(
                                        std::distance(other_features_range_first, other_features_range_last)));

        return common::math::heuristics::auto_distance(other_features_range_first,
                                                       other_features_range_last,
                                                       centroid_features_range_first_,
                                                       centroid_features_range_last_);
    }

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_to_centroid_if_within_bounds_impl(
        const OtherFeaturesIterator& other_features_range_first,
        const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == static_cast<decltype(n_features_impl())>(
                                        std::distance(other_features_range_first, other_features_range_last)));

        return std::optional<ValueType>(
            distance_to_centroid_impl(other_features_range_first, other_features_range_last));
    }

    constexpr auto length_from_centroid_impl() const {
        return common::infinity<ValueType>();
    }

    constexpr auto length_from_centroid_impl(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return common::infinity<ValueType>();
    }

    constexpr auto centroid_begin_impl() const {
        return centroid_features_range_first_;
    }

    constexpr auto centroid_end_impl() const {
        return centroid_features_range_last_;
    }

  private:
    FeaturesIterator centroid_features_range_first_, centroid_features_range_last_;
};

}  // namespace ffcl::datastruct::bounds