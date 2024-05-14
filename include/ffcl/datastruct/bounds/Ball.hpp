#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include "ffcl/common/math/heuristics/Distances.hpp"

namespace ffcl::datastruct::bounds {

template <typename Value, std::size_t NFeatures = 0>
class Ball : public StaticBoundWithCentroid<Ball<Value, NFeatures>> {
  public:
    using ValueType = Value;

    using CentroidType = FeaturesVector<ValueType, NFeatures>;

    using IteratorType = typename CentroidType::IteratorType;

    Ball(const CentroidType& centroid, const ValueType& radius)
      : centroid_{centroid}
      , radius_{radius} {}

    Ball(CentroidType&& centroid, const ValueType& radius) noexcept
      : centroid_{std::move(centroid)}
      , radius_{radius} {}

    std::size_t n_features_impl() const {
        return centroid_.size();
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& features_range_first,
                                     const OtherFeaturesIterator& features_range_last) const {
        assert(centroid_.size() == std::distance(features_range_first, features_range_last));

        return distance_to_centroid_impl(features_range_first, features_range_last) < radius_;
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

        const auto feature_distance = distance_to_centroid_impl(features_range_first, features_range_last);

        return feature_distance < radius_ ? std::optional<ValueType>(feature_distance) : std::nullopt;
    }

    constexpr auto centroid_to_bound_length_impl() const {
        return radius_;
    }

    constexpr auto centroid_to_bound_length_impl(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return radius_;
    }

    constexpr auto centroid_begin_impl() const {
        return centroid_.begin();
    }

    constexpr auto centroid_end_impl() const {
        return centroid_.end();
    }

  private:
    // a ball represented as a single point and a radius
    CentroidType centroid_;
    ValueType    radius_;
};

template <typename FeaturesIterator>
class BallView : public StaticBoundWithCentroid<BallView<FeaturesIterator>> {
  public:
    static_assert(common::is_iterator<FeaturesIterator>::value, "FeaturesIterator is not an iterator");

    using ValueType = typename std::iterator_traits<FeaturesIterator>::value_type;

    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    using IteratorType = FeaturesIterator;

    BallView(const FeaturesIterator& center_point_range_first,
             const FeaturesIterator& center_point_range_last,
             const ValueType&        radius)
      : centroid_features_range_first_{center_point_range_first}
      , centroid_features_range_last_{center_point_range_last}
      , radius_{radius} {}

    std::size_t n_features_impl() const {
        return std::distance(centroid_features_range_first_, centroid_features_range_last_);
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& other_features_range_first,
                                     const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == static_cast<decltype(n_features_impl())>(
                                        std::distance(other_features_range_first, other_features_range_last)));

        return distance_to_centroid_impl(other_features_range_first, other_features_range_last) < radius_;
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

        const auto feature_distance = distance_to_centroid_impl(other_features_range_first, other_features_range_last);

        return feature_distance < radius_ ? std::optional<ValueType>(feature_distance) : std::nullopt;
    }

    constexpr auto centroid_to_bound_length_impl() const {
        return radius_;
    }

    constexpr auto centroid_to_bound_length_impl(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return radius_;
    }

    constexpr auto centroid_begin_impl() const {
        return centroid_features_range_first_;
    }

    constexpr auto centroid_end_impl() const {
        return centroid_features_range_last_;
    }

  private:
    // a ball represented as a single point and a radius
    FeaturesIterator centroid_features_range_first_, centroid_features_range_last_;
    ValueType        radius_;
};

}  // namespace ffcl::datastruct::bounds