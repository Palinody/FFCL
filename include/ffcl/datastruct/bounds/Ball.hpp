#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/StaticBound.hpp"
#include "ffcl/datastruct/bounds/Vertex.hpp"

#include "ffcl/common/math/heuristics/Distances.hpp"

namespace ffcl::datastruct::bounds {

template <typename ValueType, std::size_t NFeatures = 0>
class Ball {
  public:
    using CentroidType = Vertex<ValueType, NFeatures>;

    Ball(const CentroidType& centroid, const ValueType& radius)
      : center_point_{centroid}
      , radius_{radius} {}

    Ball(CentroidType&& centroid, const ValueType& radius) noexcept
      : center_point_{std::move(centroid)}
      , radius_{radius} {}

    std::size_t n_features() const {
        return center_point_.size();
    }

    template <typename FeaturesIterator>
    bool is_in_bounds(const FeaturesIterator& features_range_first, const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        return this->distance(features_range_first, features_range_last) < radius_;
    }

    template <typename FeaturesIterator>
    ValueType distance(const FeaturesIterator& features_range_first,
                       const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        return common::math::heuristics::auto_distance(
            features_range_first, features_range_last, center_point_.begin());
    }

    template <typename FeaturesIterator>
    std::optional<ValueType> compute_distance_within_bounds(const FeaturesIterator& features_range_first,
                                                            const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        if (this->is_in_bounds(features_range_first, features_range_last)) {
            return this->distance(features_range_first, features_range_last);

        } else {
            return std::nullopt;
        }
    }

    ValueType length_from_centroid() const {
        return radius_;
    }

    constexpr ValueType length_from_centroid(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return radius_;
    }

    const CentroidType& centroid() const {
        return center_point_;
    }

    CentroidType make_centroid() const {
        return center_point_;
    }

  private:
    // a ball represented as a single point and a radius
    CentroidType center_point_;
    ValueType    radius_;
};

template <typename ValueType, std::size_t NFeatures = 0>
class StaticBall : public StaticBound<StaticBall<ValueType, NFeatures>> {
  public:
    using CentroidType = Vertex<ValueType, NFeatures>;

    StaticBall(const CentroidType& centroid, const ValueType& radius)
      : center_point_{centroid}
      , radius_{radius} {}

    StaticBall(CentroidType&& centroid, const ValueType& radius) noexcept
      : center_point_{std::move(centroid)}
      , radius_{radius} {}

    std::size_t n_features_impl() const {
        return center_point_.size();
    }

    template <typename FeaturesIterator>
    constexpr bool is_in_bounds_impl(const FeaturesIterator& features_range_first,
                                     const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        return distance_impl(features_range_first, features_range_last) < radius_;
    }

    template <typename FeaturesIterator>
    constexpr auto distance_impl(const FeaturesIterator& features_range_first,
                                 const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        return common::math::heuristics::auto_distance(
            features_range_first, features_range_last, center_point_.begin());
    }

    template <typename FeaturesIterator>
    constexpr auto compute_distance_if_within_bounds_impl(const FeaturesIterator& features_range_first,
                                                          const FeaturesIterator& features_range_last) const {
        assert(center_point_.size() == std::distance(features_range_first, features_range_last));

        const auto feature_distance = distance_impl(features_range_first, features_range_last);

        return feature_distance < radius_ ? std::optional<ValueType>(feature_distance) : std::nullopt;
    }

    constexpr auto length_from_centroid_impl() const {
        return radius_;
    }

    constexpr auto length_from_centroid_impl(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return radius_;
    }

    constexpr auto& centroid_reference_impl() const {
        return center_point_;
    }

    constexpr auto make_centroid_impl() const {
        return center_point_;
    }

  private:
    // a ball represented as a single point and a radius
    CentroidType center_point_;
    ValueType    radius_;
};

template <typename FeaturesIterator>
class StaticBallView : public StaticBound<StaticBallView<FeaturesIterator>> {
  public:
    static_assert(common::is_iterator<FeaturesIterator>::value, "FeaturesIterator is not an iterator");

    using ValueType = typename std::iterator_traits<FeaturesIterator>::value_type;

    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    StaticBallView(FeaturesIterator center_point_range_first,
                   FeaturesIterator center_point_range_last,
                   const ValueType& radius)
      : features_range_first_{center_point_range_first}
      , center_point_range_last_{center_point_range_last}
      , radius_{radius} {}

    std::size_t n_features_impl() const {
        return std::distance(features_range_first_, center_point_range_last_);
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& other_features_range_first,
                                     const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));

        return distance_impl(other_features_range_first, other_features_range_last) < radius_;
    }

    template <typename OtherFeaturesIterator>
    constexpr auto distance_impl(const OtherFeaturesIterator& other_features_range_first,
                                 const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));

        return common::math::heuristics::auto_distance(
            other_features_range_first, other_features_range_last, features_range_first_);
    }

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_if_within_bounds_impl(
        const OtherFeaturesIterator& other_features_range_first,
        const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));

        const auto feature_distance = distance_impl(other_features_range_first, other_features_range_last);

        return feature_distance < radius_ ? std::optional<ValueType>(feature_distance) : std::nullopt;
    }

    constexpr auto length_from_centroid_impl() const {
        return radius_;
    }

    constexpr auto length_from_centroid_impl(std::size_t feature_index) const {
        common::ignore_parameters(feature_index);
        return radius_;
    }

    constexpr auto& centroid_reference_impl() const {
        // always_false<Derived>::value is dependent on the template parameter FeaturesIterator. This means that
        // static_assert will only be evaluated when centroid_reference_impl is instantiated with a specific type,
        // allowing the base template to compile successfully until an attempt is made to instantiate this method.
        static_assert(always_false<FeaturesIterator>::value,
                      "centroid_reference_impl cannot be implemented for view data.");
        // The following return statement is unreachable but required to avoid
        // compile errors in some compilers. Use a dummy return or throw an exception.
        throw std::logic_error("Unimplemented method");
    }

    constexpr auto make_centroid_impl() const {
        return std::vector(features_range_first_, center_point_range_last_);
    }

  private:
    // a ball represented as a single point and a radius
    FeaturesIterator features_range_first_, center_point_range_last_;
    ValueType        radius_;
};

}  // namespace ffcl::datastruct::bounds