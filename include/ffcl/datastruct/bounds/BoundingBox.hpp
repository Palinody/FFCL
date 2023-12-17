#pragma once

#include <optional>
#include <vector>

#include "ffcl/datastruct/bounds/StaticBound.hpp"
#include "ffcl/datastruct/bounds/Vertex.hpp"

namespace ffcl::datastruct::bounds {

template <typename Segment>
class BoundingBox {
  public:
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = std::vector<Segment>;

    using CentroidType        = Vertex<ValueType, 0>;
    using LengthsFromCentroid = CentroidType;

    BoundingBox(const SegmentsType& segments)
      : center_point_{std::vector<ValueType>(segments.size())}
      , lengths_from_center_point_{std::vector<ValueType>(segments.size())} {
        for (std::size_t feature_index = 0; feature_index < segments.size(); ++feature_index) {
            center_point_[feature_index]              = segments[feature_index].centroid();
            lengths_from_center_point_[feature_index] = segments[feature_index].length_from_centroid();
        }
    }

    std::size_t n_features() const {
        return center_point_.size();
    }

    template <typename FeaturesIterator>
    bool is_in_bounds(const FeaturesIterator& features_range_first, const FeaturesIterator& features_range_last) const {
        const std::size_t n_features = std::distance(features_range_first, features_range_last);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // A sample is inside the bounding box if p is in [lo, hi]
            if (features_range_first[feature_index] <
                    center_point_[feature_index] - lengths_from_center_point_[feature_index] ||
                features_range_first[feature_index] >
                    center_point_[feature_index] + lengths_from_center_point_[feature_index]) {
                return false;
            }
        }
        return true;
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
        throw std::runtime_error("No half length to return if no feature dimension is specified for this bound.");
        return ValueType{};
    }

    constexpr ValueType length_from_centroid(std::size_t feature_index) const {
        return lengths_from_center_point_[feature_index];
    }

    const CentroidType& centroid() const {
        return center_point_;
    }

    CentroidType make_centroid() const {
        return center_point_;
    }

  private:
    // a bounding box represented as a center point coordinate and and the relative lengths from that point to the axis
    // aligned bounds w.r.t. each feature dimension
    CentroidType        center_point_;
    LengthsFromCentroid lengths_from_center_point_;
};

template <typename Segment>
class StaticBoundingBox : public StaticBound<StaticBoundingBox<Segment>> {
  public:
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = std::vector<Segment>;

    using CentroidType        = Vertex<ValueType, 0>;
    using LengthsFromCentroid = CentroidType;

    StaticBoundingBox(const SegmentsType& segments)
      : center_point_{std::vector<ValueType>(segments.size())}
      , lengths_from_center_point_{std::vector<ValueType>(segments.size())} {
        for (std::size_t feature_index = 0; feature_index < segments.size(); ++feature_index) {
            center_point_[feature_index]              = segments[feature_index].centroid();
            lengths_from_center_point_[feature_index] = segments[feature_index].length_from_centroid();
        }
    }

    std::size_t n_features_impl() const {
        return center_point_.size();
    }

    template <typename FeaturesIterator>
    constexpr bool is_in_bounds_impl(const FeaturesIterator& features_range_first,
                                     const FeaturesIterator& features_range_last) const {
        const std::size_t n_features = std::distance(features_range_first, features_range_last);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // A sample is inside the bounding box if p is in [lo, hi]
            if (features_range_first[feature_index] <
                    center_point_[feature_index] - lengths_from_center_point_[feature_index] ||
                features_range_first[feature_index] >
                    center_point_[feature_index] + lengths_from_center_point_[feature_index]) {
                return false;
            }
        }
        return true;
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

        return is_in_bounds_impl(features_range_first, features_range_last)
                   ? std::optional<ValueType>(distance_impl(features_range_first, features_range_last))
                   : std::nullopt;
    }

    constexpr auto length_from_centroid_impl() const {
        throw std::runtime_error("No half length to return if no feature dimension is specified for this bound.");
        return ValueType{};
    }

    constexpr auto length_from_centroid_impl(std::size_t feature_index) const {
        return lengths_from_center_point_[feature_index];
    }

    constexpr auto& centroid_reference_impl() const {
        return center_point_;
    }

    constexpr auto make_centroid_impl() const {
        return center_point_;
    }

  private:
    // a bounding box represented as a center point coordinate and and the relative lengths from that point to the axis
    // aligned bounds w.r.t. each feature dimension
    CentroidType        center_point_;
    LengthsFromCentroid lengths_from_center_point_;
};

template <typename FeaturesIterator>
class StaticBoundingBoxView : public StaticBound<StaticBoundingBoxView<FeaturesIterator>> {
  public:
    static_assert(common::is_iterator<FeaturesIterator>::value, "FeaturesIterator is not an iterator");

    using ValueType = typename std::iterator_traits<FeaturesIterator>::value_type;

    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    using LengthsFromCentroidType = std::vector<ValueType>;

    explicit StaticBoundingBoxView(FeaturesIterator               center_point_features_range_first,
                                   FeaturesIterator               center_point_features_range_last,
                                   const LengthsFromCentroidType& lengths_from_center)
      : StaticBoundingBoxView(center_point_features_range_first,
                              center_point_features_range_last,
                              LengthsFromCentroidType{lengths_from_center}) {}

    explicit StaticBoundingBoxView(FeaturesIterator          center_point_features_range_first,
                                   FeaturesIterator          center_point_features_range_last,
                                   LengthsFromCentroidType&& lengths_from_center)
      : center_point_features_range_first_{center_point_features_range_first}
      , center_point_features_range_last_{center_point_features_range_last}
      , lengths_from_center_point_{std::forward<LengthsFromCentroidType>(lengths_from_center)} {}

    std::size_t n_features_impl() const {
        return std::distance(center_point_features_range_first_, center_point_features_range_last_);
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& other_features_range_first,
                                     const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));

        const std::size_t n_features = std::distance(other_features_range_first, other_features_range_last);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // A sample is inside the bounding box if p is in [lo, hi]
            if (other_features_range_first[feature_index] <
                    center_point_features_range_first_[feature_index] - lengths_from_center_point_[feature_index] ||
                other_features_range_first[feature_index] >
                    center_point_features_range_first_[feature_index] + lengths_from_center_point_[feature_index]) {
                return false;
            }
        }
        return true;
    }

    template <typename OtherFeaturesIterator>
    constexpr auto distance_impl(const OtherFeaturesIterator& other_features_range_first,
                                 const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));

        return common::math::heuristics::auto_distance(
            other_features_range_first, other_features_range_last, center_point_features_range_first_);
    }

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_if_within_bounds_impl(
        const OtherFeaturesIterator& other_features_range_first,
        const OtherFeaturesIterator& other_features_range_last) const {
        assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));

        return is_in_bounds_impl(other_features_range_first, other_features_range_last)
                   ? std::optional<ValueType>(distance_impl(other_features_range_first, other_features_range_last))
                   : std::nullopt;
    }

    constexpr auto length_from_centroid_impl() const {
        // always_false<Derived>::value is dependent on the template parameter FeaturesIterator. This means that
        // static_assert will only be evaluated when length_from_centroid_impl is instantiated with a specific type,
        // allowing the base template to compile successfully until an attempt is made to instantiate this method.
        static_assert(always_false<FeaturesIterator>::value,
                      "length_from_centroid_impl cannot be implemented without specifying which feature index.");
        // The following return statement is unreachable but required to avoid
        // compile errors in some compilers. Use a dummy return or throw an exception.
        throw std::logic_error("Unimplemented method");
    }

    constexpr auto length_from_centroid_impl(std::size_t feature_index) const {
        return lengths_from_center_point_[feature_index];
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
        return std::vector(center_point_features_range_first_, center_point_features_range_last_);
    }

  private:
    // a bounding box represented as a reference center point range along each feature dimension
    FeaturesIterator center_point_features_range_first_, center_point_features_range_last_;
    // and the lengths from the reference center point along each feature dimension
    LengthsFromCentroidType lengths_from_center_point_;
};

}  // namespace ffcl::datastruct::bounds