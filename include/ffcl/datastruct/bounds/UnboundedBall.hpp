#pragma once

#include <vector>
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

namespace ffcl::datastruct::bounds {

template <typename Value, std::size_t NFeatures = 0>
class UnboundedBall : public StaticBoundWithCentroid<UnboundedBall<Value, NFeatures>> {
  public:
    using ValueType    = Value;
    using CentroidType = FeaturesVector<ValueType, NFeatures>;
    using IteratorType = typename CentroidType::Iterator;

    UnboundedBall(const CentroidType& centroid);
    UnboundedBall(CentroidType&& centroid) noexcept;

    std::size_t n_features_impl() const;

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& other_features_range_first,
                                     const OtherFeaturesIterator& other_features_range_last) const;

    template <typename OtherFeaturesIterator>
    constexpr auto distance_to_centroid_impl(const OtherFeaturesIterator& other_features_range_first,
                                             const OtherFeaturesIterator& other_features_range_last) const;

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_to_centroid_if_within_bounds_impl(
        const OtherFeaturesIterator& other_features_range_first,
        const OtherFeaturesIterator& other_features_range_last) const;

    constexpr auto centroid_to_furthest_bound_distance_impl() const;
    constexpr auto centroid_to_bound_distance_impl(std::size_t feature_index) const;

    const auto&    centroid_impl() const;
    constexpr auto centroid_begin_impl() const;
    constexpr auto centroid_end_impl() const;

  private:
    CentroidType centroid_;
};

template <typename Value, std::size_t NFeatures>
UnboundedBall<Value, NFeatures>::UnboundedBall(const CentroidType& centroid)
  : centroid_{centroid} {}

template <typename Value, std::size_t NFeatures>
UnboundedBall<Value, NFeatures>::UnboundedBall(CentroidType&& centroid) noexcept
  : centroid_{std::move(centroid)} {}

template <typename Value, std::size_t NFeatures>
std::size_t UnboundedBall<Value, NFeatures>::n_features_impl() const {
    return centroid_.size();
}

template <typename Value, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr bool UnboundedBall<Value, NFeatures>::is_in_bounds_impl(
    const OtherFeaturesIterator& other_features_range_first,
    const OtherFeaturesIterator& other_features_range_last) const {
    common::ignore_parameters(other_features_range_first, other_features_range_last);
    return true;
}

template <typename Value, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr auto UnboundedBall<Value, NFeatures>::distance_to_centroid_impl(
    const OtherFeaturesIterator& other_features_range_first,
    const OtherFeaturesIterator& other_features_range_last) const {
    assert(centroid_.size() == std::distance(other_features_range_first, other_features_range_last));
    return common::math::heuristics::auto_distance(
        other_features_range_first, other_features_range_last, centroid_.begin(), centroid_.end());
}

template <typename Value, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr auto UnboundedBall<Value, NFeatures>::compute_distance_to_centroid_if_within_bounds_impl(
    const OtherFeaturesIterator& other_features_range_first,
    const OtherFeaturesIterator& other_features_range_last) const {
    assert(centroid_.size() == std::distance(other_features_range_first, other_features_range_last));
    return std::optional<ValueType>(distance_to_centroid_impl(other_features_range_first, other_features_range_last));
}

template <typename Value, std::size_t NFeatures>
constexpr auto UnboundedBall<Value, NFeatures>::centroid_to_furthest_bound_distance_impl() const {
    return common::infinity<ValueType>();
}

template <typename Value, std::size_t NFeatures>
constexpr auto UnboundedBall<Value, NFeatures>::centroid_to_bound_distance_impl(std::size_t feature_index) const {
    common::ignore_parameters(feature_index);
    return common::infinity<ValueType>();
}

template <typename Value, std::size_t NFeatures>
const auto& UnboundedBall<Value, NFeatures>::centroid_impl() const {
    return centroid_;
}

template <typename Value, std::size_t NFeatures>
constexpr auto UnboundedBall<Value, NFeatures>::centroid_begin_impl() const {
    return centroid_.begin();
}

template <typename Value, std::size_t NFeatures>
constexpr auto UnboundedBall<Value, NFeatures>::centroid_end_impl() const {
    return centroid_.end();
}

template <typename FeaturesIterator>
class UnboundedBallView : public StaticBoundWithCentroid<UnboundedBallView<FeaturesIterator>> {
  public:
    static_assert(common::is_iterator<FeaturesIterator>::value, "FeaturesIterator is not an iterator");
    using ValueType = typename std::iterator_traits<FeaturesIterator>::value_type;
    static_assert(std::is_trivial_v<ValueType>, "ValueType must be trivial.");

    using IteratorType = FeaturesIterator;

    UnboundedBallView(const FeaturesIterator& centroid_features_range_first,
                      const FeaturesIterator& centroid_features_range_last);

    std::size_t n_features_impl() const;

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& other_features_range_first,
                                     const OtherFeaturesIterator& other_features_range_last) const;

    template <typename OtherFeaturesIterator>
    constexpr auto distance_to_centroid_impl(const OtherFeaturesIterator& other_features_range_first,
                                             const OtherFeaturesIterator& other_features_range_last) const;

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_to_centroid_if_within_bounds_impl(
        const OtherFeaturesIterator& other_features_range_first,
        const OtherFeaturesIterator& other_features_range_last) const;

    constexpr auto centroid_to_furthest_bound_distance_impl() const;
    constexpr auto centroid_to_bound_distance_impl(std::size_t feature_index) const;

    auto           centroid_impl() const;
    constexpr auto centroid_begin_impl() const;
    constexpr auto centroid_end_impl() const;

  private:
    FeaturesIterator centroid_features_range_first_, centroid_features_range_last_;
};

template <typename FeaturesIterator>
UnboundedBallView<FeaturesIterator>::UnboundedBallView(const FeaturesIterator& centroid_features_range_first,
                                                       const FeaturesIterator& centroid_features_range_last)
  : centroid_features_range_first_{centroid_features_range_first}
  , centroid_features_range_last_{centroid_features_range_last} {}

template <typename FeaturesIterator>
std::size_t UnboundedBallView<FeaturesIterator>::n_features_impl() const {
    return std::distance(centroid_features_range_first_, centroid_features_range_last_);
}

template <typename FeaturesIterator>
template <typename OtherFeaturesIterator>
constexpr bool UnboundedBallView<FeaturesIterator>::is_in_bounds_impl(
    const OtherFeaturesIterator& other_features_range_first,
    const OtherFeaturesIterator& other_features_range_last) const {
    common::ignore_parameters(other_features_range_first, other_features_range_last);
    return true;
}

template <typename FeaturesIterator>
template <typename OtherFeaturesIterator>
constexpr auto UnboundedBallView<FeaturesIterator>::distance_to_centroid_impl(
    const OtherFeaturesIterator& other_features_range_first,
    const OtherFeaturesIterator& other_features_range_last) const {
    assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));
    return common::math::heuristics::auto_distance(other_features_range_first,
                                                   other_features_range_last,
                                                   centroid_features_range_first_,
                                                   centroid_features_range_last_);
}

template <typename FeaturesIterator>
template <typename OtherFeaturesIterator>
constexpr auto UnboundedBallView<FeaturesIterator>::compute_distance_to_centroid_if_within_bounds_impl(
    const OtherFeaturesIterator& other_features_range_first,
    const OtherFeaturesIterator& other_features_range_last) const {
    assert(n_features_impl() == std::distance(other_features_range_first, other_features_range_last));
    return std::optional<ValueType>(distance_to_centroid_impl(other_features_range_first, other_features_range_last));
}

template <typename FeaturesIterator>
constexpr auto UnboundedBallView<FeaturesIterator>::centroid_to_furthest_bound_distance_impl() const {
    return common::infinity<ValueType>();
}

template <typename FeaturesIterator>
constexpr auto UnboundedBallView<FeaturesIterator>::centroid_to_bound_distance_impl(std::size_t feature_index) const {
    common::ignore_parameters(feature_index);
    return common::infinity<ValueType>();
}

template <typename FeaturesIterator>
auto UnboundedBallView<FeaturesIterator>::centroid_impl() const {
    return std::vector(centroid_features_range_first_, centroid_features_range_last_);
}

template <typename FeaturesIterator>
constexpr auto UnboundedBallView<FeaturesIterator>::centroid_begin_impl() const {
    return centroid_features_range_first_;
}

template <typename FeaturesIterator>
constexpr auto UnboundedBallView<FeaturesIterator>::centroid_end_impl() const {
    return centroid_features_range_last_;
}

}  // namespace ffcl::datastruct::bounds