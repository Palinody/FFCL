#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/Vertex.hpp"

namespace ffcl::datastruct::bounds {

template <typename ValueType, std::size_t NFeatures = 0>
class UnboundedBall {
  public:
    using CentroidType = Vertex<ValueType, NFeatures>;

    UnboundedBall(const CentroidType& centroid)
      : center_point_{centroid}
      , radius_{common::infinity<ValueType>()} {}

    UnboundedBall(CentroidType&& centroid) noexcept
      : center_point_{std::move(centroid)}
      , radius_{common::infinity<ValueType>()} {}

    std::size_t n_features() const {
        return center_point_.size();
    }

    template <typename FeaturesIterator>
    bool is_in_bounds(const FeaturesIterator& features_range_first, const FeaturesIterator& features_range_last) const {
        common::ignore_parameters(features_range_first, features_range_last);
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

        return this->distance(features_range_first, features_range_last);
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
    // an unbounded ball represented as a single point and an infinite radius
    CentroidType center_point_;
    ValueType    radius_;
};

}  // namespace ffcl::datastruct::bounds