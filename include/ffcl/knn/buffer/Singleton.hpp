#include "ffcl/knn/buffer/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>

namespace ffcl::knn::buffer {

template <typename IndicesIterator, typename DistancesIterator>
class Singleton : public Base<IndicesIterator, DistancesIterator> {
  public:
    using IndexType     = typename Base<IndicesIterator, DistancesIterator>::IndexType;
    using DistanceType  = typename Base<IndicesIterator, DistancesIterator>::DistanceType;
    using IndicesType   = typename Base<IndicesIterator, DistancesIterator>::IndicesType;
    using DistancesType = typename Base<IndicesIterator, DistancesIterator>::DistancesType;

    using SamplesIterator = typename Base<IndicesIterator, DistancesIterator>::SamplesIterator;

    Singleton()
      : index_{common::infinity<IndexType>()}
      , distance_{common::infinity<DistanceType>()} {}

    std::size_t size() const {
        return common::equality(index_, common::infinity<IndexType>()) ? 0 : 1;
    }

    std::size_t n_free_slots() const {
        return common::equality(index_, common::infinity<IndexType>()) ? 1 : 0;
    }

    bool empty() const {
        return common::equality(index_, common::infinity<IndexType>());
    }

    IndexType upper_bound_index() const {
        assert(common::inequality(index_, common::infinity<IndexType>()));
        return index_;
    }

    DistanceType upper_bound() const {
        return distance_;
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        common::ignore_parameters(feature_index);
        return this->upper_bound();
    }

    IndicesType indices() const {
        return IndicesType{index_};
    }

    DistancesType distances() const {
        return DistancesType{distance_};
    }

    IndicesType move_indices() {
        return this->indices();
    }

    DistancesType move_distances() {
        return this->distances();
    }

    std::tuple<IndicesType, DistancesType> move_data_to_indices_distances_pair() {
        return std::make_tuple(this->indices(), this->distances());
    }

    auto closest_neighbor_index_distance_pair() {
        return std::make_pair(index_, distance_);
    }

    void update(const IndexType& index_candidate, const DistanceType& distance_candidate) {
        if (distance_candidate < this->upper_bound()) {
            index_    = index_candidate;
            distance_ = distance_candidate;
        }
    }

    void operator()(const IndicesIterator& indices_range_first,
                    const IndicesIterator& indices_range_last,
                    const SamplesIterator& samples_range_first,
                    const SamplesIterator& samples_range_last,
                    std::size_t            n_features,
                    std::size_t            sample_index_query) {
        common::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

            if (candidate_nearest_neighbor_index != sample_index_query) {
                const auto candidate_nearest_neighbor_distance = common::math::heuristics::auto_distance(
                    samples_range_first + sample_index_query * n_features,
                    samples_range_first + sample_index_query * n_features + n_features,
                    samples_range_first + candidate_nearest_neighbor_index * n_features);

                this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
            }
        }
    }

    void operator()(const IndicesIterator& indices_range_first,
                    const IndicesIterator& indices_range_last,
                    const SamplesIterator& samples_range_first,
                    const SamplesIterator& samples_range_last,
                    std::size_t            n_features,
                    const SamplesIterator& feature_query_range_first,
                    const SamplesIterator& feature_query_range_last) {
        common::ignore_parameters(samples_range_last);

        const std::size_t n_samples = std::distance(indices_range_first, indices_range_last);

        for (std::size_t index = 0; index < n_samples; ++index) {
            const std::size_t candidate_nearest_neighbor_index = indices_range_first[index];

            const auto candidate_nearest_neighbor_distance = common::math::heuristics::auto_distance(
                feature_query_range_first,
                feature_query_range_last,
                samples_range_first + candidate_nearest_neighbor_index * n_features);

            this->update(candidate_nearest_neighbor_index, candidate_nearest_neighbor_distance);
        }
    }

    void print() const {
        std::cout << "(" << index_ << ", " << distance_ << ")\n";
    }

  private:
    IndexType    index_;
    DistanceType distance_;
};

}  // namespace ffcl::knn::buffer