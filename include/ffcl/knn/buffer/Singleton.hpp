#include "ffcl/knn/buffer/Base.hpp"

#include "ffcl/common/Utils.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <iostream>
#include <stdexcept>  // std::runtime_error
#include <tuple>

namespace ffcl::knn::buffer {

template <typename IndexType, typename DistanceType>
class Singleton : public Base<IndexType, DistanceType> {
  public:
    using IndicesType   = typename Base<IndexType, DistanceType>::IndicesType;
    using DistancesType = typename Base<IndexType, DistanceType>::DistancesType;

    Singleton()
      : index_{common::utils::infinity<IndexType>()}
      , distance_{common::utils::infinity<DistanceType>()} {}

    std::size_t size() const {
        return common::utils::equality(index_, common::utils::infinity<IndexType>()) ? 0 : 1;
    }

    std::size_t n_free_slots() const {
        return common::utils::equality(index_, common::utils::infinity<IndexType>()) ? 1 : 0;
    }

    bool empty() const {
        return common::utils::equality(index_, common::utils::infinity<IndexType>());
    }

    IndexType furthest_k_nearest_neighbor_index() const {
        assert(common::utils::inequality(index_, common::utils::infinity<IndexType>()));
        return index_;
    }

    DistanceType upper_bound() const {
        return distance_;
    }

    DistanceType upper_bound(const IndexType& feature_index) const {
        common::utils::ignore_parameters(feature_index);
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

    void update(const IndexType&    index_candidate,
                const DistanceType& distance_candidate,
                const IndexType&    feature_index) {
        common::utils::ignore_parameters(feature_index);
        this->update(index_candidate, distance_candidate);
    }

    void print() const {
        std::cout << "(" << index_ << ", " << distance_ << ")\n";
    }

  private:
    IndexType    index_;
    DistanceType distance_;
};

}  // namespace ffcl::knn::buffer