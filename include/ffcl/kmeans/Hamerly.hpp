#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/kmeans/KMeansUtils.hpp"
#include "ffcl/math/heuristics/Distances.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

#include <tuple>
#include <vector>

namespace ffcl {

template <typename SamplesIterator>
class Hamerly {
    static_assert(std::is_floating_point_v<typename SamplesIterator::value_type>,
                  "Hamerly allows floating point types.");

  public:
    using DataType = typename SamplesIterator::value_type;

    // pointers/iterators to the first and last elements of the dataset and the feature size
    using DatasetDescriptorType = std::tuple<SamplesIterator, SamplesIterator, std::size_t>;

    Hamerly(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids);

    Hamerly(const DatasetDescriptorType& dataset_descriptor,
            const std::vector<DataType>& centroids,
            const DataType&              loss);

    Hamerly(const Hamerly&) = delete;

    auto total_deviation() const;

    auto step();

  private:
    struct Buffers {
        Buffers(const SamplesIterator&       samples_first,
                const SamplesIterator&       samples_last,
                std::size_t                  n_features,
                const std::vector<DataType>& centroids);

        Buffers(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids);

        Buffers(const Buffers&) = delete;

        std::vector<std::size_t> samples_to_nearest_centroid_indices_;
        std::vector<DataType>    samples_to_nearest_centroid_distances_;
        std::vector<DataType>    samples_to_second_nearest_centroid_distances_;

        std::vector<DataType> centroid_to_nearest_centroid_distances_;

        std::vector<std::size_t> cluster_sizes_;
        std::vector<DataType>    cluster_position_sums_;
        std::vector<DataType>    centroid_velocities_;
    };

    void swap_bounds();

    void update_centroids();

    void update_centroids_velocities(const std::vector<DataType>& previous_centroids);

    auto update_bounds();

    DatasetDescriptorType    dataset_descriptor_;
    std::size_t              n_samples_;
    std::vector<DataType>    centroids_;
    std::unique_ptr<Buffers> buffers_ptr_;
    DataType                 loss_;
};

template <typename SamplesIterator>
Hamerly<SamplesIterator>::Hamerly(const DatasetDescriptorType& dataset_descriptor,
                                  const std::vector<DataType>& centroids)
  : Hamerly<SamplesIterator>::Hamerly(dataset_descriptor, centroids, common::utils::infinity<DataType>()) {
    // compute initial loss
    loss_ = std::reduce(buffers_ptr_->samples_to_nearest_centroid_distances_.begin(),
                        buffers_ptr_->samples_to_nearest_centroid_distances_.end(),
                        static_cast<DataType>(0),
                        std::plus<>());
}

template <typename SamplesIterator>
Hamerly<SamplesIterator>::Hamerly(const DatasetDescriptorType& dataset_descriptor,
                                  const std::vector<DataType>& centroids,
                                  const DataType&              loss)
  : dataset_descriptor_{dataset_descriptor}
  , n_samples_{common::utils::get_n_samples(std::get<0>(dataset_descriptor_),
                                            std::get<1>(dataset_descriptor_),
                                            std::get<2>(dataset_descriptor_))}
  , centroids_{centroids}
  , buffers_ptr_{std::make_unique<Buffers>(dataset_descriptor, centroids_)}
  , loss_{loss} {}

template <typename SamplesIterator>
auto Hamerly<SamplesIterator>::total_deviation() const {
    return loss_;
}

template <typename SamplesIterator>
auto Hamerly<SamplesIterator>::step() {
    // iterate over all the samples and swap the lower and upper bounds only if necessary
    swap_bounds();

    // keep a copy of the current non updated centroids
    const auto previous_centroids = centroids_;

    // update all the centroids with the new intra-cluster positions sum and cluster sizes
    update_centroids();

    // upate the centroids velocities based on the previous centroids and the updated centroids
    update_centroids_velocities(previous_centroids);

    // recompute the loss w.r.t. the updated buffers
    loss_ = update_bounds();

    return centroids_;
}

template <typename SamplesIterator>
void Hamerly<SamplesIterator>::swap_bounds() {
    auto& samples_to_nearest_centroid_indices          = buffers_ptr_->samples_to_nearest_centroid_indices_;
    auto& samples_to_nearest_centroid_distances        = buffers_ptr_->samples_to_nearest_centroid_distances_;
    auto& samples_to_second_nearest_centroid_distances = buffers_ptr_->samples_to_second_nearest_centroid_distances_;

    const auto& centroid_to_nearest_centroid_distances = buffers_ptr_->centroid_to_nearest_centroid_distances_;

    auto& cluster_sizes         = buffers_ptr_->cluster_sizes_;
    auto& cluster_position_sums = buffers_ptr_->cluster_position_sums_;

    for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
        auto assigned_centroid_index = samples_to_nearest_centroid_indices[sample_index];
        // triangular inequality
        const auto upper_bound_comparison =
            std::max(static_cast<DataType>(0.5) * centroid_to_nearest_centroid_distances[assigned_centroid_index],
                     samples_to_second_nearest_centroid_distances[sample_index]);
        // first bound test
        if (samples_to_nearest_centroid_distances[sample_index] > upper_bound_comparison) {
            const auto [samples_first, samples_last, n_features] = dataset_descriptor_;
            // tighten upper bound
            auto upper_bound =
                math::heuristics::auto_distance(samples_first + sample_index * n_features,
                                                samples_first + sample_index * n_features + n_features,
                                                centroids_.begin() + assigned_centroid_index * n_features);

            const auto previous_assigned_centroid_distance = samples_to_nearest_centroid_distances[sample_index];

            samples_to_nearest_centroid_distances[sample_index] = upper_bound;

            // second bound test
            if (samples_to_nearest_centroid_distances[sample_index] > upper_bound_comparison) {
                auto lower_bound = common::utils::infinity<DataType>();

                const std::size_t n_centroids =
                    common::utils::get_n_samples(centroids_.begin(), centroids_.end(), n_features);

                for (std::size_t other_centroid_index = 0; other_centroid_index < n_centroids; ++other_centroid_index) {
                    if (other_centroid_index != assigned_centroid_index) {
                        const auto other_nearest_candidate =
                            math::heuristics::auto_distance(samples_first + sample_index * n_features,
                                                            samples_first + sample_index * n_features + n_features,
                                                            centroids_.begin() + other_centroid_index * n_features);

                        // if another center is closer than the current assignment
                        if (other_nearest_candidate < upper_bound) {
                            // change the lower bound to be the current upper bound
                            lower_bound = upper_bound;
                            // adjust the upper bound
                            upper_bound = other_nearest_candidate;
                            // adjust the current assignment
                            assigned_centroid_index = other_centroid_index;

                        } else if (other_nearest_candidate < lower_bound) {
                            // reduce the lower bound to the second nearest centroid
                            lower_bound = other_nearest_candidate;
                        }
                    }
                }
                samples_to_second_nearest_centroid_distances[sample_index] = lower_bound;

                // if the assignment for sample_index has changed
                if (samples_to_nearest_centroid_indices[sample_index] != assigned_centroid_index) {
                    samples_to_nearest_centroid_distances[sample_index] = upper_bound;

                    const auto previous_assigned_centroid_index = samples_to_nearest_centroid_indices[sample_index];

                    --cluster_sizes[previous_assigned_centroid_index];
                    ++cluster_sizes[assigned_centroid_index];
                    samples_to_nearest_centroid_indices[sample_index] = assigned_centroid_index;

                    // subtract the current sample to the centroid it was previously assigned to
                    std::transform(
                        cluster_position_sums.begin() + previous_assigned_centroid_index * n_features,
                        cluster_position_sums.begin() + previous_assigned_centroid_index * n_features + n_features,
                        samples_first + sample_index * n_features,
                        cluster_position_sums.begin() + previous_assigned_centroid_index * n_features,
                        std::minus<>());

                    // add the current sample to the centroid it is now assigned to
                    std::transform(cluster_position_sums.begin() + assigned_centroid_index * n_features,
                                   cluster_position_sums.begin() + assigned_centroid_index * n_features + n_features,
                                   samples_first + sample_index * n_features,
                                   cluster_position_sums.begin() + assigned_centroid_index * n_features,
                                   std::plus<>());

                    // update the loss by removing its previous contribution and adding the new one
                    loss_ -= previous_assigned_centroid_distance;
                    loss_ += upper_bound;
                }
            }
        }
    }
}

template <typename SamplesIterator>
void Hamerly<SamplesIterator>::update_centroids() {
    const std::size_t n_features  = std::get<2>(dataset_descriptor_);
    const std::size_t n_centroids = common::utils::get_n_samples(centroids_.begin(), centroids_.end(), n_features);

    const auto& cluster_sizes         = buffers_ptr_->cluster_sizes_;
    const auto& cluster_position_sums = buffers_ptr_->cluster_position_sums_;

    // Update the centroids using the assigned samples
    for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
        const auto feature_index_start = centroid_index * n_features;
        const auto feature_index_end   = feature_index_start + n_features;

        if (cluster_sizes[centroid_index]) {
            // Compute the new centroid position for the centroid that has more than 1 associated sample
            std::transform(cluster_position_sums.begin() + feature_index_start,
                           cluster_position_sums.begin() + feature_index_end,
                           centroids_.begin() + feature_index_start,
                           [cluster_size = cluster_sizes[centroid_index]](const auto& sum) {
                               return sum / static_cast<DataType>(cluster_size);
                           });
        }
    }
}

template <typename SamplesIterator>
void Hamerly<SamplesIterator>::update_centroids_velocities(const std::vector<DataType>& previous_centroids) {
    const std::size_t n_features  = std::get<2>(dataset_descriptor_);
    const std::size_t n_centroids = common::utils::get_n_samples(centroids_.begin(), centroids_.end(), n_features);

    auto& centroid_velocities = buffers_ptr_->centroid_velocities_;

    // compute the distances between the non updated and updated centroids
    for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
        centroid_velocities[centroid_index] =
            math::heuristics::auto_distance(previous_centroids.begin() + centroid_index * n_features,
                                            previous_centroids.begin() + centroid_index * n_features + n_features,
                                            centroids_.begin() + centroid_index * n_features);
    }
}

template <typename SamplesIterator>
auto Hamerly<SamplesIterator>::update_bounds() {
    const auto [furthest_moving_centroid_index, furthest_moving_centroid_distance] =
        math::statistics::get_max_index_value_pair(buffers_ptr_->centroid_velocities_.begin(),
                                                   buffers_ptr_->centroid_velocities_.end());

    const auto [second_furthest_moving_centroid_index, second_furthest_moving_centroid_distance] =
        math::statistics::get_second_max_index_value_pair(buffers_ptr_->centroid_velocities_.begin(),
                                                          buffers_ptr_->centroid_velocities_.end());

    const auto& samples_to_nearest_centroid_indices    = buffers_ptr_->samples_to_nearest_centroid_indices_;
    auto&       samples_to_nearest_centroid_distances  = buffers_ptr_->samples_to_nearest_centroid_distances_;
    auto& samples_to_second_nearest_centroid_distances = buffers_ptr_->samples_to_second_nearest_centroid_distances_;
    const auto& centroid_velocities                    = buffers_ptr_->centroid_velocities_;

    for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
        const auto assigned_centroid_index = samples_to_nearest_centroid_indices[sample_index];
        // move the upper bound by the same distance its assigned centroid has moved
        samples_to_nearest_centroid_distances[sample_index] += centroid_velocities[assigned_centroid_index];

        samples_to_second_nearest_centroid_distances[sample_index] -=
            (assigned_centroid_index == furthest_moving_centroid_index) ? second_furthest_moving_centroid_distance
                                                                        : furthest_moving_centroid_distance;
    }
    buffers_ptr_->centroid_to_nearest_centroid_distances_ = kmeans::utils::nearest_neighbor_distances(
        centroids_.begin(), centroids_.end(), std::get<2>(dataset_descriptor_));

    return loss_;
}

template <typename SamplesIterator>
Hamerly<SamplesIterator>::Buffers::Buffers(const SamplesIterator&       samples_first,
                                           const SamplesIterator&       samples_last,
                                           std::size_t                  n_features,
                                           const std::vector<DataType>& centroids)
  : samples_to_nearest_centroid_indices_{kmeans::utils::samples_to_nearest_centroid_indices(samples_first,
                                                                                            samples_last,
                                                                                            n_features,
                                                                                            centroids)}
  , samples_to_nearest_centroid_distances_{kmeans::utils::samples_to_nearest_centroid_distances(samples_first,
                                                                                                samples_last,
                                                                                                n_features,
                                                                                                centroids)}
  , samples_to_second_nearest_centroid_distances_{kmeans::utils::samples_to_second_nearest_centroid_distances(
        samples_first,
        samples_last,
        n_features,
        centroids)}
  , centroid_to_nearest_centroid_distances_{kmeans::utils::nearest_neighbor_distances(centroids.begin(),
                                                                                      centroids.end(),
                                                                                      n_features)}
  , cluster_sizes_{kmeans::utils::compute_cluster_sizes(samples_to_nearest_centroid_indices_.begin(),
                                                        samples_to_nearest_centroid_indices_.end(),
                                                        centroids.size() / n_features)}
  , cluster_position_sums_{kmeans::utils::compute_cluster_positions_sum(samples_first,
                                                                        samples_last,
                                                                        samples_to_nearest_centroid_indices_.begin(),
                                                                        centroids.size() / n_features,
                                                                        n_features)}
  , centroid_velocities_{std::vector<DataType>(centroids.size() / n_features)} {}

template <typename SamplesIterator>
Hamerly<SamplesIterator>::Buffers::Buffers(const DatasetDescriptorType& dataset_descriptor,
                                           const std::vector<DataType>& centroids)
  : Hamerly<SamplesIterator>::Buffers::Buffers(std::get<0>(dataset_descriptor),
                                               std::get<1>(dataset_descriptor),
                                               std::get<2>(dataset_descriptor),
                                               centroids) {}

}  // namespace ffcl