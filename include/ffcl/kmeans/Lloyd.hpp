#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/kmeans/KMeansUtils.hpp"
#include "ffcl/math/heuristics/Distances.hpp"

#include <tuple>
#include <vector>

namespace ffcl {

template <typename SamplesIterator>
class Lloyd {
    static_assert(std::is_floating_point_v<typename SamplesIterator::value_type>, "Lloyd allows floating point types.");

  public:
    using DataType = typename SamplesIterator::value_type;

    // pointers/iterators to the first and last elements of the dataset and the feature size
    using DatasetDescriptorType = std::tuple<SamplesIterator, SamplesIterator, std::size_t>;

    Lloyd(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids);

    Lloyd(const DatasetDescriptorType& dataset_descriptor,
          const std::vector<DataType>& centroids,
          const DataType&              loss);

    Lloyd(const Lloyd&) = delete;

    DataType total_deviation() const;

    std::vector<DataType> step();

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

        std::vector<std::size_t> cluster_sizes_;
        std::vector<DataType>    cluster_position_sums_;
    };

    void update_clusters();

    void update_centroids();

    DataType update_buffers();

    DatasetDescriptorType    dataset_descriptor_;
    std::size_t              n_samples_;
    std::vector<DataType>    centroids_;
    std::unique_ptr<Buffers> buffers_ptr_;
    DataType                 loss_;
};

template <typename SamplesIterator>
Lloyd<SamplesIterator>::Lloyd(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids)
  : Lloyd<SamplesIterator>::Lloyd(dataset_descriptor, centroids, common::utils::infinity<DataType>()) {
    // compute initial loss
    loss_ = std::reduce(buffers_ptr_->samples_to_nearest_centroid_distances_.begin(),
                        buffers_ptr_->samples_to_nearest_centroid_distances_.end(),
                        static_cast<typename Lloyd<SamplesIterator>::DataType>(0),
                        std::plus<>());
}

template <typename SamplesIterator>
Lloyd<SamplesIterator>::Lloyd(const DatasetDescriptorType& dataset_descriptor,
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
typename Lloyd<SamplesIterator>::DataType Lloyd<SamplesIterator>::total_deviation() const {
    return loss_;
}

template <typename SamplesIterator>
std::vector<typename Lloyd<SamplesIterator>::DataType> Lloyd<SamplesIterator>::step() {
    // update the cluster sizes and intra-cluster sum of positions
    update_clusters();
    // update all the centroids with the new intra-cluster positions sum and cluster sizes
    update_centroids();
    // recompute the loss w.r.t. the updated buffers
    loss_ = update_buffers();
    return centroids_;
}

template <typename SamplesIterator>
void Lloyd<SamplesIterator>::update_centroids() {
    const auto        n_features  = std::get<2>(dataset_descriptor_);
    const std::size_t n_centroids = centroids_.size() / n_features;

    auto& cluster_sizes         = buffers_ptr_->cluster_sizes_;
    auto& cluster_position_sums = buffers_ptr_->cluster_position_sums_;

    // Update the centroids using the assigned samples
    for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
        const auto feature_index_start = centroid_index * n_features;
        const auto feature_index_end   = feature_index_start + n_features;

        if (cluster_sizes[centroid_index] == 0) {
            // For centroids with only 1 associated sample, the new position is the same as the previous one
            std::copy(cluster_position_sums.begin() + feature_index_start,
                      cluster_position_sums.begin() + feature_index_end,
                      centroids_.begin() + feature_index_start);
        } else {
            // Compute the new centroid position for the centroid that has more than 1 associated sample
            std::transform(cluster_position_sums.begin() + feature_index_start,
                           cluster_position_sums.begin() + feature_index_end,
                           centroids_.begin() + feature_index_start,
                           [cluster_size = cluster_sizes[centroid_index]](const auto& sum) {
                               return sum / static_cast<typename Lloyd<SamplesIterator>::DataType>(cluster_size);
                           });
        }
    }
}

template <typename SamplesIterator>
typename SamplesIterator::value_type Lloyd<SamplesIterator>::update_buffers() {
    buffers_ptr_->samples_to_nearest_centroid_indices_ =
        kmeans::utils::samples_to_nearest_centroid_indices(std::get<0>(dataset_descriptor_),
                                                           std::get<1>(dataset_descriptor_),
                                                           std::get<2>(dataset_descriptor_),
                                                           centroids_);

    buffers_ptr_->samples_to_nearest_centroid_distances_ =
        kmeans::utils::samples_to_nearest_centroid_distances(std::get<0>(dataset_descriptor_),
                                                             std::get<1>(dataset_descriptor_),
                                                             std::get<2>(dataset_descriptor_),
                                                             centroids_);

    return std::reduce(buffers_ptr_->samples_to_nearest_centroid_distances_.begin(),
                       buffers_ptr_->samples_to_nearest_centroid_distances_.end(),
                       static_cast<typename Lloyd<SamplesIterator>::DataType>(0),
                       std::plus<>());
}

template <typename SamplesIterator>
void Lloyd<SamplesIterator>::update_clusters() {
    const auto [samples_first, samples_last, n_features] = dataset_descriptor_;
    const std::size_t n_centroids                        = centroids_.size() / n_features;

    buffers_ptr_->cluster_sizes_ =
        kmeans::utils::compute_cluster_sizes(buffers_ptr_->samples_to_nearest_centroid_indices_.begin(),
                                             buffers_ptr_->samples_to_nearest_centroid_indices_.end(),
                                             n_centroids);
    buffers_ptr_->cluster_position_sums_ =
        kmeans::utils::compute_cluster_positions_sum(samples_first,
                                                     samples_last,
                                                     buffers_ptr_->samples_to_nearest_centroid_indices_.begin(),
                                                     n_centroids,
                                                     n_features);
}

template <typename SamplesIterator>
Lloyd<SamplesIterator>::Buffers::Buffers(const SamplesIterator&                                        samples_first,
                                         const SamplesIterator&                                        samples_last,
                                         std::size_t                                                   n_features,
                                         const std::vector<typename Lloyd<SamplesIterator>::DataType>& centroids)
  : samples_to_nearest_centroid_indices_{kmeans::utils::samples_to_nearest_centroid_indices(samples_first,
                                                                                            samples_last,
                                                                                            n_features,
                                                                                            centroids)}
  , samples_to_nearest_centroid_distances_{kmeans::utils::samples_to_nearest_centroid_distances(samples_first,
                                                                                                samples_last,
                                                                                                n_features,
                                                                                                centroids)}
  , cluster_sizes_{std::vector<std::size_t>(centroids.size() / n_features)}
  , cluster_position_sums_{std::vector<typename Lloyd<SamplesIterator>::DataType>(centroids.size())} {}

template <typename SamplesIterator>
Lloyd<SamplesIterator>::Buffers::Buffers(const DatasetDescriptorType& dataset_descriptor,
                                         const std::vector<typename Lloyd<SamplesIterator>::DataType>& centroids)
  : Lloyd<SamplesIterator>::Buffers::Buffers(std::get<0>(dataset_descriptor),
                                             std::get<1>(dataset_descriptor),
                                             std::get<2>(dataset_descriptor),
                                             centroids) {}

}  // namespace ffcl