#pragma once

#include "cpp_clustering/common/Utils.hpp"
#include "cpp_clustering/heuristics/Heuristics.hpp"
#include "cpp_clustering/kmeans/KMeansPlusPlus.hpp"
#include "cpp_clustering/kmeans/KMeansUtils.hpp"
#include "cpp_clustering/math/random/Distributions.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace cpp_clustering {

template <typename Iterator>
class Lloyd {
    static_assert(std::is_floating_point_v<typename Iterator::value_type> ||
                      std::is_signed_v<typename Iterator::value_type>,
                  "Lloyd allows floating point types or signed interger point types.");

  public:
    using DataType = typename Iterator::value_type;

    // {samples_first_, samples_last_, n_features_}
    using DatasetDescriptorType = std::tuple<Iterator, Iterator, std::size_t>;

    Lloyd(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids);

    Lloyd(const DatasetDescriptorType& dataset_descriptor,
          const std::vector<DataType>& centroids,
          const DataType&              loss);

    Lloyd(const Lloyd&) = delete;

    DataType total_deviation();

    std::vector<DataType> step();

  private:
    struct Buffers {
        Buffers(const Iterator&              samples_first,
                const Iterator&              samples_last,
                std::size_t                  n_features,
                const std::vector<DataType>& centroids);

        Buffers(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids);

        Buffers(const Buffers&) = delete;

        std::vector<std::size_t> samples_to_nearest_centroid_indices_;
        std::vector<DataType>    samples_to_nearest_centroid_distances_;
    };

    DataType update_buffers();

    DatasetDescriptorType    dataset_descriptor_;
    std::size_t              n_samples_;
    std::vector<DataType>    centroids_;
    std::unique_ptr<Buffers> buffers_ptr_;
    DataType                 loss_;
};

template <typename Iterator>
Lloyd<Iterator>::Lloyd(const DatasetDescriptorType& dataset_descriptor, const std::vector<DataType>& centroids)
  : Lloyd<Iterator>::Lloyd(dataset_descriptor, centroids, common::utils::infinity<DataType>()) {
    // compute initial loss
    loss_ = std::reduce(buffers_ptr_->samples_to_nearest_centroid_distances_.begin(),
                        buffers_ptr_->samples_to_nearest_centroid_distances_.end(),
                        static_cast<typename Lloyd<Iterator>::DataType>(0),
                        std::plus<>());
}

template <typename Iterator>
Lloyd<Iterator>::Lloyd(const DatasetDescriptorType& dataset_descriptor,
                       const std::vector<DataType>& centroids,
                       const DataType&              loss)
  : dataset_descriptor_{dataset_descriptor}
  , n_samples_{common::utils::get_n_samples(std::get<0>(dataset_descriptor_),
                                            std::get<1>(dataset_descriptor_),
                                            std::get<2>(dataset_descriptor_))}
  , centroids_{centroids}
  , buffers_ptr_{std::make_unique<Buffers>(dataset_descriptor, centroids_)}
  , loss_{loss} {}

template <typename Iterator>
typename Lloyd<Iterator>::DataType Lloyd<Iterator>::total_deviation() {
    return loss_;
}

template <typename Iterator>
std::vector<typename Lloyd<Iterator>::DataType> Lloyd<Iterator>::step() {
    const auto& nearest_centroid_indices = buffers_ptr_->samples_to_nearest_centroid_indices_;

    const auto [samples_first, samples_last, n_features] = dataset_descriptor_;
    const std::size_t n_centroids                        = centroids_.size() / n_features;
    // the number of samples associated to each centroids
    auto cluster_sizes = std::vector<std::size_t>(n_centroids);
    // accumulate the positions of each sample in each cluster
    auto cluster_positions_sum = std::vector<typename Lloyd<Iterator>::DataType>(n_centroids * n_features);

    // accumulate the positions the samples w.r.t. their clusters and count the clusters sizes
    for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
        const auto centroid_index = nearest_centroid_indices[sample_index];

        ++cluster_sizes[centroid_index];

        std::transform(cluster_positions_sum.begin() + centroid_index * n_features,
                       cluster_positions_sum.begin() + centroid_index * n_features + n_features,
                       samples_first + sample_index * n_features,
                       cluster_positions_sum.begin() + centroid_index * n_features,
                       std::plus<>());
    }
    // Update the centroids using the assigned samples
    for (std::size_t centroid_index = 0; centroid_index < n_centroids; ++centroid_index) {
        const auto feature_index_start = centroid_index * n_features;
        const auto feature_index_end   = feature_index_start + n_features;

        if (cluster_sizes[centroid_index] == 0) {
            // For centroids with only 1 associated sample, the new position is the same as the previous one
            std::copy(cluster_positions_sum.begin() + feature_index_start,
                      cluster_positions_sum.begin() + feature_index_end,
                      centroids_.begin() + feature_index_start);
        } else {
            // Compute the new centroid position for the centroid that has more than 1 associated sample
            std::transform(cluster_positions_sum.begin() + feature_index_start,
                           cluster_positions_sum.begin() + feature_index_end,
                           centroids_.begin() + feature_index_start,
                           [cluster_size = cluster_sizes[centroid_index]](const auto& sum) {
                               return sum / static_cast<typename Lloyd<Iterator>::DataType>(cluster_size);
                           });
        }
    }
    // recompute the loss w.r.t. the updated buffers
    loss_ = update_buffers();
    return centroids_;
}

template <typename Iterator>
typename Iterator::value_type Lloyd<Iterator>::update_buffers() {
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
                       static_cast<typename Lloyd<Iterator>::DataType>(0),
                       std::plus<>());
}

template <typename Iterator>
Lloyd<Iterator>::Buffers::Buffers(const Iterator&                                        samples_first,
                                  const Iterator&                                        samples_last,
                                  std::size_t                                            n_features,
                                  const std::vector<typename Lloyd<Iterator>::DataType>& centroids)
  : samples_to_nearest_centroid_indices_{kmeans::utils::samples_to_nearest_centroid_indices(samples_first,
                                                                                            samples_last,
                                                                                            n_features,
                                                                                            centroids)}
  , samples_to_nearest_centroid_distances_{
        kmeans::utils::samples_to_nearest_centroid_distances(samples_first, samples_last, n_features, centroids)} {}

template <typename Iterator>
Lloyd<Iterator>::Buffers::Buffers(const DatasetDescriptorType&                           dataset_descriptor,
                                  const std::vector<typename Lloyd<Iterator>::DataType>& centroids)
  : Lloyd<Iterator>::Buffers::Buffers(std::get<0>(dataset_descriptor),
                                      std::get<1>(dataset_descriptor),
                                      std::get<2>(dataset_descriptor),
                                      centroids) {}

}  // namespace cpp_clustering