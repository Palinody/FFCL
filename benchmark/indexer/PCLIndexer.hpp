#pragma once

#include "ffcl/common/Utils.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/flann_search.h>  // for pcl::search::FlannSearch
#include <pcl/search/kdtree.h>        // for pcl::search::KdTree
#include <pcl/search/impl/flann_search.hpp>

namespace pcl_index {

template <typename IndexContainer, typename SamplesIterator>
class BaseIndexer {
  public:
    using DataType = typename SamplesIterator::value_type;

    class NearestNeighborsBuffer {
      public:
        NearestNeighborsBuffer(IndexContainer&& indices, std::vector<DataType>&& distances) noexcept
          : indices_{std::move(indices)}
          , distances_{std::move(distances)} {}

        std::size_t size() const {
            return indices_.size();
        }

        IndexContainer indices() const {
            return indices_;
        }

        IndexContainer move_indices() {
            return std::move(indices_);
        }

      private:
        IndexContainer        indices_;
        std::vector<DataType> distances_;
    };

    virtual ~BaseIndexer() = default;

    virtual std::size_t n_samples() const = 0;

    virtual std::size_t n_features() const = 0;

    virtual NearestNeighborsBuffer radiusSearch(std::size_t sample_index_query, const DataType& radius) const = 0;

    virtual NearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                                  std::size_t k_nearest_neighbors) const = 0;
};

template <typename IndexContainer, typename SamplesIterator>
class PCLFLANNIndexer : public BaseIndexer<IndexContainer, SamplesIterator> {
  public:
    using DataType               = typename BaseIndexer<IndexContainer, SamplesIterator>::DataType;
    using NearestNeighborsBuffer = typename BaseIndexer<IndexContainer, SamplesIterator>::NearestNeighborsBuffer;

    PCLFLANNIndexer(SamplesIterator data_first,
                    SamplesIterator data_last,
                    std::size_t     n_features,
                    std::size_t     max_leaf_size)
      : data_first_{data_first}
      , data_last_{data_last}
      , n_samples_{common::utils::get_n_samples(data_first, data_last, n_features)}
      , n_features_{n_features}
      , max_leaf_size_{max_leaf_size}
      , cloud_{new pcl::PointCloud<pcl::PointXYZ>}
      , kd_tree_{new pcl::search::FlannSearch<pcl::PointXYZ>::KdTreeIndexCreator(/*max_leaf_size=*/max_leaf_size_)} {
        cloud_->resize(n_samples_);

        for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
            // Each point represents one row of the 2D matrix (n_features-dimensional point)
            cloud_->points[sample_index].x = data_first_[sample_index * n_features_];
            cloud_->points[sample_index].y = data_first_[sample_index * n_features_ + 1];
            cloud_->points[sample_index].z = data_first_[sample_index * n_features_ + 2];
        }

        kd_tree_.setInputCloud(cloud_);
    }

    std::size_t n_samples() const override {
        return n_samples_;
    }

    std::size_t n_features() const override {
        return n_features_;
    }

    NearestNeighborsBuffer radiusSearch(std::size_t sample_index_query, const DataType& radius) const override {
        IndexContainer        indices;
        std::vector<DataType> distances_squared;

        kd_tree_.radiusSearch(cloud_->points[sample_index_query], radius, indices, distances_squared);

        return NearestNeighborsBuffer(std::move(indices), std::move(distances_squared));
    }

    NearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                          std::size_t k_nearest_neighbors) const override {
        IndexContainer        indices;
        std::vector<DataType> distances_squared;

        kd_tree_.nearestKSearch(cloud_->points[sample_index_query], k_nearest_neighbors, indices, distances_squared);

        return NearestNeighborsBuffer(std::move(indices), std::move(distances_squared));
    }

  private:
    SamplesIterator data_first_, data_last_;
    std::size_t     n_samples_, n_features_;
    std::size_t     max_leaf_size_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr     cloud_;
    pcl::search::FlannSearch<pcl::PointXYZ> kd_tree_;
};

template <typename IndexContainer, typename SamplesIterator>
class PCLIndexer : public BaseIndexer<IndexContainer, SamplesIterator> {
  public:
    using DataType               = typename BaseIndexer<IndexContainer, SamplesIterator>::DataType;
    using NearestNeighborsBuffer = typename BaseIndexer<IndexContainer, SamplesIterator>::NearestNeighborsBuffer;

    PCLIndexer(SamplesIterator data_first, SamplesIterator data_last, std::size_t n_features)
      : data_first_{data_first}
      , data_last_{data_last}
      , n_samples_{common::utils::get_n_samples(data_first, data_last, n_features)}
      , n_features_{n_features}
      , cloud_{new pcl::PointCloud<pcl::PointXYZ>}
      , kd_tree_{pcl::search::KdTree<pcl::PointXYZ, pcl::KdTreeFLANN<pcl::PointXYZ>>(false)} {
        cloud_->resize(n_samples_);

        for (std::size_t sample_index = 0; sample_index < n_samples_; ++sample_index) {
            // Each point represents one row of the 2D matrix (n_features-dimensional point)
            cloud_->points[sample_index].x = data_first_[sample_index * n_features_];
            cloud_->points[sample_index].y = data_first_[sample_index * n_features_ + 1];
            cloud_->points[sample_index].z = data_first_[sample_index * n_features_ + 2];
        }
        kd_tree_.setInputCloud(cloud_);
    }

    std::size_t n_samples() const override {
        return n_samples_;
    }

    std::size_t n_features() const override {
        return n_features_;
    }

    NearestNeighborsBuffer radiusSearch(std::size_t sample_index_query, const DataType& radius) const override {
        IndexContainer        indices;
        std::vector<DataType> distances_squared;

        kd_tree_.radiusSearch(cloud_->points[sample_index_query], radius, indices, distances_squared);

        return NearestNeighborsBuffer(std::move(indices), std::move(distances_squared));
    }

    NearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                          std::size_t k_nearest_neighbors) const override {
        IndexContainer        indices;
        std::vector<DataType> distances_squared;

        kd_tree_.nearestKSearch(sample_index_query, k_nearest_neighbors, indices, distances_squared);

        return NearestNeighborsBuffer(std::move(indices), std::move(distances_squared));
    }

  private:
    SamplesIterator data_first_, data_last_;
    std::size_t     n_samples_, n_features_;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::search::KdTree<pcl::PointXYZ>  kd_tree_;
};

}  // namespace pcl_index