#pragma once

#include "BaseIndexer.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>  // for pcl::search::KdTree

namespace indexer {

template <typename IndexContainer, typename SamplesIterator>
class PCLIndexer : public BaseIndexer<IndexContainer, SamplesIterator> {
  public:
    using IndexType = typename BaseIndexer<IndexContainer, SamplesIterator>::IndexType;
    using DataType  = typename BaseIndexer<IndexContainer, SamplesIterator>::DataType;

    using IndicesIteratorType = typename BaseIndexer<IndexContainer, SamplesIterator>::IndicesIteratorType;
    using SamplesIteratorType = typename BaseIndexer<IndexContainer, SamplesIterator>::SamplesIteratorType;

    using BaseNearestNeighborsBuffer =
        typename BaseIndexer<IndexContainer, SamplesIterator>::BaseNearestNeighborsBuffer;

    PCLIndexer(SamplesIterator data_first, SamplesIterator data_last, std::size_t n_features)
      : BaseIndexer<IndexContainer, SamplesIterator>(data_first, data_last, n_features)
      , cloud_{new pcl::PointCloud<pcl::PointXYZ>}
      , kd_tree_{pcl::search::KdTree<pcl::PointXYZ, pcl::KdTreeFLANN<pcl::PointXYZ>>(false)} {
        cloud_->resize(this->n_samples_);

        for (std::size_t sample_index = 0; sample_index < this->n_samples_; ++sample_index) {
            // Each point represents one row of the 2D matrix (n_features-dimensional point)
            cloud_->points[sample_index].x = this->data_first_[sample_index * this->n_features_];
            cloud_->points[sample_index].y = this->data_first_[sample_index * this->n_features_ + 1];
            cloud_->points[sample_index].z = this->data_first_[sample_index * this->n_features_ + 2];
        }
        kd_tree_.setInputCloud(cloud_);
    }

    std::size_t n_samples() const override {
        return this->n_samples_;
    }

    std::size_t n_features() const override {
        return this->n_features_;
    }

    BaseNearestNeighborsBuffer radius_search(std::size_t sample_index_query, const DataType& radius) const override {
        IndexContainer        indices;
        std::vector<DataType> distances_squared;

        kd_tree_.radiusSearch(cloud_->points[sample_index_query], radius, indices, distances_squared);

        return BaseNearestNeighborsBuffer(std::move(indices), std::move(distances_squared));
    }

    BaseNearestNeighborsBuffer nearestKSearch(std::size_t sample_index_query,
                                              std::size_t k_nearest_neighbors) const override {
        IndexContainer        indices;
        std::vector<DataType> distances_squared;

        kd_tree_.nearestKSearch(sample_index_query, k_nearest_neighbors, indices, distances_squared);

        return BaseNearestNeighborsBuffer(std::move(indices), std::move(distances_squared));
    }

  private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
    pcl::search::KdTree<pcl::PointXYZ>  kd_tree_;
};

}  // namespace indexer
