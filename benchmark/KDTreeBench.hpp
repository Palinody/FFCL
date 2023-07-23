#pragma once

#include "IO.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"
#include "ffcl/math/random/Sampling.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <flann/flann.hpp>

namespace kdtree::benchmark {

constexpr std::size_t n_neighbors = 5;
constexpr dType       radius      = 1;

struct DurationsSummary {
    ssize_t       n_samples              = -1;
    ssize_t       n_features             = -1;
    std::uint64_t indexer_build_duration = 0;
    std::uint64_t indexer_query_duration = 0;
    std::uint64_t total_duration         = 0;
};

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

template <typename Type>
void print_data(const std::vector<Type>& data, std::size_t n_features) {
    if (!n_features) {
        return;
    }
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

namespace ffcl_ {

DurationsSummary radius_search_around_query_index_varied_bench(const fs::path& filename = "varied.txt") {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    DurationsSummary bench_summary;

    bench_summary.n_samples              = n_samples;
    bench_summary.n_features             = n_features;
    bench_summary.indexer_build_duration = 0;
    bench_summary.indexer_query_duration = 0;
    bench_summary.total_duration         = 0;

    timer.reset();

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::containers::KDTreeIndexed<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::IndexedHighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::IndexedQuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // IndexedHighestVarianceBuild, IndexedMaximumSpreadBuild, IndexedCycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        auto nearest_neighbors_buffer = indexer.radius_search_around_query_index(indices[sample_index_query], radius);

        // auto nearest_neighbors_buffer = indexer.radius_search_around_query_sample(
        //     data.begin() + indices[sample_index_query] * n_features,
        //     data.begin() + indices[sample_index_query] * n_features + n_features,
        //     radius);

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.size();
    }

    bench_summary.total_duration = timer.elapsed();
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace ffcl_

namespace pcl_ {

DurationsSummary radius_search_around_query_index_varied_bench(const fs::path& filename = "varied.txt") {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    DurationsSummary bench_summary;

    bench_summary.n_samples              = n_samples;
    bench_summary.n_features             = n_features;
    bench_summary.indexer_build_duration = 0;
    bench_summary.indexer_query_duration = 0;
    bench_summary.total_duration         = 0;

    timer.reset();

    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    cloud->resize(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        cloud->points[sample_index].x = data[sample_index * n_features];
        cloud->points[sample_index].y = data[sample_index * n_features + 1];
    }

    pcl::search::KdTree<pcl::PointXY>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXY>);
    kd_tree->setInputCloud(cloud);

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        std::vector<int>   nearest_neighbors_buffer;
        std::vector<float> distances_buffer;

        // Perform radius search for each point in the cloud
        kd_tree->radiusSearch(sample_index_query, radius, nearest_neighbors_buffer, distances_buffer);

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;

        nn_histogram[sample_index_query] = nearest_neighbors_buffer.size();
    }

    bench_summary.total_duration = timer.elapsed();
    // std::cout << "radius_search_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace pcl_

namespace flann_ {

DurationsSummary radius_search_around_query_index_varied_bench(const fs::path& filename = "varied.txt") {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    DurationsSummary bench_summary;

    bench_summary.n_samples              = n_samples;
    bench_summary.n_features             = n_features;
    bench_summary.indexer_build_duration = 0;
    bench_summary.indexer_query_duration = 0;
    bench_summary.total_duration         = 0;

    timer.reset();

    flann::Matrix<float> dataset(data.data(), n_samples, n_features);
    // build 1 kdtree
    flann::Index<flann::L2<float>> index(dataset, flann::KDTreeSingleIndexParams(std::sqrt(n_samples)));
    index.buildIndex();

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    std::vector<std::vector<int>>   indices;
    std::vector<std::vector<float>> distances;

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        flann::Matrix<float> query(&data[sample_index_query * n_features], 1, n_features);

        // Perform radius search for each point in the cloud
        index.radiusSearch(query, indices, distances, radius, flann::SearchParams{});

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;

        nn_histogram[sample_index_query] = indices[0].size();
    }

    bench_summary.total_duration = timer.elapsed();
    // std::cout << "radius_search_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace flann_

void run_benchmarks() {
    const auto filenames_list = std::array<fs::path, 7>{/**/ "noisy_circles",
                                                        /**/ "noisy_moons",
                                                        /**/ "varied",
                                                        /**/ "aniso",
                                                        /**/ "blobs",
                                                        /**/ "no_structure",
                                                        /**/ "unbalanced_blobs"};

    for (const auto& filename : filenames_list) {
        common::utils::ignore_parameters(filename);
    }
}

}  // namespace kdtree::benchmark