#pragma once

#include "IO.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/kdtree/KDTreeIndexed.hpp"
#include "ffcl/math/random/Sampling.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/flann_search.h>  // for pcl::search::FlannSearch
#include <pcl/search/kdtree.h>        // for pcl::search::KdTree
#include <pcl/search/impl/flann_search.hpp>

#include <flann/flann.hpp>

namespace kdtree::benchmark {

constexpr std::size_t n_neighbors = 5;
constexpr dType       radius      = 0.5;

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

DurationsSummary radius_search_around_query_index_varied_bench(const fs::path& filepath) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(filepath, ' ');
    const std::size_t n_features = get_num_features_in_file(filepath);
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
        // data.begin() + indices[sample_index_query] * n_features,
        // data.begin() + indices[sample_index_query] * n_features + n_features,
        // radius );

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.move_indices().size();

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;
    }

    bench_summary.total_duration = timer.elapsed();
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace ffcl_

namespace pcl_ {

DurationsSummary radius_search_around_query_index_varied_bench(const fs::path& filepath) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(filepath, ' ');
    const std::size_t n_features = get_num_features_in_file(filepath);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    DurationsSummary bench_summary;

    bench_summary.n_samples              = n_samples;
    bench_summary.n_features             = n_features;
    bench_summary.indexer_build_duration = 0;
    bench_summary.indexer_query_duration = 0;
    bench_summary.total_duration         = 0;

    pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>);
    cloud->resize(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        cloud->points[sample_index].x = data[sample_index * n_features];
        cloud->points[sample_index].y = data[sample_index * n_features + 1];
    }
    timer.reset();

    pcl::search::FlannSearch<pcl::PointXY> kd_tree(
        new pcl::search::FlannSearch<pcl::PointXY>::KdTreeIndexCreator(/*max_leaf_size=*/std::sqrt(n_samples)));
    kd_tree.setInputCloud(cloud);

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        pcl::Indices       indices;
        std::vector<float> distances;

        pcl::PointXY searchPoint;
        searchPoint.x = data[sample_index_query * n_features];
        searchPoint.y = data[sample_index_query * n_features + 1];

        // Perform radius search for each point in the cloud
        kd_tree.radiusSearch(searchPoint, radius, indices, distances);

        nn_histogram[sample_index_query] = indices.size();

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;
    }

    bench_summary.total_duration = timer.elapsed();
    // std::cout << "pcl: radius_search_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace pcl_

namespace flann_ {

DurationsSummary radius_search_around_query_index_varied_bench(const fs::path& filepath) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    auto              data       = load_data<dType>(filepath, ' ');
    const std::size_t n_features = get_num_features_in_file(filepath);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    DurationsSummary bench_summary;

    bench_summary.n_samples              = n_samples;
    bench_summary.n_features             = n_features;
    bench_summary.indexer_build_duration = 0;
    bench_summary.indexer_query_duration = 0;
    bench_summary.total_duration         = 0;

    timer.reset();

    flann::Matrix<dType> dataset(data.data(), n_samples, n_features);
    // build 1 kdtree
    flann::Index<flann::L2<dType>> index(
        dataset, flann::KDTreeSingleIndexParams(/*leaf_max_size=*/std::sqrt(n_samples), /*reorder=*/false));
    index.buildIndex();

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        // flann::Matrix<std::size_t> indices;
        // flann::Matrix<dType>       distances;

        std::vector<std::vector<std::size_t>> indices;
        std::vector<std::vector<dType>>       distances;

        flann::Matrix<dType> query(&data[sample_index_query * n_features], 1, n_features);

        // Perform radius search for each point in the cloud
        index.radiusSearch(query, indices, distances, radius, flann::SearchParams{});

        nn_histogram[sample_index_query] = indices[0].size();

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;
    }

    bench_summary.total_duration = timer.elapsed();
    // std::cout << "flann: radius_search_around_query_index (histogram)\n";
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace flann_

void run_benchmarks(const fs::path& filepath) {
    printf("\n\t--- %s ---\n", filepath.filename().c_str());

    auto ffcl_durations_summary  = ffcl_::radius_search_around_query_index_varied_bench(filepath);
    auto pcl_durations_summary   = pcl_::radius_search_around_query_index_varied_bench(filepath);
    auto flann_durations_summary = flann_::radius_search_around_query_index_varied_bench(filepath);

    double to_seconds = 1e-9;  // Conversion factor for nanoseconds to seconds

    // Duration Summary 1 (Original Time)
    double ffcl_build = ffcl_durations_summary.indexer_build_duration * to_seconds;
    double ffcl_query = ffcl_durations_summary.indexer_query_duration * to_seconds;
    double ffcl_total = ffcl_durations_summary.total_duration * to_seconds;

    // Duration Summary 2 (Improved Time)
    double pcl_build = pcl_durations_summary.indexer_build_duration * to_seconds;
    double pcl_query = pcl_durations_summary.indexer_query_duration * to_seconds;
    double pcl_total = pcl_durations_summary.total_duration * to_seconds;

    // Duration Summary 3 (Improved Time)
    double flann_build = flann_durations_summary.indexer_build_duration * to_seconds;
    double flann_query = flann_durations_summary.indexer_query_duration * to_seconds;
    double flann_total = flann_durations_summary.total_duration * to_seconds;

    printf("KDTree (FFCL) speed over %ld queries\n\tbuild: %.6f | queries: %.6f | total: %.6f\n",
           ffcl_durations_summary.n_samples,
           ffcl_build,
           ffcl_query,
           ffcl_total);

    printf("KDTree (PCL) speed over %ld queries\n\tbuild: %.6f | queries: %.6f | total: %.6f\n",
           pcl_durations_summary.n_samples,
           pcl_build,
           pcl_query,
           pcl_total);

    printf("KDTree (FLANN) speed over %ld queries\n\tbuild: %.6f | queries: %.6f | total: %.6f\n",
           flann_durations_summary.n_samples,
           flann_build,
           flann_query,
           flann_total);

    // Calculate the speedup as a percentage for each duration
    auto ffcl_pcl_build_speedup = (pcl_build - ffcl_build) / pcl_build * 100;
    auto ffcl_pcl_query_speedup = (pcl_query - ffcl_query) / pcl_query * 100;
    auto ffcl_pcl_total_speedup = (pcl_total - ffcl_total) / pcl_total * 100;

    printf("KDTree (FFCL speedup over PCL)\n\tbuild: %.3f | queries: %.3f | total: %.3f\n",
           ffcl_pcl_build_speedup,
           ffcl_pcl_query_speedup,
           ffcl_pcl_total_speedup);

    // Calculate the speedup as a percentage for each duration
    auto ffcl_flann_build_speedup = (flann_build - ffcl_build) / flann_build * 100;
    auto ffcl_flann_query_speedup = (flann_query - ffcl_query) / flann_query * 100;
    auto ffcl_flann_total_speedup = (flann_total - ffcl_total) / flann_total * 100;

    printf("KDTree (FFCL speedup over FLANN)\n\tbuild: %.3f | queries: %.3f | total: %.3f\n",
           ffcl_flann_build_speedup,
           ffcl_flann_query_speedup,
           ffcl_flann_total_speedup);
}

void run_toy_datasets_benchmarks() {
    const auto filenames = std::vector<fs::path>{/**/ "noisy_circles.txt",
                                                 /**/ "noisy_moons.txt",
                                                 /**/ "varied.txt",
                                                 /**/ "aniso.txt",
                                                 /**/ "blobs.txt",
                                                 /**/ "no_structure.txt",
                                                 /**/ "unbalanced_blobs.txt"};
    for (const auto& filename : filenames) {
        run_benchmarks(inputs_folder / filename);
    }
}

void run_pointclouds_benchmarks() {
    // the path to the files from the inputs_folder
    const auto relative_path = fs::path("pointclouds_sequences/1");
    const auto filenames     = get_files_names_at_path(inputs_folder / relative_path);

    for (const auto& filename : filenames) {
        run_benchmarks(inputs_folder / relative_path / filename);
    }
}

}  // namespace kdtree::benchmark