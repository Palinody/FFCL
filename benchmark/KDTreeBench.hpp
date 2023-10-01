#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/math/random/Sampling.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/flann_search.h>  // for pcl::search::FlannSearch
#include <pcl/search/kdtree.h>        // for pcl::search::KdTree
#include <pcl/search/impl/flann_search.hpp>

#include <flann/flann.hpp>

namespace kdtree::benchmark {

namespace ffcl_ {

utils::DurationsSummary radius_search_around_query_index_bench(const fs::path& filepath, float radius) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    n_features = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    timer.reset();

    auto indices = utils::generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType().feature_mask({0, 1, 2}))
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

utils::DurationsSummary k_nearest_neighbors_search_around_query_index_bench(const fs::path& filepath,
                                                                            std::size_t     k_nearest_neighbors) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    n_features = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    timer.reset();

    auto indices = utils::generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType().feature_mask({0, 1, 2}))
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        auto nearest_neighbors_buffer =
            indexer.k_nearest_neighbors_around_query_index(indices[sample_index_query], k_nearest_neighbors);

        // auto nearest_neighbors_buffer = indexer.k_nearest_neighbors_around_query_sample(
        // data.begin() + indices[sample_index_query] * n_features,
        // data.begin() + indices[sample_index_query] * n_features + n_features,
        // k_nearest_neighbors );

        nn_histogram[indices[sample_index_query]] = nearest_neighbors_buffer.move_indices().size();

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;
    }

    bench_summary.total_duration = timer.elapsed();
    // print_data(nn_histogram, n_samples);

    return bench_summary;
}

}  // namespace ffcl_

namespace pcl_ {

utils::DurationsSummary radius_search_around_query_index_bench(const fs::path& filepath, float radius) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    n_features = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        cloud->points[sample_index].x = data[sample_index * n_features];
        cloud->points[sample_index].y = data[sample_index * n_features + 1];
        cloud->points[sample_index].z = data[sample_index * n_features + 2];
    }
    timer.reset();

    pcl::search::FlannSearch<pcl::PointXYZ> kd_tree(
        new pcl::search::FlannSearch<pcl::PointXYZ>::KdTreeIndexCreator(/*max_leaf_size=*/std::sqrt(n_samples)));
    kd_tree.setInputCloud(cloud);

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        pcl::Indices       indices;
        std::vector<float> distances;

        pcl::PointXYZ searchPoint;
        searchPoint.x = data[sample_index_query * n_features];
        searchPoint.y = data[sample_index_query * n_features + 1];
        searchPoint.z = data[sample_index_query * n_features + 2];

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

utils::DurationsSummary k_nearest_neighbors_search_around_query_index_bench(const fs::path& filepath,
                                                                            std::size_t     k_nearest_neighbors) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    n_features = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(n_samples);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        cloud->points[sample_index].x = data[sample_index * n_features];
        cloud->points[sample_index].y = data[sample_index * n_features + 1];
        cloud->points[sample_index].z = data[sample_index * n_features + 2];
    }
    timer.reset();

    pcl::search::FlannSearch<pcl::PointXYZ> kd_tree(
        new pcl::search::FlannSearch<pcl::PointXYZ>::KdTreeIndexCreator(/*max_leaf_size=*/std::sqrt(n_samples)));
    kd_tree.setInputCloud(cloud);

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        pcl::Indices       indices;
        std::vector<float> distances;

        pcl::PointXYZ searchPoint;
        searchPoint.x = data[sample_index_query * n_features];
        searchPoint.y = data[sample_index_query * n_features + 1];
        searchPoint.z = data[sample_index_query * n_features + 2];

        // Perform k_nearest_neighbors search for each point in the cloud
        kd_tree.nearestKSearch(searchPoint, k_nearest_neighbors, indices, distances);

        nn_histogram[sample_index_query] = indices.size();

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;
    }

    bench_summary.total_duration = timer.elapsed();

    return bench_summary;
}

}  // namespace pcl_

namespace flann_ {

utils::DurationsSummary radius_search_around_query_index_bench(const fs::path& filepath, float radius) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    n_features = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    auto data_xyz = std::vector<bench::io::DataType>(n_samples * n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        data_xyz[sample_index * n_features]     = data[sample_index * n_features];
        data_xyz[sample_index * n_features + 1] = data[sample_index * n_features + 1];
        data_xyz[sample_index * n_features + 2] = data[sample_index * n_features + 2];
    }

    timer.reset();

    flann::Matrix<bench::io::DataType> dataset(data_xyz.data(), n_samples, n_features);
    // build 1 kdtree
    flann::Index<flann::L2<bench::io::DataType>> index(
        dataset, flann::KDTreeSingleIndexParams(/*leaf_max_size=*/std::sqrt(n_samples), /*reorder=*/true));
    index.buildIndex();

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        std::vector<std::vector<std::size_t>>         indices;
        std::vector<std::vector<bench::io::DataType>> distances;

        flann::Matrix<bench::io::DataType> query(&data_xyz[sample_index_query * n_features], 1, n_features);

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

utils::DurationsSummary k_nearest_neighbors_search_around_query_index_bench(const fs::path& filepath,
                                                                            std::size_t     k_nearest_neighbors) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::vector<bench::io::DataType> data;
    std::size_t                      n_samples, n_features;

    if (filepath.extension().string() == ".bin") {
        std::tie(data, n_samples, n_features) = bench::io::bin::decode(/*n_features=*/4, filepath);

    } else if (filepath.extension().string() == ".txt") {
        data       = bench::io::txt::load_data<bench::io::DataType>(filepath, ' ');
        n_features = bench::io::txt::get_num_features_in_file(filepath);
        n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    } else {
        char message[100];
        std::sprintf(message, "File extension found '%s' but only supports .txt or .bin", filepath.extension().c_str());
        throw std::runtime_error(message);
    }

    utils::DurationsSummary bench_summary;

    n_features = 3;

    bench_summary.n_samples  = n_samples;
    bench_summary.n_features = n_features;

    auto data_xyz = std::vector<bench::io::DataType>(n_samples * n_features);

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        // Each point represents one row of the 2D matrix (n_features-dimensional point)
        data_xyz[sample_index * n_features]     = data[sample_index * n_features];
        data_xyz[sample_index * n_features + 1] = data[sample_index * n_features + 1];
        data_xyz[sample_index * n_features + 2] = data[sample_index * n_features + 2];
    }

    timer.reset();

    flann::Matrix<bench::io::DataType> dataset(data_xyz.data(), n_samples, n_features);
    // build 1 kdtree
    flann::Index<flann::L2<bench::io::DataType>> index(
        dataset, flann::KDTreeSingleIndexParams(/*leaf_max_size=*/std::sqrt(n_samples), /*reorder=*/true));
    index.buildIndex();

    bench_summary.indexer_build_duration = timer.elapsed();

    std::vector<std::size_t> nn_histogram(n_samples);

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        const auto elapsed_start = timer.elapsed();

        std::vector<std::vector<std::size_t>>         indices;
        std::vector<std::vector<bench::io::DataType>> distances;

        flann::Matrix<bench::io::DataType> query(&data_xyz[sample_index_query * n_features], 1, n_features);

        // Perform radius search for each point in the cloud
        index.knnSearch(query, indices, distances, k_nearest_neighbors, flann::SearchParams{});

        nn_histogram[sample_index_query] = indices[0].size();

        bench_summary.indexer_query_duration += timer.elapsed() - elapsed_start;
    }

    bench_summary.total_duration = timer.elapsed();

    return bench_summary;
}

}  // namespace flann_

void run_benchmarks(const fs::path& filepath, float radius) {
    printf("\n\t--- %s ---\n", filepath.filename().c_str());

    auto ffcl_durations_summary  = ffcl_::radius_search_around_query_index_bench(filepath, radius);
    auto pcl_durations_summary   = pcl_::radius_search_around_query_index_bench(filepath, radius);
    auto flann_durations_summary = flann_::radius_search_around_query_index_bench(filepath, radius);

    // Conversion factor for nanoseconds to seconds
    double to_seconds = 1e-9;

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
           static_cast<std::size_t>(ffcl_durations_summary.n_samples),
           ffcl_build,
           ffcl_query,
           ffcl_total);

    printf("KDTree (PCL) speed over %ld queries\n\tbuild: %.6f | queries: %.6f | total: %.6f\n",
           static_cast<std::size_t>(pcl_durations_summary.n_samples),
           pcl_build,
           pcl_query,
           pcl_total);

    printf("KDTree (FLANN) speed over %ld queries\n\tbuild: %.6f | queries: %.6f | total: %.6f\n",
           static_cast<std::size_t>(flann_durations_summary.n_samples),
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

    const bench::io::DataType radius = 0.5;

    for (const auto& filename : filenames) {
        run_benchmarks(bench::io::inputs_folder / filename, radius);
    }
}

void run_pointclouds_benchmarks() {
    // the path to the files from the inputs_folder
    const auto relative_path = fs::path("pointclouds_sequences/1");
    const auto filenames     = bench::io::get_files_names_at_path(bench::io::inputs_folder / relative_path);

    const bench::io::DataType radius = 0.5;

    for (const auto& filename : filenames) {
        run_benchmarks(bench::io::inputs_folder / relative_path / filename, radius);
    }
}

template <typename Function, typename... Args>
void run_pointclouds_sequences_benchmark(const Function&    function,
                                         const fs::path&    relative_path,
                                         const std::string& library_name,
                                         Args&&... args) {
    // the path to the files from the inputs_folder
    // const auto relative_path = fs::path("pointclouds_sequences/2");
    const auto filenames = bench::io::get_files_names_at_path(bench::io::inputs_folder / relative_path);

    // Conversion factor for nanoseconds to seconds
    long double to_seconds = 1e-9;

    // the sequence object that will be used to compute the variance
    std::vector<utils::DurationsSummary> bench_summary_vector;
    bench_summary_vector.reserve(filenames.size());
    // the object that will be used to compute the mean
    utils::DurationsSummary bench_summary_mean;

    for (std::size_t file_index = 0; file_index < filenames.size(); ++file_index) {
        const auto& filename = filenames[file_index];

        auto bench_summary = function(bench::io::inputs_folder / relative_path / filename, std::forward<Args>(args)...);

        bench_summary.apply_timer_multiplier(to_seconds);

        bench_summary_mean += bench_summary;

        bench_summary_vector.emplace_back(std::move(bench_summary));

        utils::print_progress_bar(file_index, filenames.size());
    }
    bench_summary_mean /= filenames.size();

    utils::DurationsSummary bench_summary_variance;

    for (auto& bench_summary : bench_summary_vector) {
        bench_summary -= bench_summary_mean;
        bench_summary *= bench_summary;
        bench_summary_variance += bench_summary;
    }
    bench_summary_variance /= filenames.size();

    printf("%s computation time average ± variance (n_samples/queries: %.2Lf ± %.2Lf | n_features: %ld):"
           "\n\tbuild: %.12Lf ± %.12Lf"
           "\n\tqueries: %.12Lf ± %.12Lf"
           "\n\ttotal: %.12Lf ± %.12Lf\n",
           library_name.c_str(),
           bench_summary_mean.n_samples,
           bench_summary_variance.n_samples,
           static_cast<std::size_t>(bench_summary_mean.n_features),
           bench_summary_mean.indexer_build_duration,
           bench_summary_variance.indexer_build_duration,
           bench_summary_mean.indexer_query_duration,
           bench_summary_variance.indexer_query_duration,
           bench_summary_mean.total_duration,
           bench_summary_variance.total_duration);
}

void run_radius_search_benchmarks_on_point_cloud_sequences() {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    const std::vector<bench::io::DataType> radiuses = {0.1, 0.3, 0.5, 1.0};

    const std::vector<fs::path> relative_paths = {fs::path("pointclouds_sequences/1"),
                                                  fs::path("pointclouds_sequences/2"),
                                                  fs::path("pointclouds_sequences/0000"),
                                                  fs::path("pointclouds_sequences/0001")};

    for (const auto& radius : radiuses) {
        std::cout << "Running benchmarks with radius: " << radius << "\n";
        for (const auto& relative_path : relative_paths) {
            std::cout << "---\nSequence in folder: " << relative_path.c_str() << "\n\n";
            timer.reset();
            run_pointclouds_sequences_benchmark(
                &ffcl_::radius_search_around_query_index_bench, relative_path, "FFCL", radius);
            timer.print_elapsed_seconds();

            std::cout << "---\n";

            timer.reset();
            run_pointclouds_sequences_benchmark(
                &flann_::radius_search_around_query_index_bench, relative_path, "FLANN", radius);
            timer.print_elapsed_seconds();

            std::cout << "---\n";

            timer.reset();
            run_pointclouds_sequences_benchmark(
                &pcl_::radius_search_around_query_index_bench, relative_path, "PCL", radius);
            timer.print_elapsed_seconds();

            std::cout << "\n---\n";
        }
    }
}

void run_k_nearest_neighbors_search_benchmarks_on_point_cloud_sequences() {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    const std::vector<std::size_t> n_neighbors_choices = {10, 5, 3};

    const std::vector<fs::path> relative_paths = {fs::path("pointclouds_sequences/1"),
                                                  fs::path("pointclouds_sequences/2"),
                                                  fs::path("pointclouds_sequences/0000"),
                                                  fs::path("pointclouds_sequences/0001")};

    for (const auto& n_neighbors : n_neighbors_choices) {
        std::cout << "Running benchmarks with n_neighbors: " << n_neighbors << "\n";
        for (const auto& relative_path : relative_paths) {
            std::cout << "---\nSequence in folder: " << relative_path.c_str() << "\n\n";
            timer.reset();
            run_pointclouds_sequences_benchmark(
                &ffcl_::k_nearest_neighbors_search_around_query_index_bench, relative_path, "FFCL", n_neighbors);
            timer.print_elapsed_seconds();

            std::cout << "---\n";

            timer.reset();
            // n_neighbors + 1 because flann also returns the query
            run_pointclouds_sequences_benchmark(
                &flann_::k_nearest_neighbors_search_around_query_index_bench, relative_path, "FLANN", n_neighbors + 1);
            timer.print_elapsed_seconds();

            std::cout << "---\n";

            timer.reset();
            // n_neighbors + 1 because pcl also returns the query
            run_pointclouds_sequences_benchmark(
                &pcl_::k_nearest_neighbors_search_around_query_index_bench, relative_path, "PCL", n_neighbors + 1);
            timer.print_elapsed_seconds();

            std::cout << "\n---\n";
        }
    }
}

}  // namespace kdtree::benchmark