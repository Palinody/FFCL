#include "DBSCANBench.hpp"
#include "KDTreeBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

#include <stdio.h>

int main() {
    // kmeans::benchmark::mnist_bench();
    // kmedoids::benchmark::mnist_bench();
    // dbscan::benchmark::noisy_circles_bench();

    auto ffcl_durations_summary  = kdtree::benchmark::ffcl_::radius_search_around_query_index_varied_bench();
    auto pcl_durations_summary   = kdtree::benchmark::pcl_::radius_search_around_query_index_varied_bench();
    auto flann_durations_summary = kdtree::benchmark::flann_::radius_search_around_query_index_varied_bench();

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

    return 0;
}