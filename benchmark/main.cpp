#include "DBSCANBench.hpp"
#include "KDTreeBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

#include <stdio.h>

int main() {
    // kmeans::benchmark::mnist_bench();
    // kmedoids::benchmark::mnist_bench();
    // dbscan::benchmark::noisy_circles_bench();

    auto durations_summary_1 = kdtree::benchmark::radius_search_around_query_index_varied_bench();
    auto durations_summary_2 = kdtree::benchmark::pcl_radius_search_around_query_index_varied_bench();

    double to_seconds = 1e-9;  // Conversion factor for nanoseconds to seconds

    // Duration Summary 1 (Original Time)
    double ffcl_build = durations_summary_1.indexer_build_duration * to_seconds;
    double ffcl_query = durations_summary_1.indexer_query_duration * to_seconds;
    double ffcl_total = durations_summary_1.total_duration * to_seconds;

    // Duration Summary 2 (Improved Time)
    double pcl_build = durations_summary_2.indexer_build_duration * to_seconds;
    double pcl_query = durations_summary_2.indexer_query_duration * to_seconds;
    double pcl_total = durations_summary_2.total_duration * to_seconds;

    printf("KDTree (FFCL)\n\tbuild: %.6f | queries: %.6f | total: %.6f\n", ffcl_build, ffcl_query, ffcl_total);

    printf("KDTree (PCL)\n\tbuild: %.6f | queries: %.6f | total: %.6f\n", pcl_build, pcl_query, pcl_total);

    // Calculate the speedup as a percentage for each duration
    auto build_speedup = (pcl_build - ffcl_build) / pcl_build * 100;
    auto query_speedup = (pcl_query - ffcl_query) / pcl_query * 100;
    auto total_speedup = (pcl_total - ffcl_total) / pcl_total * 100;

    printf("KDTree (FFCL speedup over PCL)\n\tbuild: %.3f | queries: %.3f | total: %.3f\n",
           build_speedup,
           query_speedup,
           total_speedup);

    return 0;
}