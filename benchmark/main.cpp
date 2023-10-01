#include "DBSCANBench.hpp"
#include "KDTreeBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

#include "ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp"

#include <stdio.h>

int main() {
    // kmeans::benchmark::bench_mnist();
    // kmedoids::benchmark::bench_mnist();

    // dbscan::benchmark::run_dbscan_benchmarks_on_point_cloud_sequences();

    // kdtree::benchmark::run_toy_datasets_benchmarks();

    kdtree::benchmark::run_radius_search_benchmarks_on_point_cloud_sequences();
    kdtree::benchmark::run_k_nearest_neighbors_search_benchmarks_on_point_cloud_sequences();

    return 0;
}