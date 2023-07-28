#include "DBSCANBench.hpp"
#include "KDTreeBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

#include <stdio.h>

int main() {
    // kmeans::benchmark::mnist_bench();
    // kmedoids::benchmark::mnist_bench();
    dbscan::benchmark::run_pointclouds_benchmarks();

    // kdtree::benchmark::run_toy_datasets_benchmarks();

    // kdtree::benchmark::run_pointclouds_benchmarks();

    return 0;
}