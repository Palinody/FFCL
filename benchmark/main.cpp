#include "DBSCANBench.hpp"
#include "KDTreeBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

#include <stdio.h>

int main() {
    // kmeans::benchmark::mnist_bench();
    // kmedoids::benchmark::mnist_bench();
    // dbscan::benchmark::noisy_circles_bench();

    kdtree::benchmark::run_benchmarks();

    return 0;
}