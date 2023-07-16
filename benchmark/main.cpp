#include "DBSCANBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

int main() {
    // benchmark::kmeans::mnist_bench();
    // benchmark::kmedoids::mnist_bench();

    benchmark::dbscan::noisy_circles_bench();

    return 0;
}