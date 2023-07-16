#include "DBSCANBench.hpp"
// #include "KDTreeBench.hpp"
#include "KMeansBench.hpp"
#include "KMedoidsBench.hpp"

int main() {
    // benchmark::kmeans::mnist_bench();
    // benchmark::kmedoids::mnist_bench();
    benchmark::dbscan::noisy_circles_bench();
    // benchmark::kdtree::radius_search_around_query_index_varied_bench();

    return 0;
}