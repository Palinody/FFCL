#include "./DBSCANBench.hpp"
#include "./KDTreeBench.hpp"
#include "./KMeansBench.hpp"
#include "./KMedoidsBench.hpp"

int main() {
    // kmeans::benchmark::mnist_bench();
    // kmedoids::benchmark::mnist_bench();
    dbscan::benchmark::noisy_circles_bench();
    // kdtree::benchmark::radius_search_around_query_index_varied_bench();

    return 0;
}