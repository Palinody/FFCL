// FFCL benchmark entry point.
//
// Writes one JSON line per measurement into benchmark/bin/results/ffcl.json. The reference libraries
// are measured by benchmark/scripts/benchmark_reference.py and land in the same format, then
// benchmark/scripts/make_tables.py renders benchmark/README.md.
#include "ffcl/benchmark/Dataset.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace ffcl::benchmark {

void benchmark_kdtree(std::ostream& results, std::size_t n_queries, std::size_t k_nearest_neighbors);
void benchmark_dbscan(std::ostream& results, float radius, std::size_t min_samples);
void benchmark_hdbscan(std::ostream& results, std::size_t k_nearest_neighbors, std::size_t min_cluster_size);
void benchmark_kmeans(std::ostream& results, std::size_t n_centroids);
void benchmark_kmedoids(std::ostream& results, std::size_t n_medoids);
void benchmark_minimum_spanning_tree(std::ostream& results, std::size_t k_nearest_neighbors);
void benchmark_flann(std::ostream& results, std::size_t n_queries, std::size_t k_nearest_neighbors);

}  // namespace ffcl::benchmark

namespace {

// Parameters shared with benchmark/scripts/benchmark_reference.py. Keep the two in sync: the tables
// are only meaningful when both sides use the same settings.
constexpr std::size_t kNQueries         = 1000;
constexpr std::size_t kNearestNeighbors = 2;
constexpr std::size_t kMinClusterSize   = 20;
constexpr std::size_t kNCentroids       = 8;
constexpr std::size_t kNMedoids         = 8;
constexpr float       kDbscanRadius     = 0.5f;
constexpr std::size_t kDbscanMinSamples = 5;

// FFCL's core test is strictly `size() > min_samples` and its radius buffer includes the query
// sample, so FFCL min_samples = 5 means 6 points including itself, while every reference library's
// min_pts = 5 means 5 including itself. FFCL_DBSCAN_MIN_SAMPLES=4 matches the references; the
// published tables were measured at 5 and are therefore one stricter.
auto dbscan_min_samples() -> std::size_t {
    if (const auto* value = std::getenv("FFCL_DBSCAN_MIN_SAMPLES")) {
        return std::stoul(value);
    }
    return kDbscanMinSamples;
}

}  // namespace

int main(int argument_count, char** argument_values) {
    const auto selected =
        argument_count > 1 ? std::string(argument_values[1]) : std::string{"all"};

    ffcl::benchmark::set_repository_root_from_executable(argument_values[0]);

    const auto results_root = ffcl::benchmark::results_root();

    std::error_code error_code;
    std::filesystem::create_directories(results_root, error_code);

    auto results =
        std::ofstream(results_root / "ffcl.json", std::ios::out | std::ios::trunc);

    if (!results.is_open()) {
        std::cerr << "cannot write " << (results_root / "ffcl.json") << "\n";
        return 1;
    }
    const auto wanted = [&](const char* name) {
        return selected == "all" || selected == name;
    };

    try {
        if (wanted("kdtree")) {
            std::cout << "kdtree...\n" << std::flush;
            ffcl::benchmark::benchmark_kdtree(results, kNQueries, kNearestNeighbors);
        }
        if (wanted("flann")) {
            std::cout << "flann...\n" << std::flush;
            ffcl::benchmark::benchmark_flann(results, kNQueries, kNearestNeighbors);
        }
        if (wanted("mst")) {
            std::cout << "boruvka (single tree / dual tree)...\n" << std::flush;
            ffcl::benchmark::benchmark_minimum_spanning_tree(results, kNearestNeighbors);
        }
        if (wanted("hdbscan")) {
            std::cout << "hdbscan...\n" << std::flush;
            ffcl::benchmark::benchmark_hdbscan(results, kNearestNeighbors, kMinClusterSize);
        }
        if (wanted("dbscan")) {
            std::cout << "dbscan...\n" << std::flush;
            ffcl::benchmark::benchmark_dbscan(results, kDbscanRadius, dbscan_min_samples());
        }
        if (wanted("kmeans")) {
            std::cout << "kmeans...\n" << std::flush;
            ffcl::benchmark::benchmark_kmeans(results, kNCentroids);
        }
        if (wanted("kmedoids")) {
            std::cout << "kmedoids...\n" << std::flush;
            ffcl::benchmark::benchmark_kmedoids(results, kNMedoids);
        }
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << "\n";
        return 1;
    }
    std::cout << "results written to " << (results_root / "ffcl.json") << "\n";

    return 0;
}
