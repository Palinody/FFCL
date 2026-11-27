// pico_tree adapter: build, kNN and radius. Own executable so the other libraries' code cannot
// perturb its code layout.
#include "refbench_common.hpp"

#include <pico_tree/kd_tree.hpp>
#include <pico_tree/map_traits.hpp>

using namespace refbench;

namespace {

using PicoPoint    = pico_tree::point_map<double, pico_tree::dynamic_extent>;
using PicoSpace    = pico_tree::space_map<PicoPoint>;
using PicoTree     = pico_tree::kd_tree<PicoSpace>;
using PicoNeighbor = pico_tree::neighbor<int, double>;

void run(const Dataset& dataset, const char* workload, std::size_t threads) {
    const auto n = dataset.n_samples;
    const auto d = dataset.n_features;
    // The tree only stores split bounds and index ranges; the points stay in `dataset.data`.
    auto space = PicoSpace(const_cast<double*>(dataset.data.data()), n, d);

    measure("pico_tree", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
        auto tree = PicoTree(space, pico_tree::max_leaf_size_t(kLeaf));
        if (tree.space().size() == 0) std::abort();
    }, "leaf=40");

    if (std::string{workload} == "knn") {
        auto tree = PicoTree(space, pico_tree::max_leaf_size_t(kLeaf));
        measure("pico_tree", "kdtree", "knn", dataset, kLeaf, threads, [&] {
            auto knn   = std::vector<PicoNeighbor>{};
            auto total = 0.0;
            for (std::size_t query = 0; query < kNQueries; ++query) {
                auto point = PicoPoint(const_cast<double*>(&dataset.data[query * d]), d);
                tree.search_knn(point, kK, knn);
                total += knn.back().distance;
            }
            if (total < 0) std::abort();
        }, "k=2 squared-l2");
    } else if (std::string{workload} == "radius") {
        auto tree = PicoTree(space, pico_tree::max_leaf_size_t(kLeaf));
        measure("pico_tree", "kdtree", "radius", dataset, kLeaf, threads, [&] {
            auto neighbors = std::vector<PicoNeighbor>{};
            auto total     = std::size_t{0};
            for (std::size_t query = 0; query < kNQueries; ++query) {
                auto point = PicoPoint(const_cast<double*>(&dataset.data[query * d]), d);
                // The default metric is squared L2, so the radius is squared too.
                tree.search_radius(point, kRadius * kRadius, neighbors);
                total += neighbors.size();
            }
            if (total == 0) std::abort();
        }, "radius=0.5 squared-l2");
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty()) {
        std::fprintf(stderr, "usage: refbench_pico_tree --datasets DIR [--workload build|knn|radius]\n");
        return 2;
    }
    announce(options);
    return for_each_dataset(options, [&](const Dataset& dataset) {
        run(dataset, options.workload.c_str(), options.threads);
    });
}
