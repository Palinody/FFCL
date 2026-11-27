// FLANN adapter, exact single tree. Own executable so the other libraries' code cannot perturb its
// code layout.
#include "refbench_common.hpp"

#include <flann/flann.hpp>

using namespace refbench;

namespace {

void run(const Dataset& dataset, const char* workload, std::size_t threads) {
    using flann_matrix = flann::Matrix<double>;
    using flann_result = flann::Matrix<int>;
    using flann_dist   = flann::Matrix<double>;

    const auto n = dataset.n_samples;
    const auto d = dataset.n_features;

    measure("flann", "kdtree", "tree_build", dataset, kLeaf, threads, [&] {
        auto data   = flann_matrix(const_cast<double*>(dataset.data.data()), n, d);
        // KDTreeSingleIndexParams, not KDTreeIndexParams: the latter's argument is the number of
        // randomised trees, not the leaf size.
        auto params = flann::KDTreeSingleIndexParams(static_cast<int>(kLeaf));
        auto index  = flann::Index<flann::L2_Simple<double>>(data, params);
        index.buildIndex();
    }, "leaf=40 single tree");

    if (std::string{workload} != "knn") return;

    auto data   = flann_matrix(const_cast<double*>(dataset.data.data()), n, d);
    auto params = flann::KDTreeSingleIndexParams(static_cast<int>(kLeaf));
    auto index  = flann::Index<flann::L2_Simple<double>>(data, params);
    index.buildIndex();

    auto queries = flann_matrix(const_cast<double*>(dataset.data.data()), kNQueries, d);
    auto indices = flann_result(new int[kNQueries * kK], kNQueries, kK);
    auto dists   = flann_dist(new double[kNQueries * kK], kNQueries, kK);

    measure("flann", "kdtree", "knn", dataset, kLeaf, threads, [&] {
        index.knnSearch(queries, indices, dists, static_cast<int>(kK), flann::SearchParams(-1));
    }, "k=2 checks=exhaustive");

    delete[] indices.ptr();
    delete[] dists.ptr();
}

}  // namespace

int main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    if (options.datasets_dir.empty()) {
        std::fprintf(stderr, "usage: refbench_flann --datasets DIR [--workload build|knn] [--threads N]\n");
        return 2;
    }
    announce(options);
    return for_each_dataset(options, [&](const Dataset& dataset) {
        run(dataset, options.workload.c_str(), options.threads);
    });
}
