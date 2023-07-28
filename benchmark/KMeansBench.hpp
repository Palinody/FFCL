#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/kmeans/KMeans.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"

namespace kmeans::benchmark {

template <typename InputsIterator>
void fit_once_kmeans(const InputsIterator& inputs_first,
                     const InputsIterator& inputs_last,
                     std::size_t           n_centroids,
                     std::size_t           n_features) {
    using KMeans = ffcl::KMeans<dType>;

    auto kmeans = KMeans(n_centroids, n_features);

    kmeans.set_options(
        /*KMeans options=*/KMeans::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "\nRunning kmeans: \n";
#endif

    common::timer::Timer<common::timer::Nanoseconds> timer;

    kmeans.fit(inputs_first, inputs_last);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

void mnist_bench() {
    fs::path filename = "mnist.txt";

    const auto        data       = bench::io::txt::load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = bench::io::txt::get_num_features_in_file(inputs_folder / filename);

    std::cout << data.size() << " | " << n_features << "\n\n";
    const auto n_centroids = 10;

    fit_once_kmeans(data.begin(), data.end(), n_centroids, n_features);
}

}  // namespace kmeans::benchmark