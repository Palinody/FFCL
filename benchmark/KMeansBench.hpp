#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/kmeans/KMeans.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"

namespace kmeans::benchmark {

template <typename SamplesIterator>
void fit_once(const SamplesIterator& samples_first,
              const SamplesIterator& samples_last,
              std::size_t            n_centroids,
              std::size_t            n_features) {
    using KMeans = ffcl::KMeans<bench::io::DataType>;

    auto kmeans = KMeans(n_centroids, n_features);

    kmeans.set_options(KMeans::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

    std::cout << "KMeans: \n";

    common::timer::Timer<common::timer::Nanoseconds> timer;

    kmeans.fit(samples_first, samples_last);

    timer.print_elapsed_seconds(/*n_decimals=*/6);
}

void bench_mnist() {
    fs::path filename = "mnist.txt";

    const auto        data = bench::io::txt::load_data<bench::io::DataType>(bench::io::inputs_folder / filename, ' ');
    const std::size_t n_features = bench::io::txt::get_num_features_in_file(bench::io::inputs_folder / filename);

    std::cout << data.size() << " | " << n_features << "\n\n";
    const auto n_centroids = 10;

    fit_once(data.begin(), data.end(), n_centroids, n_features);
}

}  // namespace kmeans::benchmark