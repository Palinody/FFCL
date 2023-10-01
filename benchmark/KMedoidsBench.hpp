#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp"
#include "ffcl/kmedoids/KMedoids.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"

namespace kmedoids::benchmark {

template <typename SamplesIterator>
void dynamic_fit(const SamplesIterator& samples_first,
                 const SamplesIterator& samples_last,
                 std::size_t            n_medoids,
                 std::size_t            n_features) {
    using KMedoids = ffcl::KMedoids<bench::io::DataType, false>;
    // using PAM = ffcl::FasterMSC;

    auto kmedoids = KMedoids(n_medoids, n_features);

    kmedoids.set_options(
        /*KMedoids options=*/KMedoids::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

    std::cout << "Dynamic kmedoids: \n";

    common::timer::Timer<common::timer::Nanoseconds> timer;

    kmedoids.fit<ffcl::FasterPAM>(samples_first, samples_last);

    timer.print_elapsed_seconds(/*n_decimals=*/6);
}

template <typename SamplesIterator>
void precomputed_fit(const SamplesIterator& samples_first,
                     const SamplesIterator& samples_last,
                     std::size_t            n_medoids,
                     std::size_t            n_features) {
    using KMedoids = ffcl::KMedoids<bench::io::DataType, false>;
    // using PAM = ffcl::FasterMSC;

    auto kmedoids = KMedoids(n_medoids, n_features);

    kmedoids.set_options(
        /*KMedoids options=*/KMedoids::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

    std::cout << "Pairwise distances matrix: \n";

    common::timer::Timer<common::timer::Nanoseconds> timer;

    const auto pairwise_distance_matrix =
        ffcl::datastruct::PairwiseDistanceMatrix(samples_first, samples_last, n_features);

    timer.print_elapsed_seconds(/*n_decimals=*/6);

    std::cout << "Precomputed pairwise distances kmedoids: \n";

    timer.reset();

    const auto medoids = kmedoids.fit<ffcl::FasterPAM>(pairwise_distance_matrix);

    timer.print_elapsed_seconds(/*n_decimals=*/6);
}

void distance_matrix_mnist() {
    fs::path filename = "mnist.txt";

    const auto        data = bench::io::txt::load_data<bench::io::DataType>(bench::io::inputs_folder / filename, ' ');
    const std::size_t n_features = bench::io::txt::get_num_features_in_file(bench::io::inputs_folder / filename);

    common::timer::Timer<common::timer::Nanoseconds> timer;

    ffcl::datastruct::PairwiseDistanceMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);

    timer.print_elapsed_seconds(/*n_decimals=*/6);
}

void bench_mnist() {
    fs::path filename = "mnist.txt";

    const auto        data = bench::io::txt::load_data<bench::io::DataType>(bench::io::inputs_folder / filename, ' ');
    const std::size_t n_features = bench::io::txt::get_num_features_in_file(bench::io::inputs_folder / filename);

    std::cout << data.size() << " | " << n_features << "\n\n";
    const auto n_medoids = 10;

    dynamic_fit(data.begin(), data.end(), n_medoids, n_features);

    precomputed_fit(data.begin(), data.end(), n_medoids, n_features);
}

}  // namespace kmedoids::benchmark