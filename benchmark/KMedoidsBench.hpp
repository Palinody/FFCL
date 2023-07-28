#pragma once

#include "IO.hpp"
#include "Utils.hpp"

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/containers/LowerTriangleMatrix.hpp"
#include "ffcl/kmedoids/KMedoids.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"

namespace kmedoids::benchmark {

template <typename InputsIterator>
void fit_once(const InputsIterator& inputs_first,
              const InputsIterator& inputs_last,
              std::size_t           n_medoids,
              std::size_t           n_features) {
    using KMedoids = ffcl::KMedoids<dType, true>;
    // using PAM = ffcl::FasterMSC;

    auto kmedoids = KMedoids(n_medoids, n_features);

    kmedoids.set_options(
        /*KMedoids options=*/KMedoids::Options().max_iter(10).early_stopping(true).patience(0).n_init(1));

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "\nRunning kmedoids dynamically: \n";
#endif

    common::timer::Timer<common::timer::Nanoseconds> timer;

    kmedoids.fit<ffcl::FasterMSC>(inputs_first, inputs_last);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

template <typename InputsIterator>
void fit_once_with_pairwise_distance_matrix(const InputsIterator& inputs_first,
                                            const InputsIterator& inputs_last,
                                            std::size_t           n_medoids,
                                            std::size_t           n_features) {
    using KMedoids = ffcl::KMedoids<dType, true>;
    // using PAM = ffcl::FasterMSC;

    auto kmedoids = KMedoids(n_medoids, n_features);

    kmedoids.set_options(
        /*KMedoids options=*/KMedoids::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

    const auto pairwise_distance_matrix = ffcl::containers::LowerTriangleMatrix(inputs_first, inputs_last, n_features);

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "\nRunning kmedoids with precomputed distances matrix: \n";
#endif

    common::timer::Timer<common::timer::Nanoseconds> timer;

    const auto medoids = kmedoids.fit<ffcl::FasterMSC>(pairwise_distance_matrix);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif

    timer.reset();

    const auto centroids = pam::utils::medoids_to_centroids(inputs_first, inputs_last, n_features, medoids);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

void distance_matrix_benchmark() {
    fs::path filename = "mnist.txt";

    const auto        data       = bench::io::txt::load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = bench::io::txt::get_num_features_in_file(inputs_folder / filename);

    common::timer::Timer<common::timer::Nanoseconds> timer;

    ffcl::containers::LowerTriangleMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

void mnist_bench() {
    fs::path filename = "mnist.txt";

    const auto        data       = bench::io::txt::load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = bench::io::txt::get_num_features_in_file(inputs_folder / filename);

    std::cout << data.size() << " | " << n_features << "\n\n";
    const auto n_medoids = 10;

    fit_once(data.begin(), data.end(), n_medoids, n_features);

    fit_once_with_pairwise_distance_matrix(data.begin(), data.end(), n_medoids, n_features);
}

}  // namespace kmedoids::benchmark