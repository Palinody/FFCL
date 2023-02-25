#include "cpp_clustering/common/Timer.hpp"
#include "cpp_clustering/containers/LowerTriangleMatrix.hpp"
#include "cpp_clustering/kmeans/KMeans.hpp"
#include "cpp_clustering/kmedoids/KMedoids.hpp"
#include "cpp_clustering/math/random/VosesAliasMethod.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

using dType = float;

const fs::path folder_root        = fs::path("../datasets/clustering");
const fs::path inputs_folder      = folder_root / fs::path("inputs");
const fs::path targets_folder     = folder_root / fs::path("targets");
const fs::path predictions_folder = folder_root / fs::path("predictions");
const fs::path centroids_folder   = folder_root / fs::path("centroids");

ssize_t get_num_features_in_file(const fs::path& filepath, char delimiter = ' ') {
    std::ifstream file(filepath);
    ssize_t       n_features = -1;
    if (file.is_open()) {
        std::string line;
        std::getline(file, line);
        n_features = std::count(line.begin(), line.end(), ' ') + 1;
        file.close();
    } else {
        throw std::ios_base::failure("Unable to open file: " + filepath.string());
    }
    return n_features;
}

template <typename T = float>
std::vector<T> load_data(const fs::path& filename, char delimiter = ' ') {
    std::ifstream  filestream(filename);
    std::vector<T> data;

    if (filestream.is_open()) {
        // temporary string data
        std::string row_str, elem_str;
        while (std::getline(filestream, row_str, '\n')) {
            std::stringstream row_str_stream(row_str);
            while (std::getline(row_str_stream, elem_str, delimiter)) {
                data.emplace_back(std::stof(elem_str));
            }
        }
        filestream.close();
    }
    return data;
}

template <typename T = float>
void write_data(const std::vector<T>& data, std::size_t n_features, const fs::path& filename) {
    std::ofstream filestream(filename);

    if (filestream.is_open()) {
        std::size_t iter{};
        for (const auto& elem : data) {
            filestream << elem;
            ++iter;
            if (iter % n_features == 0 && iter != 0) {
                filestream << '\n';
            } else {
                filestream << ' ';
            }
        }
    }
}

template <typename InputsIterator, typename LabelsIterator>
void fit_once_kmeans(const InputsIterator& inputs_first,
                     const InputsIterator& inputs_last,
                     LabelsIterator        labels_first,
                     LabelsIterator        labels_last,
                     std::size_t           n_centroids,
                     std::size_t           n_features) {
    using KMeans = cpp_clustering::KMeans<dType>;

    auto kmeans = KMeans(n_centroids, n_features);

    kmeans.set_options(
        /*KMeans options=*/KMeans::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "\nRunning kmeans: \n";
#endif

    common::timer::Timer<common::timer::Nanoseconds> timer;

    std::cout << "HERE: "
              << "\n";

    kmeans.fit(inputs_first, inputs_last);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

template <typename InputsIterator, typename LabelsIterator>
void fit_once(const InputsIterator& inputs_first,
              const InputsIterator& inputs_last,
              LabelsIterator        labels_first,
              LabelsIterator        labels_last,
              std::size_t           n_medoids,
              std::size_t           n_features) {
    using KMedoids = cpp_clustering::KMedoids<dType, true>;
    // using PAM = cpp_clustering::FasterMSC;

    auto kmedoids = KMedoids(n_medoids, n_features);

    kmedoids.set_options(
        /*KMedoids options=*/KMedoids::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "\nRunning kmedoids dynamically: \n";
#endif

    common::timer::Timer<common::timer::Nanoseconds> timer;

    kmedoids.fit<cpp_clustering::FasterMSC>(inputs_first, inputs_last);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

template <typename InputsIterator, typename LabelsIterator>
void fit_once_with_pairwise_distance_matrix(const InputsIterator& inputs_first,
                                            const InputsIterator& inputs_last,
                                            LabelsIterator        labels_first,
                                            LabelsIterator        labels_last,
                                            std::size_t           n_medoids,
                                            std::size_t           n_features) {
    using KMedoids = cpp_clustering::KMedoids<dType, true>;
    // using PAM = cpp_clustering::FasterMSC;

    auto kmedoids = KMedoids(n_medoids, n_features);

    kmedoids.set_options(
        /*KMedoids options=*/KMedoids::Options().max_iter(100).early_stopping(true).patience(0).n_init(1));

    const auto pairwise_distance_matrix =
        cpp_clustering::containers::LowerTriangleMatrix(inputs_first, inputs_last, n_features);

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "\nRunning kmedoids with precomputed distances matrix: \n";
#endif

    common::timer::Timer<common::timer::Nanoseconds> timer;

    kmedoids.fit<cpp_clustering::FasterMSC>(pairwise_distance_matrix);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif

    timer.reset();

    const auto [best_match_count_remap, swapped_medoids] =
        kmedoids.remap_centroid_to_label_index(pairwise_distance_matrix, labels_first, labels_last, n_medoids);

    const auto swapped_centroids =
        pam::utils::medoids_to_centroids(inputs_first, inputs_last, n_features, swapped_medoids);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
    std::cout << "Best match count remap: " << best_match_count_remap << "\n";
}

void distance_matrix_benchmark() {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    common::timer::Timer<common::timer::Nanoseconds> timer;

    cpp_clustering::containers::LowerTriangleMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);

#if defined(VERBOSE) && VERBOSE == true
    timer.print_elapsed_seconds(/*n_decimals=*/6);
#endif
}

void mnist_train_benchmark() {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    std::cout << data.size() << " | " << labels.size() << " | " << n_features << "\n\n";
    const auto n_medoids = 10;
    // const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    fit_once_kmeans(data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features);

    fit_once(data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features);

    fit_once_with_pairwise_distance_matrix(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features);
}

int main() {
#if defined(VERBOSE) && VERBOSE == true
    std::cout << "Making the pairwise distance matrix: \n";
#endif
    distance_matrix_benchmark();

#if defined(VERBOSE) && VERBOSE == true
    std::cout << "Making the pairwise distance matrix and kmedoids fit: \n";
#endif
    mnist_train_benchmark();

    return 0;
}