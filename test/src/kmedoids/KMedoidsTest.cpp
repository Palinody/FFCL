#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/kmedoids/KMedoids.hpp"
#include "ffcl/math/random/Sampling.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class KMedoidsErrorsTest : public ::testing::Test {
  public:
    using dType = float;

  protected:
    ssize_t get_num_features_in_file(const fs::path& filepath, char delimiter = ' ') {
        std::ifstream file(filepath);
        ssize_t       n_features = -1;
        if (file.is_open()) {
            std::string line;

            std::getline(file, line);
            // count the number of values at the first line, delimited by the specified delimiter
            n_features = std::count(line.begin(), line.end(), delimiter) + 1;

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

    void make_directories(const fs::path& directory_path) {
        try {
            if (!std::filesystem::exists(directory_path)) {
                std::filesystem::create_directories(directory_path);

#if defined(VERBOSE) && VERBOSE == true
                std::cout << "Directory created: " << directory_path << "\n";

            } else {
                std::cout << "Dir. already exists\n";
#endif
            }
        } catch (std::filesystem::filesystem_error& e) {
            std::cerr << e.what() << std::endl;
        }
    }

    template <typename T = float>
    void write_data(const std::vector<T>& data, std::size_t n_features, const fs::path& filename) {
        const auto parent_path = filename.parent_path();

        make_directories(parent_path);

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

    template <typename InputsIterator>
    void simple_fit(const InputsIterator& inputs_first,
                    const InputsIterator& inputs_last,
                    std::size_t           n_medoids,
                    std::size_t           n_features,
                    std::size_t           n_iterations = 1) {
        using KMedoids = ffcl::KMedoids<dType, true>;
        // using PAM = ffcl::FasterMSC;

        auto kmedoids = KMedoids(n_medoids, n_features);

        kmedoids.set_options(
            /*KMedoids options=*/KMedoids::Options().max_iter(n_iterations).early_stopping(true).patience(0).n_init(1));

        const auto medoids   = kmedoids.fit<ffcl::FasterPAM>(inputs_first, inputs_last);
        const auto centroids = pam::utils::medoids_to_centroids(inputs_first, inputs_last, n_features, medoids);
    }

    template <typename InputsIterator>
    std::pair<std::vector<std::size_t>, std::vector<typename InputsIterator::value_type>> fit_predict(
        const InputsIterator& inputs_first,
        const InputsIterator& inputs_last,
        std::size_t           n_medoids,
        std::size_t           n_features,
        std::size_t           n_iterations = 1) {
        using KMedoids = ffcl::KMedoids<dType, true>;
        // using PAM = ffcl::FasterMSC;

        using DatasetDescriptorType              = std::tuple<InputsIterator, InputsIterator, std::size_t>;
        DatasetDescriptorType dataset_descriptor = std::make_tuple(inputs_first, inputs_last, n_features);
        const auto            pairwise_distance_matrix =
            ffcl::datastruct::PairwiseDistanceMatrix<InputsIterator>(dataset_descriptor);

        auto kmedoids = KMedoids(n_medoids, n_features);

        kmedoids.set_options(KMedoids::Options().max_iter(n_iterations).early_stopping(true).patience(0).n_init(10));

        const auto medoids = kmedoids.fit<ffcl::FasterPAM>(pairwise_distance_matrix);

        const auto centroids = pam::utils::medoids_to_centroids(inputs_first, inputs_last, n_features, medoids);

        const auto predictions = kmedoids.predict(inputs_first, inputs_last);

        return {predictions, centroids};
    }

    static constexpr std::size_t n_iterations_global = 100;
    static constexpr std::size_t n_medoids_global    = 4;

    const fs::path folder_root_        = fs::path("../bin/clustering");
    const fs::path inputs_folder_      = folder_root_ / fs::path("inputs");
    const fs::path targets_folder_     = folder_root_ / fs::path("targets");
    const fs::path predictions_folder_ = folder_root_ / fs::path("predictions");
    const fs::path centroids_folder_   = folder_root_ / fs::path("centroids");
};

TEST_F(KMedoidsErrorsTest, NoisyCirclesTest) {
    fs::path filename = "noisy_circles.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 2;  // 2
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, NoisyMoonsTest) {
    fs::path filename = "noisy_moons.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 2;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, VariedTest) {
    fs::path filename = "varied.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, AnisoTest) {
    fs::path filename = "aniso.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, BlobsTest) {
    fs::path filename = "blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const auto n_medoids = 4;
    // const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, IrisTest) {
    fs::path filename = "iris.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, UnbalancedBlobsTest) {
    fs::path filename = "unbalanced_blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        fit_predict(data.begin(), data.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, PairwiseDistanceMatrixTest) {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    auto pairwise_distance_matrix =
        ffcl::datastruct::PairwiseDistanceMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);
}

TEST_F(KMedoidsErrorsTest, MnistSimpleFitTest) {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    std::cout << data.size() << " | " << labels.size() << " | " << n_features << std::endl;
    const auto n_medoids = 10;

    simple_fit(data.begin(), data.end(), n_medoids, n_features, 100);
}

TEST_F(KMedoidsErrorsTest, ClusterInitializationTest) {
    /*
    const std::size_t n_samples   = 10;
    const std::size_t n_features  = 3;
    const std::size_t n_centroids = 0;
    auto              data        = std::vector<float>(n_samples * n_features);
    std::iota(data.begin(), data.end(), 0);

    std::cout << "data:" << std::endl;

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            std::cout << data[i * n_features + j] << " ";
        }
        std::cout << std::endl;
    }
    const auto centroids = math::random::init_uniform(data.begin(), data.end(), n_centroids, n_features);

    std::cout << "centroids:" << std::endl;

    for (std::size_t i = 0; i < n_centroids; ++i) {
        for (std::size_t j = 0; j < n_features; ++j) {
            std::cout << centroids[i * n_features + j] << " ";
            // ASSERT_FLOAT_EQ(centroids[i * n_features + j], data[no_random_index_access * n_features + j], 1e-7);
        }
        std::cout << std::endl;
    }
    */
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
