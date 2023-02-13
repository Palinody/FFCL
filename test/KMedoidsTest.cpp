#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "cpp_clustering/kmedoids/KMedoids.hpp"
#include "cpp_clustering/math/random/VosesAliasMethod.hpp"

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

    template <typename InputsIterator, typename LabelsIterator>
    void simple_fit(const InputsIterator& inputs_first,
                    const InputsIterator& inputs_last,
                    LabelsIterator        labels_first,
                    LabelsIterator        labels_last,
                    std::size_t           n_medoids,
                    std::size_t           n_features,
                    std::size_t           n_iterations = 1) {
        using KMedoids = cpp_clustering::KMedoids<dType, true>;
        // using PAM = cpp_clustering::FasterMSC;

        auto kmedoids = KMedoids(n_medoids, n_features);

        kmedoids.set_options(
            /*KMedoids options=*/KMedoids::Options().max_iter(n_iterations).early_stopping(true).patience(0).n_init(1));

        const auto centroids = kmedoids.fit<cpp_clustering::FasterPAM>(inputs_first, inputs_last);
    }

    template <typename InputsIterator, typename LabelsIterator>
    std::pair<std::vector<std::size_t>, std::vector<typename InputsIterator::value_type>> fit_predict_orig(
        const InputsIterator& inputs_first,
        const InputsIterator& inputs_last,
        LabelsIterator        labels_first,
        LabelsIterator        labels_last,
        std::size_t           n_medoids,
        std::size_t           n_features,
        std::size_t           n_iterations = 1) {
        using KMedoids = cpp_clustering::KMedoids<dType, true>;
        // using PAM = cpp_clustering::FasterMSC;

        // common::clustering::kmeansplusplus::make_centroids<InputsIterator>

        const auto uniform_random_indices =
            common::utils::select_from_range(n_medoids, {0, (inputs_last - inputs_first) / n_features});
        const auto uniform_random_centroids =
            common::utils::init_uniform(inputs_first, inputs_last, n_medoids, n_features);

        auto kmedoids = KMedoids(n_medoids, n_features);

        // KMedoids::Options().max_iter(n_iterations).n_init(10).early_stopping(true).patience(5).tolerance(0)
        // KMedoids::Options().max_iter(n_iterations).early_stopping(true).patience(0)
        kmedoids.set_options(
            /*KMedoids options=*/KMedoids::Options().max_iter(n_iterations).early_stopping(true).patience(0).n_init(1));

        const auto centroids = kmedoids.fit<cpp_clustering::FasterMSC>(inputs_first, inputs_last);

        // const auto [best_match_count, swapped_centroids] =
        // kmedoids.swap_to_best_count_match(inputs_first, inputs_last, labels_first, labels_last);

        const auto predictions = kmedoids.predict(inputs_first, inputs_last);

        // std::cout << "Best match count: " << best_match_count << " / " << std::distance(labels_first, labels_last)
        //   << std::endl;

        return {predictions, centroids};
    }

    template <typename InputsIterator, typename LabelsIterator>
    std::pair<std::vector<std::size_t>, std::vector<typename InputsIterator::value_type>> fit_predict(
        const InputsIterator& inputs_first,
        const InputsIterator& inputs_last,
        LabelsIterator        labels_first,
        LabelsIterator        labels_last,
        std::size_t           n_medoids,
        std::size_t           n_features,
        std::size_t           n_iterations = 1) {
        using KMedoids = cpp_clustering::KMedoids<dType>;
        // using PAM = cpp_clustering::FasterMSC;

        // common::clustering::kmeansplusplus::make_centroids<InputsIterator>

        const auto uniform_random_indices =
            common::utils::select_from_range(n_medoids, {0, (inputs_last - inputs_first) / n_features});
        const auto uniform_random_centroids =
            common::utils::init_uniform(inputs_first, inputs_last, n_medoids, n_features);

        auto kmedoids = KMedoids(n_medoids, n_features);

        // KMedoids::Options().max_iter(n_iterations).n_init(10).early_stopping(true).patience(5).tolerance(0)
        // KMedoids::Options().max_iter(n_iterations).early_stopping(true).patience(0)
        kmedoids.set_options(
            /*KMedoids options=*/KMedoids::Options()
                .max_iter(n_iterations)
                .early_stopping(true)
                .patience(0)
                .n_init(10));

        const auto centroids = kmedoids.fit<cpp_clustering::FasterMSC>(inputs_first, inputs_last);

        // const auto predictions = kmedoids.predict(inputs_first, inputs_last);

        const auto [best_match_count, swapped_centroids] =
            kmedoids.remap_centroid_to_label_index(inputs_first, inputs_last, labels_first, labels_last, n_medoids);

        const auto new_predictions = kmedoids.predict(inputs_first, inputs_last);

        std::cout << "Best match count: " << best_match_count << " / " << std::distance(labels_first, labels_last)
                  << std::endl;

        return {new_predictions, swapped_centroids};
    }

    static constexpr std::size_t n_iterations_global = 10;
    static constexpr std::size_t n_medoids_global    = 4;

    const fs::path folder_root        = fs::path("../datasets/clustering");
    const fs::path inputs_folder      = folder_root / fs::path("inputs");
    const fs::path targets_folder     = folder_root / fs::path("targets");
    const fs::path predictions_folder = folder_root / fs::path("predictions");
    const fs::path centroids_folder   = folder_root / fs::path("centroids");
};

TEST_F(KMedoidsErrorsTest, NoisyCirclesTest) {
    fs::path filename = "noisy_circles.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 2;  // 2
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, NoisyMoonsTest) {
    fs::path filename = "noisy_moons.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 2;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, VariedTest) {
    fs::path filename = "varied.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, AnisoTest) {
    fs::path filename = "aniso.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, BlobsTest) {
    fs::path filename = "blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    const auto n_medoids = 4;
    // const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, IrisTest) {
    fs::path filename = "iris.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, UnbalancedBlobsTest) {
    fs::path filename = "unbalanced_blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_medoids = 3;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMedoidsErrorsTest, PairwiseDistanceMatrixTest) {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    auto pairwise_distance_matrix =
        cpp_clustering::containers::LowerTriangleMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);
}

/*
TEST_F(KMedoidsErrorsTest, MnistTrainTest) {
    fs::path filename = "mnist_train.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    std::cout << data.size() << " | " << labels.size() << " | " << n_features << std::endl;
    // const auto n_medoids = 10;
    const std::size_t n_medoids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = fit_predict_orig(
        data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}
*/

TEST_F(KMedoidsErrorsTest, MnistSimpleFitTest) {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    std::cout << data.size() << " | " << labels.size() << " | " << n_features << std::endl;
    const auto n_medoids = 10;

    simple_fit(data.begin(), data.end(), labels.begin(), labels.end(), n_medoids, n_features, 100);
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
    const auto centroids = common::utils::init_uniform(data.begin(), data.end(), n_centroids, n_features);

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
