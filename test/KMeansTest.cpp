#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "cpp_clustering/kmeans/Hamerly.hpp"
#include "cpp_clustering/kmeans/Lloyd.hpp"

#include "cpp_clustering/kmeans/KMeans.hpp"
#include "cpp_clustering/math/random/VosesAliasMethod.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class KMeansErrorsTest : public ::testing::Test {
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
    std::pair<std::vector<std::size_t>, std::vector<typename InputsIterator::value_type>> simple_fit(
        const InputsIterator& inputs_first,
        const InputsIterator& inputs_last,
        LabelsIterator        labels_first,
        LabelsIterator        labels_last,
        std::size_t           n_centroids,
        std::size_t           n_features,
        std::size_t           n_iterations = 1) {
        using KMeans = cpp_clustering::KMeans<dType>;
        // using PAM = cpp_clustering::FasterMSC;

        auto kmeans = KMeans(n_centroids, n_features);

        kmeans.set_options(
            /*KMeans options=*/KMeans::Options()
                .max_iter(n_iterations)
                .early_stopping(false)
                .tolerance(0.001)
                .patience(0)
                .n_init(10));

        const auto centroids = kmeans.fit<cpp_clustering::Hamerly>(
            inputs_first, inputs_last, cpp_clustering::kmeansplusplus::make_centroids<InputsIterator>);

        const auto predictions = kmeans.predict(inputs_first, inputs_last);

        return {predictions, centroids};
    }

    static constexpr std::size_t n_iterations_global = 100;
    static constexpr std::size_t n_centroids_global  = 4;

    const fs::path folder_root        = fs::path("../datasets/clustering");
    const fs::path inputs_folder      = folder_root / fs::path("inputs");
    const fs::path targets_folder     = folder_root / fs::path("targets");
    const fs::path predictions_folder = folder_root / fs::path("predictions");
    const fs::path centroids_folder   = folder_root / fs::path("centroids");
};

TEST_F(KMeansErrorsTest, NoisyCirclesTest) {
    fs::path filename = "noisy_circles.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 2;  // 2
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, NoisyMoonsTest) {
    fs::path filename = "noisy_moons.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 2;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, VariedTest) {
    fs::path filename = "varied.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, AnisoTest) {
    fs::path filename = "aniso.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, BlobsTest) {
    fs::path filename = "blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    const auto n_centroids = 4;
    // const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, IrisTest) {
    fs::path filename = "iris.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, UnbalancedBlobsTest) {
    fs::path filename = "unbalanced_blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

TEST_F(KMeansErrorsTest, MnistSimpleFitTest) {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder / filename);

    std::cout << data.size() << " | " << labels.size() << " | " << n_features << std::endl;
    const auto n_centroids = 10;

    const auto [predictions, centroids] = simple_fit(
        data.begin(), data.end(), labels.begin(), labels.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder / fs::path(filename));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
