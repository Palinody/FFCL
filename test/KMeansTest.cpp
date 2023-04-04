#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/kmeans/Hamerly.hpp"
#include "ffcl/kmeans/KMeans.hpp"
#include "ffcl/kmeans/Lloyd.hpp"
#include "ffcl/math/heuristics/SilhouetteMethod.hpp"
#include "ffcl/math/random/VosesAliasMethod.hpp"
#include "ffcl/math/statistics/Statistics.hpp"

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
    std::pair<std::vector<std::size_t>, std::vector<typename InputsIterator::value_type>> simple_fit(
        const InputsIterator& inputs_first,
        const InputsIterator& inputs_last,
        std::size_t           n_centroids,
        std::size_t           n_features,
        std::size_t           n_iterations = 1) {
        using KMeans = ffcl::KMeans<dType>;
        // using PAM = ffcl::FasterMSC;

        auto kmeans = KMeans(n_centroids, n_features);

        kmeans.set_options(
            /*KMeans options=*/KMeans::Options()
                .max_iter(n_iterations)
                .early_stopping(true)
                .tolerance(0.001)
                .patience(0)
                .n_init(10));

        const auto centroids =
            kmeans.fit<ffcl::Hamerly>(inputs_first, inputs_last, ffcl::kmeansplusplus::make_centroids<InputsIterator>);

        const auto predictions = kmeans.predict(inputs_first, inputs_last);

        return {predictions, centroids};
    }

    static constexpr std::size_t n_iterations_global = 100;
    static constexpr std::size_t n_centroids_global  = 4;

    const fs::path folder_root_        = fs::path("../bin/clustering");
    const fs::path inputs_folder_      = folder_root_ / fs::path("inputs");
    const fs::path targets_folder_     = folder_root_ / fs::path("targets");
    const fs::path predictions_folder_ = folder_root_ / fs::path("predictions");
    const fs::path centroids_folder_   = folder_root_ / fs::path("centroids");
};

TEST_F(KMeansErrorsTest, SilhouetteTest) {
    using KMeans = ffcl::KMeans<dType>;

    fs::path filename = "unbalanced_blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    std::size_t k_min = 2;
    std::size_t k_max = 10;

    std::vector<dType> scores(k_max - k_min);

    // range n_centroids/n_medoids in [2, 10[
    for (std::size_t k = k_min; k < k_max; ++k) {
        // use any clustering algorithm that better suits your use case
        KMeans kmeans(k, n_features);
        // fit the centroids (or medoids if it was KMedoids)
        kmeans.fit(data.begin(), data.end());
        // map the samples to their closest centroid/medoid
        const auto predictions = kmeans.predict(data.begin(), data.end());
        // compute the silhouette scores for each sample
        const auto samples_silhouette_values =
            math::heuristics::silhouette(data.begin(), data.end(), predictions.begin(), predictions.end(), n_features);
        // get the average score
        const auto mean_silhouette_coefficient = math::heuristics::get_mean_silhouette_coefficient(
            samples_silhouette_values.begin(), samples_silhouette_values.end());
        // accumulate the current scores
        scores[k - k_min] = mean_silhouette_coefficient;
    }
    // find the k corresponding to the number of centroids/medoids k with the best average silhouette score
    const auto best_k = k_min + math::statistics::argmax(scores.begin(), scores.end());

    std::cout << "best k: " << best_k << "\n";
}

TEST_F(KMeansErrorsTest, NoisyCirclesTest) {
    fs::path filename = "noisy_circles.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 2;  // 2
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, NoisyMoonsTest) {
    fs::path filename = "noisy_moons.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 2;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());  // 2

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, VariedTest) {
    fs::path filename = "varied.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, AnisoTest) {
    fs::path filename = "aniso.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, BlobsTest) {
    fs::path filename = "blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    const auto n_centroids = 4;
    // const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, IrisTest) {
    fs::path filename = "iris.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, UnbalancedBlobsTest) {
    fs::path filename = "unbalanced_blobs.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    // const auto n_centroids = 3;
    const std::size_t n_centroids = 1 + *std::max_element(labels.begin(), labels.end());

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

TEST_F(KMeansErrorsTest, MnistSimpleFitTest) {
    fs::path filename = "mnist.txt";

    const auto        data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const auto        labels     = load_data<std::size_t>(targets_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);

    std::cout << data.size() << " | " << labels.size() << " | " << n_features << std::endl;
    const auto n_centroids = 10;

    const auto [predictions, centroids] =
        simple_fit(data.begin(), data.end(), n_centroids, n_features, n_iterations_global);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
    write_data<dType>(centroids, 1, centroids_folder_ / fs::path(filename));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
