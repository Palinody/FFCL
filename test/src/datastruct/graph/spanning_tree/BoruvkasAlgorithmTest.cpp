#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class BoruvkasAlgorithmErrorsTest : public ::testing::Test {
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

    template <typename MinimumSpanningTreeType>
    void write_minimum_spanning_tree(const MinimumSpanningTreeType& mst, const fs::path& filename) {
        const auto parent_path = filename.parent_path();

        const std::size_t n_features = 3;

        make_directories(parent_path);

        std::ofstream filestream(filename);

        if (filestream.is_open()) {
            std::size_t iter{};

            for (const auto& elem : mst) {
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

    template <typename MinimumSpanningTreeType>
    void write_mst(const MinimumSpanningTreeType& mst, const fs::path& filename) {
        std::ofstream filestream(filename);

        if (!filestream.is_open()) {
            std::cerr << "Error: Failed to open the file for writing." << std::endl;
            return;
        }
        for (const auto& [vertex_1, vertex2, distance] : mst) {
            // rite the elements to the file separated by spaces
            filestream << vertex_1 << " " << vertex2 << " " << distance << std::endl;
        }
        filestream.close();
    }

    static constexpr std::size_t n_iterations_global = 100;
    static constexpr std::size_t n_centroids_global  = 4;

    const fs::path folder_root_        = fs::path("../bin/clustering");
    const fs::path inputs_folder_      = folder_root_ / fs::path("inputs");
    const fs::path targets_folder_     = folder_root_ / fs::path("targets");
    const fs::path predictions_folder_ = folder_root_ / fs::path("predictions");
};

std::vector<std::size_t> generate_indices(std::size_t n_samples) {
    std::vector<std::size_t> elements(n_samples);
    std::iota(elements.begin(), elements.end(), static_cast<std::size_t>(0));
    return elements;
}

TEST_F(BoruvkasAlgorithmErrorsTest, NoisyCirclesTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, NoisyMoonsTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "noisy_moons.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, VariedTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, AnisoTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "aniso.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, BlobsTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, NoStructureTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "no_structure.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, UnbalancedBlobsTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "unbalanced_blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    const auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    timer.print_elapsed_seconds(9);

    write_mst(minimum_spanning_tree, predictions_folder_ / fs::path(filename));
}

TEST_F(BoruvkasAlgorithmErrorsTest, ForestPartitionTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    auto              data       = std::vector<float>({1, 2, 1.5, 2.2, 2.5, 2.9, 2, 3, 4, 2, 3, 3, 3.5, 2.2, 2.3, 2});
    const std::size_t n_features = 2;
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    timer.reset();

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType())
                                   .splitting_rule_policy(SplittingRulePolicyType()));

    timer.print_elapsed_seconds(9);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<IndexerType>();

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(6));

    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    std::cout << "MST size: " << minimum_spanning_tree.size() << "\n";

    ffcl::datastruct::mst::print(ffcl::datastruct::mst::sort(std::move(minimum_spanning_tree)));

    timer.print_elapsed_seconds(9);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
