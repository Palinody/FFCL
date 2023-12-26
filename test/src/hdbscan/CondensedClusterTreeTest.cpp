#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/datastruct/single_linkage_cluster_tree/SingleLinkageClusterTree.hpp"
#include "ffcl/datastruct/spanning_tree/BoruvkasAlgorithm.hpp"
#include "ffcl/hdbscan/CondensedClusterTree.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class CondensedClusterTreeErrorsTest : public ::testing::Test {
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

/*
TEST_F(CondensedClusterTreeErrorsTest, MainTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    using IndexType = int;
    using ValueType = float;

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree =
        ffcl::mst::MinimumSpanningTree<IndexType, ValueType>{ ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{},
                                                              ffcl::mst::Edge<IndexType, ValueType>{}};

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);
}
*/

TEST_F(CondensedClusterTreeErrorsTest, NoisyCirclesTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(CondensedClusterTreeErrorsTest, NoisyMoonsTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "noisy_moons.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(CondensedClusterTreeErrorsTest, VariedTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(CondensedClusterTreeErrorsTest, AnisoTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "aniso.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(CondensedClusterTreeErrorsTest, BlobsTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(CondensedClusterTreeErrorsTest, NoStructureTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "no_structure.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(CondensedClusterTreeErrorsTest, UnbalancedBlobsTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "unbalanced_blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndexType       = std::size_t /*decltype(indices[0])*/;
    using ValueType       = dType /*decltype(data[0])*/;
    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    printf("Indexer build:\n");
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

    boruvkas_algorithm.set_options(ffcl::BoruvkasAlgorithm<IndexerType>::Options().k_nearest_neighbors(10));

    printf("Boruvka's MST build:\n");
    timer.reset();

    auto minimum_spanning_tree = boruvkas_algorithm.make_tree(std::move(indexer));

    timer.print_elapsed_seconds(9);

    printf("SingleLinkageClusterTree build:\n");
    timer.reset();

    ffcl::SingleLinkageClusterTree single_linkage_cluster_tree(std::move(minimum_spanning_tree));

    timer.print_elapsed_seconds(9);

    auto single_linkage_cluster_tree_root = single_linkage_cluster_tree.root();

    printf("CondensedClusterTree build:\n");
    timer.reset();

    ffcl::CondensedClusterTree<IndexType, ValueType> condensed_cluster_tree(
        single_linkage_cluster_tree_root,
        ffcl::CondensedClusterTree<IndexType, ValueType>::Options()
            .min_cluster_size(15)
            .return_leaf_nodes(false)
            .allow_single_cluster(true));

    timer.print_elapsed_seconds(9);

    printf("CondensedClusterTree prediction:\n");
    timer.reset();

    const auto predictions = condensed_cluster_tree.predict();

    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
