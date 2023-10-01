#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"
#include "ffcl/datastruct/kdtree/KDTree.hpp"
#include "ffcl/dbscan/DBSCAN.hpp"

#include <sys/types.h>  // std::ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace fs = std::filesystem;

class DBSCANErrorsTest : public ::testing::Test {
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

template <typename Type>
void print_data(const std::vector<Type>& data, std::size_t n_features) {
    if (!n_features) {
        return;
    }
    const std::size_t n_samples = data.size() / n_features;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            std::cout << data[sample_index * n_features + feature_index] << " ";
        }
        std::cout << "\n";
    }
}

template <typename Indexer, typename IndexerFunction, typename... Args>
auto get_neighbors(std::size_t query_index, const Indexer& indexer, IndexerFunction&& func, Args&&... args) {
    using LabelType = ssize_t;

    const std::size_t n_samples = indexer.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    LabelType cluster_label = static_cast<LabelType>(0);

    auto query_function = [&indexer = static_cast<const Indexer&>(indexer), func = std::forward<IndexerFunction>(func)](
                              std::size_t index, auto&&... funcArgs) mutable {
        return std::invoke(func, indexer, index, std::forward<decltype(funcArgs)>(funcArgs)...);
    };

    // the indices of the neighbors in the global dataset with their corresponding distances
    // the query sample is not included
    auto initial_neighbors_buffer = query_function(query_index, std::forward<Args>(args)...);

    ++cluster_label;

    predictions[query_index] = cluster_label;

    auto initial_neighbors_indices = initial_neighbors_buffer.extract_indices();

    // iterate over the samples that are assigned to the current cluster
    for (std::size_t cluster_sample_index = 0; cluster_sample_index < initial_neighbors_indices.size();
         ++cluster_sample_index) {
        const auto neighbor_index   = initial_neighbors_indices[cluster_sample_index];
        predictions[neighbor_index] = cluster_label;
    }

    predictions[query_index] = ++cluster_label;
    return predictions;
}

// /*
TEST_F(DBSCANErrorsTest, NoisyCirclesTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "noisy_circles.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(2).min_samples(5));

    timer.reset();

    const dType radius = 2;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(DBSCANErrorsTest, NoisyMoonsTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "noisy_moons.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(1).min_samples(5));

    timer.reset();

    const dType radius = 1;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(DBSCANErrorsTest, VariedTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "varied.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(1).min_samples(3));

    timer.reset();

    const dType radius = 1;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(DBSCANErrorsTest, AnisoTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "aniso.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(1.2).min_samples(10));

    timer.reset();

    const dType radius = 1.2;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(DBSCANErrorsTest, BlobsTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(1).min_samples(10));

    timer.reset();

    const dType radius = 1;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(DBSCANErrorsTest, NoStructureTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "no_structure.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(1).min_samples(5));

    timer.reset();

    const dType radius = 1;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(DBSCANErrorsTest, UnbalancedBlobsTest) {
    common::timer::Timer<common::timer::Nanoseconds> timer;

    fs::path filename = "unbalanced_blobs.txt";

    auto              data       = load_data<dType>(inputs_folder_ / filename, ' ');
    const std::size_t n_features = get_num_features_in_file(inputs_folder_ / filename);
    const std::size_t n_samples  = common::utils::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator         = decltype(indices)::iterator;
    using SamplesIterator         = decltype(data)::iterator;
    using IndexerType             = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType             = typename IndexerType::Options;
    using AxisSelectionPolicyType = kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType = kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

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

    auto dbscan = ffcl::DBSCAN<IndexerType>();

    dbscan.set_options(ffcl::DBSCAN<IndexerType>::Options().radius(2).min_samples(5));

    timer.reset();

    const dType radius = 2;
    // /*
    const auto predictions = dbscan.predict(indexer, &IndexerType::radius_search_around_query_index, radius);
    // */
    /*
    const float edge        = std::sqrt(2.0) * radius;
    const auto  predictions = dbscan.predict(indexer,
                                            &IndexerType::range_search_around_query_index,
                                            ffcl::bbox::HyperRangeType<SamplesIterator>({{-edge, edge}, {-edge,
    edge}}));
    */
    timer.print_elapsed_seconds(9);

    write_data<std::size_t>(predictions, 1, predictions_folder_ / fs::path(filename));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
