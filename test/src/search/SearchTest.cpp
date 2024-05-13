#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Timer.hpp"
#include "ffcl/common/Utils.hpp"

#include "ffcl/common/math/random/Distributions.hpp"

#include "ffcl/datastruct/tree/kdtree/KDTree.hpp"

#include "ffcl/search/Search.hpp"
#include "ffcl/search/buffer/Unsorted.hpp"

#include "ffcl/datastruct/bounds/segment/CenterAndLength.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndLength.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

#include "ffcl/datastruct/UnionFind.hpp"
#include "ffcl/datastruct/bounds/Ball.hpp"
#include "ffcl/datastruct/bounds/BoundingBox.hpp"
#include "ffcl/datastruct/bounds/UnboundedBall.hpp"

#include "ffcl/search/ClosestPairOfSamples.hpp"

#include <sys/types.h>  // ssize_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#define TIME_IT true
#define ASSERT_IT true

namespace fs = std::filesystem;

class SearcherErrorsTest : public ::testing::Test {
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

    const fs::path kdtree_folder_root_ = fs::path("../bin/kdtree");
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

TEST_F(SearcherErrorsTest, NoStructureTest) {
    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType{})
                                   .splitting_rule_policy(SplittingRulePolicyType{}));

    // using SegmentType = ffcl::datastruct::bounds::segment::LowerBoundAndUpperBound<ValueType>;
    // using BoundType   = ffcl::datastruct::bounds::BoundingBox<SegmentType>;
    // using BoundType = ffcl::datastruct::bounds::Ball<ValueType, 2>;
    // using BoundType = ffcl::datastruct::bounds::UnboundedBall<ValueType, 2>;
    // using BoundType = ffcl::datastruct::bounds::UnboundedBallView<SamplesIterator>;
    // using BoundType = ffcl::datastruct::bounds::BallView<SamplesIterator>;
    // using BoundType = ffcl::datastruct::bounds::BoundingBoxView<SamplesIterator>;

    // using BufferType = ffcl::search::buffer::Unsorted<SamplesIterator, BoundType>;

    // auto bound_ptr = std::make_shared<BoundType>(BoundType({{-15, -10}, {-15, -3}}));
    // auto bound_ptr = std::make_shared<BoundType>(BoundType{{-10, -10}, 10});
    // auto bound_ptr = std::make_shared<BoundType>(BoundType{{-10, -10}});

    auto center_point_query = std::vector<ValueType>{-10, -10};
    // const ValueType radius_query = 5;
    auto lengths_from_center_point_query = std::vector<ValueType>{2, 6};

    // auto bound_query = BoundType(center_point_query.begin(), center_point_query.end(), radius_query);
    auto bound_query = ffcl::datastruct::bounds::BoundingBoxView(
        /**/ center_point_query.begin(),
        /**/ center_point_query.end(),
        /**/ lengths_from_center_point_query);

    const IndexType max_capacity = ffcl::common::infinity<IndexType>();

    auto bounded_buffer_query = ffcl::search::buffer::Unsorted(std::move(bound_query), /*max_capacity=*/max_capacity);
    // auto bounded_buffer_query =
    // BufferType(center_point_query.begin(), center_point_query.end(), /*max_capacity=*/max_capacity);

    auto searcher = ffcl::search::Searcher(std::move(indexer));

    const auto returned_indices = searcher(std::move(bounded_buffer_query)).indices();

    auto predictions = std::vector<IndexType>(n_samples);

    for (const auto& index : returned_indices) {
        predictions[index] = 1;
    }
    write_data<IndexType>(predictions, 1, predictions_folder_ / fs::path(filename));
}

TEST_F(SearcherErrorsTest, NoStructureBenchmarkTest) {
    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto indexer = IndexerType(indices.begin(),
                               indices.end(),
                               data.begin(),
                               data.end(),
                               n_features,
                               OptionsType()
                                   .bucket_size(std::sqrt(n_samples))
                                   .max_depth(std::log2(n_samples))
                                   .axis_selection_policy(AxisSelectionPolicyType{})
                                   .splitting_rule_policy(SplittingRulePolicyType{}));

    // using BufferType = ffcl::search::buffer::Unsorted<SamplesIterator>;

    auto searcher = ffcl::search::Searcher(std::move(indexer));

    constexpr IndexType max_capacity = 10;  // ffcl::common::infinity<IndexType>();

    std::size_t returned_indices_counter = 0;

    for (std::size_t sample_index_query = 0; sample_index_query < n_samples; ++sample_index_query) {
        auto bounded_buffer_query =
            ffcl::search::buffer::Unsorted(data.begin() + sample_index_query * n_features,
                                           data.begin() + sample_index_query * n_features + n_features,
                                           max_capacity);

        const auto returned_indices = searcher(std::move(bounded_buffer_query)).indices();

        returned_indices_counter += returned_indices.size();
    }
    std::cout << "returned_indices_counter: " << returned_indices_counter << "\n";
}

template <typename IndicesIterator>
void shuffle_indices(IndicesIterator indices_first, IndicesIterator indices_last) {
    std::shuffle(indices_first, indices_last, std::mt19937{std::random_device{}()});
}

/*
TEST_F(SearcherErrorsTest, DualTreeClosestPairTest) {
    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    auto uniform_random_number_generator =
        ffcl::common::math::random::uniform_distribution<std::size_t>(1, n_samples - 1);

    for (std::size_t test_index = 0; test_index < 10; ++test_index) {
        shuffle_indices(indices.begin(), indices.end());

        const std::size_t n_queries = uniform_random_number_generator();

        std::cout << "n_queries = " << n_queries << "\n";

        auto reference_indices = std::vector(indices.begin(), indices.begin() + n_queries);
        auto query_indices     = std::vector(indices.begin() + n_queries, indices.end());

        // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
        auto reference_indexer = IndexerType(reference_indices.begin(),
                                             reference_indices.end(),
                                             data.begin(),
                                             data.end(),
                                             n_features,
                                             OptionsType()
                                                 .bucket_size(1)
                                                 .max_depth(n_samples)
                                                 .axis_selection_policy(AxisSelectionPolicyType{})
                                                 .splitting_rule_policy(SplittingRulePolicyType{}));

        auto reference_indexer_copy = reference_indexer;

        auto searcher = ffcl::search::Searcher(std::move(reference_indexer));

        // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
        auto query_indexer = IndexerType(query_indices.begin(),
                                         query_indices.end(),
                                         data.begin(),
                                         data.end(),
                                         n_features,
                                         OptionsType()
                                             .bucket_size(1)
                                             .max_depth(n_samples)
                                             .axis_selection_policy(AxisSelectionPolicyType{})
                                             .splitting_rule_policy(SplittingRulePolicyType{}));

        auto query_indexer_copy = query_indexer;

        const auto shortest_edge = searcher.dual_tree_shortest_edge(std::move(query_indexer), 1);

        {
            const auto brute_force_shortest_edge =
                ffcl::search::algorithms::dual_set_shortest_edge(query_indices.begin(),
                                                                 query_indices.end(),
                                                                 data.begin(),
                                                                 data.end(),
                                                                 n_features,
                                                                 reference_indices.begin(),
                                                                 reference_indices.end(),
                                                                 data.begin(),
                                                                 data.end(),
                                                                 n_features,
                                                                 1);

            std::cout << "n_queries: " << n_queries << "\n";

            if (ffcl::common::inequality(std::get<0>(brute_force_shortest_edge), std::get<0>(shortest_edge))) {
                std::cout << "\t>>> " << std::get<0>(brute_force_shortest_edge) << " != " << std::get<0>(shortest_edge)
                          << "\n";
            } else {
                std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << "\n";
            }

            if (ffcl::common::inequality(std::get<1>(brute_force_shortest_edge), std::get<1>(shortest_edge))) {
                std::cout << "\t>>> " << std::get<1>(brute_force_shortest_edge) << " != " << std::get<1>(shortest_edge)
                          << "\n";
            } else {
                std::cout << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << "\n";
            }

            if (ffcl::common::inequality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge))) {
                std::cout << "\t>>> " << std::get<2>(brute_force_shortest_edge) << " != " << std::get<2>(shortest_edge)
                          << "\n";
            } else {
                std::cout << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";
            }
            std::cout << "---\n";
        }

        auto labels = std::vector<IndexType>(n_samples);

        for (const auto& query_index : query_indices) {
            labels[query_index] = 0;
        }

        for (const auto& reference_index : reference_indices) {
            labels[reference_index] = 1;
        }
        labels[std::get<0>(shortest_edge)] = 2;
        labels[std::get<1>(shortest_edge)] = 3;

        write_data<IndexType>(labels, 1, predictions_folder_ / fs::path(filename));

        reference_indexer_copy.serialize(kdtree_folder_root_ / fs::path(filename.stem().string() + "_reference.json"));
        query_indexer_copy.serialize(kdtree_folder_root_ / fs::path(filename.stem().string() + "_query.json"));
    }
}
*/

// /*
TEST_F(SearcherErrorsTest, DualTreeClosestPairLoopTimerTest) {
#if defined(TIME_IT) && TIME_IT
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;
#endif

    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    ValueType dummy_acc = 0;

#if defined(TIME_IT) && TIME_IT
    static constexpr std::uint8_t n_decimals = 9;
    timer.reset();
    auto total_elapsed_time = timer.elapsed();

#endif

    const std::size_t increment = 1;  // std::max(std::size_t{1}, n_samples / 100);

    for (std::size_t split_index = 1; split_index < n_samples; split_index += increment) {
        shuffle_indices(indices.begin(), indices.end());

        auto reference_indices = std::vector(indices.begin(), indices.begin() + split_index);
        auto query_indices     = std::vector(indices.begin() + split_index, indices.end());

        // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
        auto reference_indexer = IndexerType(reference_indices.begin(),
                                             reference_indices.end(),
                                             data.begin(),
                                             data.end(),
                                             n_features,
                                             OptionsType()
                                                 .bucket_size(0)
                                                 .max_depth(n_samples)
                                                 .axis_selection_policy(AxisSelectionPolicyType{})
                                                 .splitting_rule_policy(SplittingRulePolicyType{}));

        auto searcher = ffcl::search::Searcher(std::move(reference_indexer));

#if defined(TIME_IT) && TIME_IT
        timer.reset();
#endif

        // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
        auto query_indexer = IndexerType(query_indices.begin(),
                                         query_indices.end(),
                                         data.begin(),
                                         data.end(),
                                         n_features,
                                         OptionsType()
                                             .bucket_size(0)
                                             .max_depth(n_samples)
                                             .axis_selection_policy(AxisSelectionPolicyType{})
                                             .splitting_rule_policy(SplittingRulePolicyType{}));

        // auto      union_find             = ffcl::datastruct::UnionFind<IndexType>(n_samples);
        // IndexType queries_representative = union_find.find(query_indices[0]);
        // for (const auto& query_index : query_indices) {
        // queries_representative = union_find.merge(queries_representative, query_index);
        // }
        // const auto shortest_edge =
        // searcher.dual_tree_shortest_edge(std::move(query_indexer), union_find, queries_representative, 1);

        const auto shortest_edge = searcher.dual_tree_shortest_edge(std::move(query_indexer), 1);

#if defined(TIME_IT) && TIME_IT
        {
            const auto elapsed_time = timer.elapsed();
            total_elapsed_time += elapsed_time;
            // printf("[%ld/%ld]: %.5f\n", split_index, n_samples, std::get<2>(shortest_edge));

            if (split_index == 1) {
                printf("times_array = [");

            } else if (split_index >= n_samples - increment) {
                printf("%.*f]\n", n_decimals, (elapsed_time * 1e-9f));

            } else {
                printf("%.*f, ", n_decimals, (elapsed_time * 1e-9f));
            }
        }
#endif

#if defined(ASSERT_IT) && ASSERT_IT
        {
#if defined(TIME_IT) && TIME_IT
            timer.reset();
#endif

            const auto brute_force_shortest_edge =
                ffcl::search::algorithms::dual_set_shortest_edge(query_indices.begin(),
                                                                 query_indices.end(),
                                                                 data.begin(),
                                                                 data.end(),
                                                                 n_features,
                                                                 reference_indices.begin(),
                                                                 reference_indices.end(),
                                                                 data.begin(),
                                                                 data.end(),
                                                                 n_features,
                                                                 1);

            // const auto brute_force_shortest_edge =
            //     ffcl::search::algorithms::dual_set_shortest_edge(query_indices.begin(),
            //                                                      query_indices.end(),
            //                                                      data.begin(),
            //                                                      data.end(),
            //                                                      n_features,
            //                                                      reference_indices.begin(),
            //                                                      reference_indices.end(),
            //                                                      data.begin(),
            //                                                      data.end(),
            //                                                      n_features,
            //                                                      union_find,
            //                                                      queries_representative,
            //                                                      1);

#if defined(TIME_IT) && TIME_IT
            const auto elapsed_time = timer.elapsed();

            printf("brute force time: %.*f\n", n_decimals, (elapsed_time * 1e-9f));

#endif

            std::cout << "split_index: " << split_index << "\n";

            if (ffcl::common::inequality(std::get<0>(brute_force_shortest_edge), std::get<0>(shortest_edge))) {
                std::cout << "\t>>> " << std::get<0>(brute_force_shortest_edge) << " != " << std::get<0>(shortest_edge)
                          << "\n";
            } else {
                std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << "\n";
            }

            if (ffcl::common::inequality(std::get<1>(brute_force_shortest_edge), std::get<1>(shortest_edge))) {
                std::cout << "\t>>> " << std::get<1>(brute_force_shortest_edge) << " != " << std::get<1>(shortest_edge)
                          << "\n";
            } else {
                std::cout << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << "\n";
            }

            if (ffcl::common::inequality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge))) {
                std::cout << "\t>>> " << std::get<2>(brute_force_shortest_edge) << " != " << std::get<2>(shortest_edge)
                          << "\n";
            } else {
                std::cout << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";
            }
            std::cout << "---\n";

            // std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << " && "
            //   << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << " && "
            //   << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";

            // ASSERT_TRUE(ffcl::common::equality(std::get<0>(brute_force_shortest_edge), std::get<0>(shortest_edge)));
            // ASSERT_TRUE(ffcl::common::equality(std::get<1>(brute_force_shortest_edge), std::get<1>(shortest_edge)));
            // ASSERT_TRUE(ffcl::common::equality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge)));
        }
#endif
        dummy_acc += std::get<2>(shortest_edge);
    }
#if defined(TIME_IT) && TIME_IT
    printf("Total runtime: %.*f\n", n_decimals, (total_elapsed_time * 1e-9f));
#endif
    std::cout << dummy_acc << "\n";
}
// */

// TEST_F(SearcherErrorsTest, DualTreeClosestPairWithUnionFindLoopTimerTest) {
// #if defined(TIME_IT) && TIME_IT
//     ffcl::common::Timer<ffcl::common::Nanoseconds> timer;
// #endif

//     fs::path filename = "no_structure.txt";

//     using IndexType = std::size_t;
//     using ValueType = dType;

//     auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
//     const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
//     const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

//     auto indices = generate_indices(n_samples);

//     using IndicesIterator = decltype(indices)::iterator;
//     using SamplesIterator = decltype(data)::iterator;
//     using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
//     using OptionsType     = IndexerType::Options;
//     using AxisSelectionPolicyType =
//         ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
//     using SplittingRulePolicyType =
//         ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

//     ValueType dummy_acc = 0;

// #if defined(TIME_IT) && TIME_IT
//     static constexpr std::uint8_t n_decimals = 9;
//     timer.reset();
//     auto total_elapsed_time = timer.elapsed();
// #endif

//     const std::size_t increment = std::max(std::size_t{1}, n_samples / 100);

//     for (std::size_t split_index = 1; split_index < n_samples; split_index += increment) {
//         shuffle_indices(indices.begin(), indices.end());

//         // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
//         auto reference_indexer = IndexerType(indices.begin(),
//                                              indices.end(),
//                                              data.begin(),
//                                              data.end(),
//                                              n_features,
//                                              OptionsType()
//                                                  .bucket_size(40)
//                                                  .max_depth(n_samples)
//                                                  .axis_selection_policy(AxisSelectionPolicyType{})
//                                                  .splitting_rule_policy(SplittingRulePolicyType{}));

//         auto searcher = ffcl::search::Searcher(std::move(reference_indexer));

//         auto union_find = ffcl::datastruct::UnionFind<IndexType>(n_samples);

//         auto queries_indices = std::vector(indices.begin(), indices.begin() + split_index);

//         IndexType queries_representative = union_find.find(queries_indices[0]);

//         // merge all the indices in the queries_indices into the same component
//         for (const auto& query_index : queries_indices) {
//             queries_representative = union_find.merge(queries_representative, query_index);
//         }
// #if defined(TIME_IT) && TIME_IT
//         timer.reset();
// #endif
//         // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
//         auto query_indexer = IndexerType(queries_indices.begin(),
//                                          queries_indices.end(),
//                                          data.begin(),
//                                          data.end(),
//                                          n_features,
//                                          OptionsType()
//                                              .bucket_size(40)
//                                              .axis_selection_policy(AxisSelectionPolicyType{})
//                                              .splitting_rule_policy(SplittingRulePolicyType{}));

// #if defined(TIME_IT) && TIME_IT
//         {
//             const auto elapsed_time = timer.elapsed();

//             printf("Queries tree build: %.*f\n", n_decimals, (elapsed_time * 1e-9f));
//         }
// #endif

// #if defined(TIME_IT) && TIME_IT
//         timer.reset();
// #endif
//         const auto shortest_edge = searcher.dual_tree_shortest_edge(/**/ std::move(query_indexer),
//                                                                     /**/ union_find,
//                                                                     /**/ queries_representative,
//                                                                     /**/ 1);

// #if defined(TIME_IT) && TIME_IT
//         {
//             const auto elapsed_time = timer.elapsed();
//             total_elapsed_time += elapsed_time;
//             // printf("[%ld/%ld]: %.5f\n", split_index, n_samples, std::get<2>(shortest_edge));

//             if (split_index == 1) {
//                 printf("times_array = [");

//             } else if (split_index >= n_samples - increment) {
//                 printf("%.*f]\n", n_decimals, (elapsed_time * 1e-9f));

//             } else {
//                 printf("%.*f, ", n_decimals, (elapsed_time * 1e-9f));
//             }
//         }
// #endif

// #if defined(ASSERT_IT) && ASSERT_IT
//         {
// #if defined(TIME_IT) && TIME_IT
//             timer.reset();
// #endif
//             const auto brute_force_shortest_edge =
//                 ffcl::search::algorithms::dual_set_shortest_edge(queries_indices.begin(),
//                                                                  queries_indices.end(),
//                                                                  data.begin(),
//                                                                  data.end(),
//                                                                  n_features,
//                                                                  indices.begin(),
//                                                                  indices.end(),
//                                                                  data.begin(),
//                                                                  data.end(),
//                                                                  n_features,
//                                                                  /**/ union_find,
//                                                                  /**/ queries_representative,
//                                                                  /**/ 1);

// #if defined(TIME_IT) && TIME_IT
//             const auto elapsed_time = timer.elapsed();

//             printf("brute force time: %.*f\n", n_decimals, (elapsed_time * 1e-9f));

// #endif

//             std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << " && "
//                       << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << " && "
//                       << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";

//             ASSERT_TRUE(ffcl::common::equality(std::get<0>(brute_force_shortest_edge), std::get<0>(shortest_edge)));
//             ASSERT_TRUE(ffcl::common::equality(std::get<1>(brute_force_shortest_edge), std::get<1>(shortest_edge)));
//             ASSERT_TRUE(ffcl::common::equality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge)));
//         }
// #endif
//     }
// #if defined(TIME_IT) && TIME_IT
//     printf("Total runtime: %.*f\n", n_decimals, (total_elapsed_time * 1e-9f));
// #endif
//     std::cout << dummy_acc << "\n";
// }

/*
TEST_F(SearcherErrorsTest, DualTreeClosestPairLoopTest) {
    ffcl::common::Timer<ffcl::common::Nanoseconds> timer;

    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    ValueType   dummy_acc           = 0;
    std::size_t wrong_value_counter = 0;

    static constexpr std::uint8_t n_decimals = 9;

    const std::size_t increment = std::max(std::size_t{1}, n_samples / 100);

    for (std::size_t split_index = 1; split_index < n_samples; split_index += increment) {
        shuffle_indices(indices.begin(), indices.end());

        timer.reset();

        // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
        auto query_indexer = IndexerType(indices.begin(),
                                         indices.begin() + split_index,
                                         data.begin(),
                                         data.end(),
                                         n_features,
                                         OptionsType()
                                             .bucket_size(40)
                                             .max_depth(n_samples)
                                             .axis_selection_policy(AxisSelectionPolicyType{})
                                             .splitting_rule_policy(SplittingRulePolicyType{}));

        // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
        auto reference_indexer = IndexerType(indices.begin() + split_index,
                                             indices.end(),
                                             data.begin(),
                                             data.end(),
                                             n_features,
                                             OptionsType()
                                                 .bucket_size(40)
                                                 .max_depth(n_samples)
                                                 .axis_selection_policy(AxisSelectionPolicyType{})
                                                 .splitting_rule_policy(SplittingRulePolicyType{}));

        auto searcher = ffcl::search::Searcher(std::move(reference_indexer));

        const auto shortest_edge = searcher.dual_tree_shortest_edge(std::move(query_indexer), 1);

        printf("dual_tree_shortest_edge time: %.*f\n", n_decimals, (timer.elapsed() * 1e-9f));

        dummy_acc += std::get<2>(shortest_edge);

        // printf("[%ld/%ld]: %.5f\n", split_index, n_samples, std::get<2>(shortest_edge));

        // if (split_index == 1) {
        //     printf("times_array = [");

        // } else if (split_index >= n_samples - increment) {
        //     printf("%.*f] ", n_decimals, (timer.elapsed() * 1e-9f));

        // } else {
        //     printf("%.*f, ", n_decimals, (timer.elapsed() * 1e-9f));
        // }

        timer.reset();

        const auto brute_force_shortest_edge =
            ffcl::search::algorithms::dual_set_shortest_edge(indices.begin(),
                                                             indices.begin() + split_index,
                                                             data.begin(),
                                                             data.end(),
                                                             n_features,
                                                             indices.begin() + split_index,
                                                             indices.end(),
                                                             data.begin(),
                                                             data.end(),
                                                             n_features,
                                                             1);

        printf("brute_force_shortest_edge time: %.*f\n", n_decimals, (timer.elapsed() * 1e-9f));

        if (!ffcl::common::equality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge))) {
            if (std::find(indices.begin(), indices.begin() + split_index, std::get<0>(brute_force_shortest_edge)) ==
                indices.begin() + split_index) {
                printf("brute_force_shortest_edge: %ld\n", std::get<0>(brute_force_shortest_edge));
                ASSERT_TRUE(false);
            }
            if (std::find(indices.begin() + split_index, indices.end(), std::get<1>(brute_force_shortest_edge)) ==
                indices.end()) {
                printf("brute_force_shortest_edge: %ld\n", std::get<1>(brute_force_shortest_edge));
                ASSERT_TRUE(false);
            }
            if (std::find(indices.begin(), indices.begin() + split_index, std::get<0>(shortest_edge)) ==
                indices.begin() + split_index) {
                printf("shortest_edge: %ld\n", std::get<0>(shortest_edge));
                ASSERT_TRUE(false);
            }
            if (std::find(indices.begin() + split_index, indices.end(), std::get<1>(shortest_edge)) == indices.end()) {
                printf("shortest_edge: %ld\n", std::get<1>(shortest_edge));
                ASSERT_TRUE(false);
            }

            std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << " && "
                      << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << " && "
                      << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";

            wrong_value_counter++;
        }

        ASSERT_TRUE(std::get<0>(brute_force_shortest_edge) == std::get<0>(shortest_edge) &&
                    std::get<1>(brute_force_shortest_edge) == std::get<1>(shortest_edge) &&
                    ffcl::common::equality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge)));

        ASSERT_EQ(std::get<1>(brute_force_shortest_edge), std::get<1>(shortest_edge));
        ASSERT_TRUE(ffcl::common::equality(std::get<2>(brute_force_shortest_edge), std::get<2>(shortest_edge)));
    }
    std::cout << "\nwrong_value_counter:" << wrong_value_counter << "/" << n_samples << "\n";

    std::cout << dummy_acc << "\n";
}
*/

/*
TEST_F(SearcherErrorsTest, DualTreeClosestEdgeWithDifferentTreesTest) {
    fs::path filename = "unbalanced_blobs.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    const IndexType n_queries_samples = std::max(std::size_t{1}, n_samples / 10);

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto reference_indexer = IndexerType(indices.begin() + n_queries_samples,
                                         indices.end(),
                                         data.begin(),
                                         data.end(),
                                         n_features,
                                         OptionsType()
                                             .bucket_size(1)
                                             .axis_selection_policy(AxisSelectionPolicyType{})
                                             .splitting_rule_policy(SplittingRulePolicyType{}));

    auto searcher = ffcl::search::Searcher(std::move(reference_indexer));

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto query_indexer = IndexerType(indices.begin(),
                                     indices.begin() + n_queries_samples,
                                     data.begin(),
                                     data.end(),
                                     n_features,
                                     OptionsType()
                                         .bucket_size(1)
                                         .axis_selection_policy(AxisSelectionPolicyType{})
                                         .splitting_rule_policy(SplittingRulePolicyType{}));

    const auto shortest_edge = searcher.dual_tree_shortest_edge(std::move(query_indexer), 1);

    auto labels = std::vector<IndexType>(n_samples);

    for (auto query_index_it = indices.begin(); query_index_it != indices.begin() + n_queries_samples;
         ++query_index_it) {
        labels[*query_index_it] = 0;
    }

    for (auto reference_index_it = indices.begin() + n_queries_samples; reference_index_it != indices.end();
         ++reference_index_it) {
        labels[*reference_index_it] = 1;
    }
    labels[std::get<0>(shortest_edge)] = 2;
    labels[std::get<1>(shortest_edge)] = 3;

    const auto brute_force_shortest_edge =
        ffcl::search::algorithms::dual_set_shortest_edge(indices.begin(),
                                                         indices.begin() + n_queries_samples,
                                                         data.begin(),
                                                         data.end(),
                                                         n_features,
                                                         indices.begin() + n_queries_samples,
                                                         indices.end(),
                                                         data.begin(),
                                                         data.end(),
                                                         n_features,
                                                         1);

    std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << " && "
              << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << " && "
              << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";

    write_data<IndexType>(labels, 1, predictions_folder_ / fs::path(filename));
}
*/

/*
TEST_F(SearcherErrorsTest, DualTreeClosestEdgeWithSameTreesTest) {
    fs::path filename = "no_structure.txt";

    using IndexType = std::size_t;
    using ValueType = dType;

    auto            data       = load_data<ValueType>(inputs_folder_ / filename, ' ');
    const IndexType n_features = get_num_features_in_file(inputs_folder_ / filename);
    const IndexType n_samples  = ffcl::common::get_n_samples(data.begin(), data.end(), n_features);

    const IndexType n_queries_samples = n_samples / 10;

    auto indices = generate_indices(n_samples);

    using IndicesIterator = decltype(indices)::iterator;
    using SamplesIterator = decltype(data)::iterator;
    using IndexerType     = ffcl::datastruct::KDTree<IndicesIterator, SamplesIterator>;
    using OptionsType     = IndexerType::Options;
    using AxisSelectionPolicyType =
        ffcl::datastruct::kdtree::policy::HighestVarianceBuild<IndicesIterator, SamplesIterator>;
    using SplittingRulePolicyType =
        ffcl::datastruct::kdtree::policy::QuickselectMedianRange<IndicesIterator, SamplesIterator>;

    // HighestVarianceBuild, MaximumSpreadBuild, CycleThroughAxesBuild
    auto reference_indexer = IndexerType(indices.begin(),
                                         indices.end(),
                                         data.begin(),
                                         data.end(),
                                         n_features,
                                         OptionsType()
                                             .bucket_size(1)
                                             .axis_selection_policy(AxisSelectionPolicyType{})
                                             .splitting_rule_policy(SplittingRulePolicyType{}));

    auto searcher = ffcl::search::Searcher(std::move(reference_indexer));

    auto queries_indices = std::vector(indices.begin(), indices.begin() + n_queries_samples);

    auto union_find = ffcl::datastruct::UnionFind<IndexType>(n_samples);

    // merge all the indices in the queries_indices into the same component
    for (const auto& query_index : queries_indices) {
        union_find.merge(queries_indices[0], query_index);
    }
    const auto shortest_edge = searcher.dual_tree_shortest_edge(union_find, union_find.find(queries_indices[0]), 1);

    std::cout << std::get<0>(shortest_edge) << ", " << std::get<1>(shortest_edge) << ", " << std::get<2>(shortest_edge)
              << "\n";

    auto labels = std::vector<IndexType>(n_samples);

    for (const auto& reference_index : indices) {
        labels[reference_index] = 1;
    }
    for (const auto& query_index : queries_indices) {
        labels[query_index] = 0;
    }
    labels[std::get<0>(shortest_edge)] = 2;
    labels[std::get<1>(shortest_edge)] = 3;

    const auto brute_force_shortest_edge =
        ffcl::search::algorithms::dual_set_shortest_edge(indices.begin(),
                                                         indices.begin() + n_queries_samples,
                                                         data.begin(),
                                                         data.end(),
                                                         n_features,
                                                         indices.begin() + n_queries_samples,
                                                         indices.end(),
                                                         data.begin(),
                                                         data.end(),
                                                         n_features,
                                                         1);

    std::cout << std::get<0>(brute_force_shortest_edge) << " == " << std::get<0>(shortest_edge) << " && "
              << std::get<1>(brute_force_shortest_edge) << " == " << std::get<1>(shortest_edge) << " && "
              << std::get<2>(brute_force_shortest_edge) << " == " << std::get<2>(shortest_edge) << "\n";

    write_data<IndexType>(labels, 1, predictions_folder_ / fs::path(filename));
}
*/

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}