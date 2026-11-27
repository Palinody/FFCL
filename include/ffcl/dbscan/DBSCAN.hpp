#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/search/buffer/Unsorted.hpp"

#include "ffcl/datastruct/bounds/Ball.hpp"

#include "ffcl/search/Search.hpp"

#include "ffcl/datastruct/UnionFind.hpp"

#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ffcl {

template <typename Indexer>
class DBSCAN {
  public:
    using LabelType = typename Indexer::IndexType;
    using DataType  = typename Indexer::DataType;

    static_assert(std::is_trivial_v<LabelType>, "LabelType must be trivial.");
    static_assert(std::is_trivial_v<DataType>, "DataType must be trivial.");

    struct Options {
        Options() = default;

        Options(const Options&) = default;

        Options& min_samples(std::size_t min_samples) {
            min_samples_ = min_samples;
            return *this;
        }

        Options& radius(const DataType& radius) {
            radius_ = radius;
            return *this;
        }

        Options& operator=(const Options& options) {
            min_samples_ = options.min_samples_;
            radius_      = options.radius_;
            return *this;
        }

        std::size_t min_samples_ = 5;
        DataType    radius_      = 0.1;
    };

  public:
    DBSCAN() = default;

    DBSCAN(const Options& options);

    DBSCAN(const DBSCAN&) = delete;

    DBSCAN<Indexer>& set_options(const Options& options);

    /**
     * @brief Predicts the cluster label of a range of indices in a global dataset.
     *
     * The client can use a subset of indices in the dataset that needs to be explored, or the entire range of the
     * indices vector. The indices point to samples in the global dataset. The range of indices specifies the portion of
     * the dataset to cluster, which can range from the empty set to [index.begin(), index.end()] or any variation in
     * between.
     *
     * @param indexer the indexer that was used to index the dataset and rearranged the index
     * @return auto std::vector<LabelType> that has the same length as the input index range:
     * std::distance(global_index_first, global_index_last)
     */
    auto predict(Indexer&& indexer) const;

  private:
    template <typename NeighborsIndices, typename VisitedIndices, typename Predictions>
    void predict_inner(NeighborsIndices&                neighbors_indices,
                       VisitedIndices&                  visited_indices,
                       Predictions&                     predictions,
                       const LabelType&                 cluster_label,
                       const LabelType&                 entry_point_index,
                       const search::Searcher<Indexer>& searcher) const;

    // The original frontier expansion, kept as the general fallback for dimensions where a grid is not
    // worth it (or for a grid that would be too large).
    auto predict_frontier(const search::Searcher<Indexer>& searcher) const -> std::vector<LabelType>;

    // Grid accelerated DBSCAN. The samples are bucketed into cells whose diagonal is below the radius,
    // so every pair of samples in one cell is a neighbour without any distance computation. Core points
    // are found cell by cell, connected with a union-find over core-core pairs, and the labels are
    // assigned afterwards: core samples take their component's label, the others take the label of the
    // earliest component that reaches them, or stay noise.
    auto predict_grid(const search::Searcher<Indexer>& searcher) const -> std::vector<LabelType>;

    enum class SampleStatus : LabelType { noise = 0 };

    // Above this many features the (2k+1)^d cell neighbourhood of the grid grows faster than the tree
    // traversal it replaces, so the frontier expansion is used instead.
    static constexpr std::size_t kMaxGridFeatures = 3;
    // A grid larger than this is not worth building either; fall back to the tree.
    static constexpr std::size_t kMaxGridCells = 4'000'000;
    static constexpr std::size_t kNoCell       = std::numeric_limits<std::size_t>::max();

    Options options_;
};

template <typename Indexer>
DBSCAN<Indexer>::DBSCAN(const Options& options)
  : options_{options} {}

template <typename Indexer>
DBSCAN<Indexer>& DBSCAN<Indexer>::set_options(const Options& options) {
    options_ = options;
    return *this;
}

template <typename Indexer>
auto DBSCAN<Indexer>::predict(Indexer&& indexer) const {
    const auto searcher = search::Searcher(std::forward<Indexer>(indexer));

    const std::size_t n_samples  = searcher.n_samples();
    const std::size_t n_features = searcher.n_features();

    if (n_samples == 0) {
        return std::vector<LabelType>{};
    }

    const bool grid_is_worth_it =
        n_features > 0 && n_features <= kMaxGridFeatures && options_.radius_ > static_cast<DataType>(0);

    if (!grid_is_worth_it) {
        return predict_frontier(searcher);
    }

    return predict_grid(searcher);
}

template <typename Indexer>
auto DBSCAN<Indexer>::predict_frontier(const search::Searcher<Indexer>& searcher) const -> std::vector<LabelType> {
    // the total number of samples that will be searched by index
    const std::size_t n_samples = searcher.n_samples();

    // vector keeping track of the cluster label for each index specified by the global index range
    auto predictions = std::vector<LabelType>(n_samples);

    // initialize the initial cluster counter that's in [0, n_samples)
    auto cluster_label = static_cast<LabelType>(SampleStatus::noise);

    // boolean buffer that keep tracks of the samples that have been already visited
    auto visited_indices = std::make_unique<bool[]>(n_samples);

    // entry_point_candidate_index is the first sample index that is a candidate to initiate a cluster
    for (std::size_t entry_point_candidate_index = 0; entry_point_candidate_index < n_samples;
         ++entry_point_candidate_index) {
        // process the current sample only if it's not visited
        if (!visited_indices[entry_point_candidate_index]) {
            // mark the current sample index as visited
            visited_indices[entry_point_candidate_index] = true;
            // we construct a ball (in hyperspace) object that will be used to query nearby samples
            auto ball_view = datastruct::bounds::BallView(searcher.features_range_first(entry_point_candidate_index),
                                                          searcher.features_range_last(entry_point_candidate_index),
                                                          options_.radius_);
            // the ball view object is passed to a buffer that's used by the searcher to query and accumulate the
            // nearby objects
            auto nn_buffer_query = searcher(search::buffer::Unsorted(std::move(ball_view)));
            // if the nearby samples caught in the ball are dense enough
            if (nn_buffer_query.size() > options_.min_samples_) {
                // we get a new cluster
                ++cluster_label;
                // and assign the cluster label to the initial sample that formed a new cluster
                predictions[entry_point_candidate_index] = cluster_label;
                // we retrieve the nearby samples indices
                auto neighbors_indices = std::move(nn_buffer_query).indices();
                // Every sample joins the frontier at most once. The visited flag is set when a sample is
                // queued rather than when it is dequeued, so a sample reachable from many of its
                // neighbours is not queued again by each of them: the frontier stays the size of the
                // cluster instead of the number of its edges.
                for (const auto& neighbor_index : neighbors_indices) {
                    visited_indices[neighbor_index] = true;
                }
                // neighbors_indices will be populated with all the nearby samples that satisfy the density criterion
                predict_inner(neighbors_indices,
                              visited_indices,
                              predictions,
                              cluster_label,
                              static_cast<LabelType>(entry_point_candidate_index),
                              searcher);
            } else {
                predictions[entry_point_candidate_index] = static_cast<LabelType>(SampleStatus::noise);
            }
        }
    }
    return predictions;
}

template <typename Indexer>
auto DBSCAN<Indexer>::predict_grid(const search::Searcher<Indexer>& searcher) const -> std::vector<LabelType> {
    const std::size_t n_samples   = searcher.n_samples();
    const std::size_t n_features  = searcher.n_features();
    const DataType    radius      = options_.radius_;
    const DataType    radius_sq   = radius * radius;
    const std::size_t min_samples = options_.min_samples_;

    // The cell diagonal is sqrt(n_features) * cell_width. At 0.9 of the radius it stays strictly below
    // the radius, so every pair of samples in one cell is a neighbour without a distance computation,
    // and a sample within the radius is at most ceil(radius / cell_width) cells away on each axis. The
    // slightly smaller cells also keep fewer candidate samples in the neighbourhood than the exact
    // radius / sqrt(n_features) would.
    const double cell_width = static_cast<double>(radius) / std::sqrt(static_cast<double>(n_features)) * 0.9;

    auto squared_distance = [&](std::size_t left_index, std::size_t right_index) -> DataType {
        const auto left  = searcher.features_range_first(left_index);
        const auto right = searcher.features_range_first(right_index);
        if (n_features == 2) {
            const auto dx = left[0] - right[0];
            const auto dy = left[1] - right[1];
            return dx * dx + dy * dy;
        }
        if (n_features == 3) {
            const auto dx = left[0] - right[0];
            const auto dy = left[1] - right[1];
            const auto dz = left[2] - right[2];
            return dx * dx + dy * dy + dz * dz;
        }
        auto sum = DataType{0};
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto difference = left[feature_index] - right[feature_index];
            sum += difference * difference;
        }
        return sum;
    };

    // --- grid extent -------------------------------------------------------------------------------
    auto min_bound = std::vector<DataType>(n_features, std::numeric_limits<DataType>::max());
    auto max_bound = std::vector<DataType>(n_features, std::numeric_limits<DataType>::lowest());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        const auto features = searcher.features_range_first(sample_index);
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            min_bound[feature_index] = std::min(min_bound[feature_index], features[feature_index]);
            max_bound[feature_index] = std::max(max_bound[feature_index], features[feature_index]);
        }
    }

    auto        n_cells_per_feature = std::vector<std::size_t>(n_features, 1);
    std::size_t n_cells             = 1;

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        const auto extent =
            static_cast<double>(max_bound[feature_index]) - static_cast<double>(min_bound[feature_index]);
        auto cells = static_cast<std::size_t>(std::floor(extent / cell_width)) + 1;
        if (cells < 1) {
            cells = 1;
        }
        if (n_cells > kMaxGridCells / cells) {
            return predict_frontier(searcher);
        }
        n_cells *= cells;
        n_cells_per_feature[feature_index] = cells;
    }

    // --- bucket the samples into cells (counting sort, so the samples of a cell are contiguous) ---
    auto sample_cell = std::vector<std::size_t>(n_samples);
    auto cell_of     = [&](std::size_t sample_index) -> std::size_t {
        const auto features = searcher.features_range_first(sample_index);
        auto       cell     = std::size_t{0};
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto quotient =
                (static_cast<double>(features[feature_index]) - static_cast<double>(min_bound[feature_index])) /
                cell_width;
            auto coordinate = quotient <= 0.0 ? std::size_t{0} : static_cast<std::size_t>(quotient);
            if (coordinate >= n_cells_per_feature[feature_index]) {
                coordinate = n_cells_per_feature[feature_index] - 1;
            }
            cell = cell * n_cells_per_feature[feature_index] + coordinate;
        }
        return cell;
    };

    auto cell_start = std::vector<std::size_t>(n_cells + 1, 0);
    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        sample_cell[sample_index] = cell_of(sample_index);
        ++cell_start[sample_cell[sample_index] + 1];
    }
    for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
        cell_start[cell_index + 1] += cell_start[cell_index];
    }

    auto cell_items = std::vector<std::size_t>(n_samples);
    {
        auto cursor = std::vector<std::size_t>(cell_start.begin(), cell_start.end() - 1);
        for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
            cell_items[cursor[sample_cell[sample_index]]++] = sample_index;
        }
    }

    // --- cell neighbourhood ------------------------------------------------------------------------
    // All offsets whose cells can hold a sample within the radius. The minimum gap between a cell and
    // the cell at `offset` is sum(max(0, |offset| - 1) * cell_width)^2, which has to stay below the
    // radius for the cell to be reachable at all.
    const auto  reach          = static_cast<int>(std::ceil(static_cast<double>(radius) / cell_width - 1e-9));
    const auto  span           = static_cast<std::size_t>(2 * reach + 1);
    const auto  radius_sq_d    = static_cast<double>(radius) * static_cast<double>(radius);
    std::size_t n_combinations = 1;
    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        n_combinations *= span;
    }

    auto offsets      = std::vector<std::vector<int>>{};
    auto half_offsets = std::vector<std::vector<int>>{};
    offsets.reserve(n_combinations);
    half_offsets.reserve(n_combinations / 2);

    for (std::size_t combination = 0; combination < n_combinations; ++combination) {
        auto   offset      = std::vector<int>(n_features, 0);
        auto   remainder   = combination;
        double min_gap_sq  = 0.0;
        bool   is_half     = false;
        bool   found_first = false;
        bool   is_zero     = true;
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto digit = static_cast<int>(remainder % span);
            remainder /= span;
            const auto value      = digit - reach;
            offset[feature_index] = value;
            if (value != 0) {
                is_zero        = false;
                const auto gap = static_cast<double>(std::abs(value) - 1) * cell_width;
                min_gap_sq += gap * gap;
                if (!found_first) {
                    is_half     = value > 0;
                    found_first = true;
                }
            }
        }
        if (min_gap_sq >= radius_sq_d) {
            continue;
        }
        // the zero offset is the own cell, which every phase handles explicitly
        if (is_zero) {
            continue;
        }
        if (is_half) {
            half_offsets.push_back(offset);
        }
        offsets.push_back(std::move(offset));
    }

    // Visit the closest cells first: a sample usually reaches the density threshold in the nearby
    // cells, so the scan stops earlier.
    auto offset_gap_sq = [&](const std::vector<int>& offset) {
        auto gap_sq = 0.0;
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            if (offset[feature_index] != 0) {
                const auto gap = static_cast<double>(std::abs(offset[feature_index]) - 1) * cell_width;
                gap_sq += gap * gap;
            }
        }
        return gap_sq;
    };
    std::stable_sort(offsets.begin(), offsets.end(), [&](const auto& left, const auto& right) {
        return offset_gap_sq(left) < offset_gap_sq(right);
    });
    std::stable_sort(half_offsets.begin(), half_offsets.end(), [&](const auto& left, const auto& right) {
        return offset_gap_sq(left) < offset_gap_sq(right);
    });

    auto coordinates = std::vector<std::ptrdiff_t>(n_features, 0);
    auto decode      = [&](std::size_t cell_index) {
        for (std::size_t feature_index = n_features; feature_index-- > 0;) {
            coordinates[feature_index] = static_cast<std::ptrdiff_t>(cell_index % n_cells_per_feature[feature_index]);
            cell_index /= n_cells_per_feature[feature_index];
        }
    };
    auto neighbour_of = [&](const std::vector<int>& offset) -> std::size_t {
        auto cell = std::size_t{0};
        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const auto coordinate = coordinates[feature_index] + offset[feature_index];
            if (coordinate < 0 || coordinate >= static_cast<std::ptrdiff_t>(n_cells_per_feature[feature_index])) {
                return kNoCell;
            }
            cell = cell * n_cells_per_feature[feature_index] + static_cast<std::size_t>(coordinate);
        }
        return cell;
    };

    auto cell_is_dense = std::vector<char>(n_cells, 0);
    for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
        cell_is_dense[cell_index] =
            (cell_start[cell_index + 1] - cell_start[cell_index]) > min_samples ? char{1} : char{0};
    }

    const auto average_candidates =
        static_cast<double>(n_samples) * static_cast<double>(offsets.size() + 1) / static_cast<double>(n_cells);

    // --- phase 1: core points ----------------------------------------------------------------------
    auto core = std::vector<char>(n_samples, 0);

    // When the neighbourhood of a cell holds no more samples than `min_samples` on average, most cells
    // cannot produce a core point at all and the count can stop as soon as the remaining candidates
    // cannot reach the threshold. On denser data that bookkeeping only costs time, so it is skipped.
    const bool use_bounds = average_candidates <= static_cast<double>(min_samples);

    if (!use_bounds) {
        for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
            const auto begin = cell_start[cell_index];
            const auto end   = cell_start[cell_index + 1];
            if (begin == end) {
                continue;
            }
            const auto cell_size = end - begin;

            if (cell_is_dense[cell_index]) {
                for (auto item = begin; item < end; ++item) {
                    core[cell_items[item]] = 1;
                }
                continue;
            }

            decode(cell_index);
            for (auto item = begin; item < end; ++item) {
                const auto sample_index    = cell_items[item];
                auto       neighbour_count = cell_size;
                if (neighbour_count > min_samples) {
                    core[sample_index] = 1;
                    continue;
                }
                for (const auto& offset : offsets) {
                    const auto neighbour_cell = neighbour_of(offset);
                    if (neighbour_cell == kNoCell) {
                        continue;
                    }
                    for (auto neighbour_item = cell_start[neighbour_cell];
                         neighbour_item < cell_start[neighbour_cell + 1];
                         ++neighbour_item) {
                        if (squared_distance(sample_index, cell_items[neighbour_item]) < radius_sq) {
                            ++neighbour_count;
                            if (neighbour_count > min_samples) {
                                break;
                            }
                        }
                    }
                    if (neighbour_count > min_samples) {
                        break;
                    }
                }
                core[sample_index] = neighbour_count > min_samples ? char{1} : char{0};
            }
        }
    } else {
        for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
            const auto begin = cell_start[cell_index];
            const auto end   = cell_start[cell_index + 1];
            if (begin == end) {
                continue;
            }
            const auto cell_size = end - begin;

            if (cell_is_dense[cell_index]) {
                for (auto item = begin; item < end; ++item) {
                    core[cell_items[item]] = 1;
                }
                continue;
            }

            decode(cell_index);

            // The whole neighbourhood cannot hold enough samples: every sample of the cell is noise.
            auto neighbourhood_size = cell_size;
            for (const auto& offset : offsets) {
                const auto neighbour_cell = neighbour_of(offset);
                if (neighbour_cell != kNoCell) {
                    neighbourhood_size += cell_start[neighbour_cell + 1] - cell_start[neighbour_cell];
                }
            }
            if (neighbourhood_size <= min_samples) {
                continue;
            }

            for (auto item = begin; item < end; ++item) {
                const auto sample_index    = cell_items[item];
                auto       neighbour_count = cell_size;
                if (neighbour_count > min_samples) {
                    core[sample_index] = 1;
                    continue;
                }
                auto remaining = neighbourhood_size - cell_size;
                for (const auto& offset : offsets) {
                    const auto neighbour_cell = neighbour_of(offset);
                    if (neighbour_cell == kNoCell) {
                        continue;
                    }
                    const auto neighbour_size = cell_start[neighbour_cell + 1] - cell_start[neighbour_cell];
                    for (auto neighbour_item = cell_start[neighbour_cell];
                         neighbour_item < cell_start[neighbour_cell + 1];
                         ++neighbour_item) {
                        if (squared_distance(sample_index, cell_items[neighbour_item]) < radius_sq) {
                            ++neighbour_count;
                            if (neighbour_count > min_samples) {
                                break;
                            }
                        }
                    }
                    if (neighbour_count > min_samples) {
                        break;
                    }
                    remaining -= neighbour_size;
                    // even counting every remaining candidate cannot reach the threshold
                    if (neighbour_count + remaining <= min_samples) {
                        break;
                    }
                }
                core[sample_index] = neighbour_count > min_samples ? char{1} : char{0};
            }
        }
    }

    // --- phase 2: connectivity of the core points --------------------------------------------------
    // All core samples of a cell are within the radius of each other, so they form a clique and the
    // union-find can run over cells instead of samples.
    auto cell_has_core = std::vector<char>(n_cells, 0);
    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        if (core[sample_index]) {
            cell_has_core[sample_cell[sample_index]] = 1;
        }
    }

    // Cells whose own or neighbouring cells hold a core sample. A non-core sample outside of that
    // region is noise and does not have to look at its neighbourhood at all.
    auto cell_has_core_nearby = std::vector<char>(n_cells, use_bounds ? 0 : 1);
    if (use_bounds) {
        for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
            if (!cell_has_core[cell_index]) {
                continue;
            }
            cell_has_core_nearby[cell_index] = 1;
            decode(cell_index);
            for (const auto& offset : offsets) {
                const auto neighbour_cell = neighbour_of(offset);
                if (neighbour_cell != kNoCell) {
                    cell_has_core_nearby[neighbour_cell] = 1;
                }
            }
        }
    }

    auto union_find = datastruct::UnionFind<LabelType>(n_cells);

    // Only the half of the offsets with a positive first non-zero component is visited, so each pair
    // of neighbouring cells is handled once. As soon as one core pair of two cells is within the
    // radius the cells are connected and the remaining pairs cannot change that, so the scan stops.
    for (std::size_t cell_index = 0; cell_index < n_cells; ++cell_index) {
        if (!cell_has_core[cell_index]) {
            continue;
        }
        decode(cell_index);
        for (const auto& offset : half_offsets) {
            const auto neighbour_cell = neighbour_of(offset);
            if (neighbour_cell == kNoCell || !cell_has_core[neighbour_cell]) {
                continue;
            }
            if (union_find.find(static_cast<LabelType>(cell_index)) ==
                union_find.find(static_cast<LabelType>(neighbour_cell))) {
                continue;
            }
            bool connected = false;
            for (auto item = cell_start[cell_index]; item < cell_start[cell_index + 1] && !connected; ++item) {
                const auto sample_index = cell_items[item];
                if (!core[sample_index]) {
                    continue;
                }
                for (auto neighbour_item = cell_start[neighbour_cell]; neighbour_item < cell_start[neighbour_cell + 1];
                     ++neighbour_item) {
                    const auto neighbour_index = cell_items[neighbour_item];
                    if (!core[neighbour_index]) {
                        continue;
                    }
                    if (squared_distance(sample_index, neighbour_index) < radius_sq) {
                        union_find.merge(static_cast<LabelType>(cell_index), static_cast<LabelType>(neighbour_cell));
                        connected = true;
                        break;
                    }
                }
            }
        }
    }

    // --- phase 3: labels ---------------------------------------------------------------------------
    // Components are numbered by their smallest core sample index, which is exactly the order in which
    // the sequential frontier expansion discovers them.
    auto      predictions     = std::vector<LabelType>(n_samples, static_cast<LabelType>(SampleStatus::noise));
    auto      component_label = std::vector<LabelType>(n_cells, static_cast<LabelType>(0));
    LabelType next_label      = 0;

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        if (!core[sample_index]) {
            continue;
        }
        const auto root = union_find.find(static_cast<LabelType>(sample_cell[sample_index]));
        if (component_label[root] == static_cast<LabelType>(0)) {
            component_label[root] = ++next_label;
        }
    }

    auto core_label     = std::vector<LabelType>(n_samples, static_cast<LabelType>(0));
    auto cell_min_label = std::vector<LabelType>(n_cells, std::numeric_limits<LabelType>::max());

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        if (!core[sample_index]) {
            continue;
        }
        const auto cell_index      = sample_cell[sample_index];
        const auto label           = component_label[union_find.find(static_cast<LabelType>(cell_index))];
        core_label[sample_index]   = label;
        predictions[sample_index]  = label;
        cell_min_label[cell_index] = std::min(cell_min_label[cell_index], label);
    }

    const auto no_label = std::numeric_limits<LabelType>::max();

    for (std::size_t sample_index = 0; sample_index < n_samples; ++sample_index) {
        if (core[sample_index]) {
            continue;
        }
        const auto cell_index = sample_cell[sample_index];
        if (!cell_has_core_nearby[cell_index]) {
            continue;
        }
        // every sample of the own cell is within the radius, so its smallest label is a valid choice
        auto best_label = cell_min_label[cell_index];
        if (best_label > static_cast<LabelType>(1)) {
            decode(cell_index);
            for (const auto& offset : half_offsets) {
                // the mirrored offset covers the other side of the neighbourhood
                for (int sign = 0; sign < 2; ++sign) {
                    auto mirrored = offset;
                    if (sign == 1) {
                        for (auto& value : mirrored) {
                            value = -value;
                        }
                    }
                    const auto neighbour_cell = neighbour_of(mirrored);
                    if (neighbour_cell == kNoCell || cell_min_label[neighbour_cell] >= best_label) {
                        continue;
                    }
                    for (auto neighbour_item = cell_start[neighbour_cell];
                         neighbour_item < cell_start[neighbour_cell + 1];
                         ++neighbour_item) {
                        const auto neighbour_index = cell_items[neighbour_item];
                        if (!core[neighbour_index]) {
                            continue;
                        }
                        if (squared_distance(sample_index, neighbour_index) < radius_sq) {
                            best_label = std::min(best_label, core_label[neighbour_index]);
                        }
                    }
                    if (best_label == static_cast<LabelType>(1)) {
                        break;
                    }
                }
                if (best_label == static_cast<LabelType>(1)) {
                    break;
                }
            }
        }
        if (best_label != no_label) {
            predictions[sample_index] = best_label;
        }
    }

    return predictions;
}

template <typename Indexer>
template <typename NeighborsIndices, typename VisitedIndices, typename Predictions>
void DBSCAN<Indexer>::predict_inner(NeighborsIndices&                neighbors_indices,
                                    VisitedIndices&                  visited_indices,
                                    Predictions&                     predictions,
                                    const LabelType&                 cluster_label,
                                    const LabelType&                 entry_point_index,
                                    const search::Searcher<Indexer>& searcher) const {
    // iterate over the samples that are assigned to the current cluster, appending the ones that are
    // discovered along the way
    for (std::size_t cluster_sample_index = 0; cluster_sample_index < neighbors_indices.size();
         ++cluster_sample_index) {
        const auto neighbor_index = neighbors_indices[cluster_sample_index];

        // every sample in the frontier is queued but not queried yet, except the entry point, which
        // was queried to open this cluster
        if (neighbor_index != entry_point_index) {
            auto ball_view = datastruct::bounds::BallView(searcher.features_range_first(neighbor_index),
                                                          searcher.features_range_last(neighbor_index),
                                                          options_.radius_);

            auto inner_neighbors_buffer = searcher(search::buffer::Unsorted(std::move(ball_view)));

            if (inner_neighbors_buffer.size() > options_.min_samples_) {
                auto inner_neighbors_indices = std::move(inner_neighbors_buffer).indices();

                // queue each newly discovered neighbor exactly once
                for (const auto& inner_neighbor_index : inner_neighbors_indices) {
                    if (!visited_indices[inner_neighbor_index]) {
                        visited_indices[inner_neighbor_index] = true;
                        neighbors_indices.push_back(inner_neighbor_index);

                    } else if (predictions[inner_neighbor_index] == static_cast<LabelType>(SampleStatus::noise)) {
                        // A border point: already visited, found not to be core, and labelled noise before
                        // any cluster reached it. It belongs to this cluster, but it is not core, so it
                        // does not expand the frontier. "Noise" is provisional until every core point has
                        // been expanded.
                        predictions[inner_neighbor_index] = cluster_label;
                    }
                }
            }
        }
        // assign neighbor_index to a cluster if its not already the case
        if (predictions[neighbor_index] == static_cast<LabelType>(SampleStatus::noise)) {
            predictions[neighbor_index] = cluster_label;
        }
    }
}

}  // namespace ffcl
