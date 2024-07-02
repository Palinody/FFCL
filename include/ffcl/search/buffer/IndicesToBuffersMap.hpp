#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"
#include "ffcl/datastruct/bounds/distances/MinDistance.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace ffcl::search::buffer {

template <typename QueryNodePtr, typename ReferenceNodePtr>
struct NodesCombinationKey {
    bool operator==(const NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>& other) const {
        return query_node == other.query_node && reference_node == other.reference_node;
    }

    QueryNodePtr     query_node;
    ReferenceNodePtr reference_node;
};

template <typename QueryNodePtr, typename ReferenceNodePtr>
NodesCombinationKey(QueryNodePtr, ReferenceNodePtr) -> NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>;

}  // namespace ffcl::search::buffer

namespace std {

template <typename QueryNodePtr, typename ReferenceNodePtr>
struct hash<ffcl::search::buffer::NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>> {
    std::size_t operator()(
        const ffcl::search::buffer::NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>& key) const noexcept {
        std::size_t hash_1 = std::hash<typename QueryNodePtr::element_type*>{}(key.query_node.get());
        std::size_t hash_2 = std::hash<typename ReferenceNodePtr::element_type*>{}(key.reference_node.get());

        if constexpr (sizeof(std::size_t) >= 8) {
            hash_1 ^= hash_2 + 0x517cc1b727220a95 + (hash_1 << 6) + (hash_1 >> 2);
        } else {
            hash_1 ^= hash_2 + 0x9e3779b9 + (hash_1 << 6) + (hash_1 >> 2);
        }
        return hash_1;
    }
};

}  // namespace std

namespace ffcl::search::buffer {

template <typename Index, typename Distance>
using Edge = std::tuple<Index, Index, Distance>;

template <typename Index, typename Distance>
constexpr auto make_edge(const Index& index_1, const Index& index_2, const Distance& distance) {
    return std::make_tuple(index_1, index_2, distance);
}

template <typename Index, typename Distance>
constexpr auto make_default_edge() {
    return make_edge(common::infinity<Index>(), common::infinity<Index>(), common::infinity<Distance>());
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
class IndicesToBuffersMap {
  public:
    using IndexType    = typename Buffer::IndexType;
    using DistanceType = typename Buffer::DistanceType;

    static_assert(std::is_trivial_v<IndexType>, "IndexType must be trivial.");
    static_assert(std::is_trivial_v<DistanceType>, "DistanceType must be trivial.");

    using BufferType = Buffer;

    static_assert(common::is_crtp_of<BufferType, buffer::StaticBuffer>::value,
                  "BufferType must inherit from StaticBuffer<Derived>");

    static_assert(!std::is_same_v<BufferType, void>,
                  "Deduced BufferType: void. Buffer type couldn't be deduced from 'BufferArgs&&...'.");

    using IndexToBufferMapType          = std::unordered_map<IndexType, BufferType>;
    using IndexToBufferMapIterator      = typename IndexToBufferMapType::iterator;
    using IndexToBufferMapConstIterator = typename IndexToBufferMapType::const_iterator;

    using QueryToBufferType = std::pair<IndexType, BufferType>;

    using QueryNodePtr     = typename QueryIndexer::NodePtr;
    using ReferenceNodePtr = typename ReferenceIndexer::NodePtr;

    using QuerySamplesIteratorType     = typename QueryIndexer::SamplesIteratorType;
    using ReferenceSamplesIteratorType = typename ReferenceIndexer::SamplesIteratorType;

    IndicesToBuffersMap(const QueryIndexer& query_indexer, const ReferenceIndexer& reference_indexer);

    auto tightest_edge() const;

    auto tightest_query_to_buffer() && -> QueryToBufferType;

    template <typename QueryIndicesIterator, typename ReferenceIndicesIterator, typename... BufferArgs>
    void partial_search_for_each_query(const QueryIndicesIterator&     query_indices_range_first,
                                       const QueryIndicesIterator&     query_indices_range_last,
                                       const ReferenceIndicesIterator& reference_indices_range_first,
                                       const ReferenceIndicesIterator& reference_indices_range_last,
                                       BufferArgs&&... buffer_args);

    auto cost(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node) -> std::optional<DistanceType>;

    auto update_cost(const QueryNodePtr& query_node, const ReferenceNodePtr&, const DistanceType& cost)
        -> std::optional<DistanceType>;

    bool emplace_nodes_combination_if_not_found(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node);

  private:
    struct BoundsLimits {
        BoundsLimits()
          : closest_limit{common::infinity<DistanceType>()}
          , furthest_limit{0} {}

        void try_update_closest_limit(const DistanceType& closest_limit_candidate) {
            closest_limit = std::min(closest_limit, closest_limit_candidate);
        }

        void try_update_furthest_limit(const DistanceType& furthest_limit_candidate) {
            closest_limit = std::max(closest_limit, furthest_limit_candidate);
        }

        void try_update_limits(const DistanceType& bound_candidate) {
            try_update_closest_limit(bound_candidate);
            try_update_furthest_limit(bound_candidate);
        }

        auto compute_adjusted_bound(const DistanceType& node_diameter) const -> DistanceType {
            return (closest_limit < common::infinity<DistanceType>() - node_diameter)
                       ? std::min(furthest_limit, closest_limit + node_diameter)
                       : common::infinity<DistanceType>();
        }

        DistanceType closest_limit;
        DistanceType furthest_limit;
    };

    template <typename FeaturesRangeIterator, typename... BufferArgs>
    constexpr auto find_or_emplace_buffer(const IndexType&             index,
                                          const FeaturesRangeIterator& features_range_first,
                                          const FeaturesRangeIterator& features_range_last,
                                          BufferArgs&&... buffer_args) -> IndexToBufferMapIterator;

    constexpr auto emplace(const IndexType& index, BufferType&& buffer) -> std::pair<IndexToBufferMapIterator, bool>;

    auto query_node_furthest_bound(const QueryNodePtr& query_node) -> DistanceType;

    auto find_max_furthest_distance_in_node(const QueryNodePtr& query_node) const -> DistanceType;

    auto update_bounds_limits(const QueryNodePtr& query_node) ->
        typename std::unordered_map<QueryNodePtr, BoundsLimits>::iterator;

    QuerySamplesIteratorType query_samples_range_first_;
    QuerySamplesIteratorType query_samples_range_last_;
    std::size_t              query_n_features_;

    ReferenceSamplesIteratorType reference_samples_range_first_;
    ReferenceSamplesIteratorType reference_samples_range_last_;
    std::size_t                  reference_n_features_;

    IndexToBufferMapType queries_to_buffers_map_ = IndexToBufferMapType{};

    IndexToBufferMapIterator tightest_query_to_buffer_it_ = queries_to_buffers_map_.end();

    std::unordered_set<NodesCombinationKey<QueryNodePtr, ReferenceNodePtr>> visited_nodes_combinations_;

    std::unordered_map<QueryNodePtr, BoundsLimits> query_nodes_to_bounds_limits_map_;
};

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
IndicesToBuffersMap(const QueryIndexer&, const ReferenceIndexer&)
    -> IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>;

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto make_indices_to_buffers_map(const QueryIndexer& query_indexer, const ReferenceIndexer& reference_indexer)
    -> IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer> {
    return IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>(query_indexer, reference_indexer);
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::IndicesToBuffersMap(
    const QueryIndexer&     query_indexer,
    const ReferenceIndexer& reference_indexer)
  : query_samples_range_first_{query_indexer.begin()}
  , query_samples_range_last_{query_indexer.end()}
  , query_n_features_{query_indexer.n_features()}
  , reference_samples_range_first_{reference_indexer.begin()}
  , reference_samples_range_last_{reference_indexer.end()}
  , reference_n_features_{reference_indexer.n_features()} {}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::tightest_edge() const {
    return make_edge(tightest_query_to_buffer_it_->first,
                     tightest_query_to_buffer_it_->second.furthest_index(),
                     tightest_query_to_buffer_it_->second.furthest_distance());
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::tightest_query_to_buffer() && -> QueryToBufferType {
    return std::move(*tightest_query_to_buffer_it_);
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
template <typename QueryIndicesIterator, typename ReferenceIndicesIterator, typename... BufferArgs>
void IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::partial_search_for_each_query(
    const QueryIndicesIterator&     query_indices_range_first,
    const QueryIndicesIterator&     query_indices_range_last,
    const ReferenceIndicesIterator& reference_indices_range_first,
    const ReferenceIndicesIterator& reference_indices_range_last,
    BufferArgs&&... buffer_args) {
    // Iterate through all query indices within the specified range of the query node.
    for (auto query_index_it = query_indices_range_first; query_index_it != query_indices_range_last;
         ++query_index_it) {
        auto query_to_buffer_it = this->find_or_emplace_buffer(
            *query_index_it,
            query_samples_range_first_ + (*query_index_it) * query_n_features_,
            query_samples_range_first_ + (*query_index_it) * query_n_features_ + query_n_features_,
            std::forward<BufferArgs>(buffer_args)...);

        // Regardless of whether the buffer was just inserted or already existed, perform a partial search
        // operation on the buffer. This operation updates the buffer based on a range of reference samples.
        query_to_buffer_it->second.partial_search(reference_indices_range_first,
                                                  reference_indices_range_last,
                                                  reference_samples_range_first_,
                                                  reference_samples_range_last_,
                                                  reference_n_features_);

        // Check if tightest_query_to_buffer_it_ is unset.
        const bool is_tightest_query_to_buffer_unset = (tightest_query_to_buffer_it_ == queries_to_buffers_map_.end());

        // Only proceed with capacity and distance comparisons if tightest_query_to_buffer_it_ is set.
        if (!is_tightest_query_to_buffer_unset) {
            const auto current_capacity           = query_to_buffer_it->second.remaining_capacity();
            const auto tightest_capacity          = tightest_query_to_buffer_it_->second.remaining_capacity();
            const auto current_furthest_distance  = query_to_buffer_it->second.furthest_distance();
            const auto tightest_furthest_distance = tightest_query_to_buffer_it_->second.furthest_distance();

            // We update the current best query to buffer iterator if the buffer is fuller than the current one or if
            // both are completely full and the buffer's furthest distance is lesser than the one currently registered.
            if ((current_capacity < tightest_capacity) ||
                (!current_capacity && !tightest_capacity && current_furthest_distance < tightest_furthest_distance)) {
                tightest_query_to_buffer_it_ = query_to_buffer_it;
            }
        } else {
            // If tightest_query_to_buffer_it_ is unset, set it to the current iterator.
            tightest_query_to_buffer_it_ = query_to_buffer_it;
        }
    }
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::cost(const QueryNodePtr&     query_node,
                                                                       const ReferenceNodePtr& reference_node)
    -> std::optional<DistanceType> {
    const auto min_distance = datastruct::bounds::min_distance(query_node->bound_, reference_node->bound_);

    return (query_node_furthest_bound(query_node) < min_distance) ? std::nullopt : std::make_optional(min_distance);
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::update_cost(const QueryNodePtr& query_node,
                                                                              const ReferenceNodePtr&,
                                                                              const DistanceType& cost)
    -> std::optional<DistanceType> {
    return (query_node_furthest_bound(query_node) < cost) ? std::nullopt : std::make_optional(cost);
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
bool IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::emplace_nodes_combination_if_not_found(
    const QueryNodePtr&     query_node,
    const ReferenceNodePtr& reference_node) {
    auto nodes_combination_key = NodesCombinationKey{query_node, reference_node};

    if (visited_nodes_combinations_.find(nodes_combination_key) == visited_nodes_combinations_.end()) {
        // Returns a pair consisting of an iterator to the inserted element (or to the element that prevented the
        // insertion) and a bool value set to true if and only if the insertion took place.
        // We are only interested in the boolean value.
        return visited_nodes_combinations_.emplace(nodes_combination_key).second;
    }
    return false;
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
template <typename FeaturesRangeIterator, typename... BufferArgs>
constexpr auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::find_or_emplace_buffer(
    const IndexType&             index,
    const FeaturesRangeIterator& features_range_first,
    const FeaturesRangeIterator& features_range_last,
    BufferArgs&&... buffer_args) -> IndexToBufferMapIterator {
    // Attempt to find the buffer associated with the current index in the buffer map.
    auto index_to_buffer_it = queries_to_buffers_map_.find(index);
    // If the current index does not have an associated buffer in the map,
    if (index_to_buffer_it == queries_to_buffers_map_.end()) {
        auto buffer = BufferType(/**/ features_range_first,
                                 /**/ features_range_last,
                                 /**/ std::forward<BufferArgs>(buffer_args)...);
        // Attempt to insert the newly created buffer into the map. If an element with the same
        // index already exists, emplace does nothing. Otherwise, it inserts the new element.
        // The method returns a pair, where the first element is an iterator to the inserted element
        // (or to the element that prevented the insertion) and the second element is a boolean
        // indicating whether the insertion took place.
        // We are only interested in the first element of the pair.
        index_to_buffer_it = emplace(index, std::move(buffer)).first;
    }
    return index_to_buffer_it;
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
constexpr auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::emplace(const IndexType& index,
                                                                                    BufferType&&     buffer)
    -> std::pair<IndexToBufferMapIterator, bool> {
    return queries_to_buffers_map_.emplace(index, std::move(buffer));
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>

auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::query_node_furthest_bound(
    const QueryNodePtr& query_node) -> DistanceType {
    // auto furthest_distance = find_max_furthest_distance_in_node(query_node);
    // return furthest_distance;
    // /*
    auto query_node_to_bound_limits_it = update_bounds_limits(query_node);

    if (!query_node->is_leaf()) {
        auto left_query_node_to_bound_limits_it  = query_nodes_to_bounds_limits_map_.find(query_node->left_);
        auto right_query_node_to_bound_limits_it = query_nodes_to_bounds_limits_map_.find(query_node->right_);

        // Make a default bounds limits cache if the one for the node isnt available yet.
        if (left_query_node_to_bound_limits_it == query_nodes_to_bounds_limits_map_.end()) {
            left_query_node_to_bound_limits_it =
                query_nodes_to_bounds_limits_map_.emplace(query_node->left_, BoundsLimits{}).first;

        } else {
            query_node_to_bound_limits_it->second.try_update_closest_limit(
                left_query_node_to_bound_limits_it->second.closest_limit);

            query_node_to_bound_limits_it->second.try_update_furthest_limit(
                left_query_node_to_bound_limits_it->second.furthest_limit);
        }
        // Make a default bounds limits cache if the one for the node isnt available yet.
        if (right_query_node_to_bound_limits_it == query_nodes_to_bounds_limits_map_.end()) {
            right_query_node_to_bound_limits_it =
                query_nodes_to_bounds_limits_map_.emplace(query_node->right_, BoundsLimits{}).first;

        } else {
            query_node_to_bound_limits_it->second.try_update_closest_limit(
                right_query_node_to_bound_limits_it->second.closest_limit);

            query_node_to_bound_limits_it->second.try_update_furthest_limit(
                right_query_node_to_bound_limits_it->second.furthest_limit);
        }
    }
    return query_node_to_bound_limits_it->second.compute_adjusted_bound(query_node->diameter());
    // */
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::update_bounds_limits(const QueryNodePtr& query_node)
    -> typename std::unordered_map<QueryNodePtr, BoundsLimits>::iterator {
    static constexpr auto infinity = common::infinity<DistanceType>();

    auto query_node_to_bound_limits_it = query_nodes_to_bounds_limits_map_.find(query_node);
    // Make a default bounds limits cache if the one for the current query node isnt available yet.
    if (query_node_to_bound_limits_it == query_nodes_to_bounds_limits_map_.end()) {
        query_node_to_bound_limits_it = query_nodes_to_bounds_limits_map_.emplace(query_node, BoundsLimits{}).first;
    }
    for (const auto& query_index : *query_node) {
        const auto index_to_buffer_it = queries_to_buffers_map_.find(query_index);
        // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by
        // default. We also don't need to iterate further since the next nodes will never be greater than
        // infinity.
        if (index_to_buffer_it == queries_to_buffers_map_.cend()) {
            query_node_to_bound_limits_it->second.furthest_limit = infinity;
        }
        // If there's remaining space in the buffers, then candidates might potentially be further than the
        // buffer's current furthest distance.
        else if (index_to_buffer_it->second.remaining_capacity()) {
            query_node_to_bound_limits_it->second.furthest_limit = infinity;

        } else {
            const auto query_buffer_furthest_distance = index_to_buffer_it->second.furthest_distance();

            query_node_to_bound_limits_it->second.try_update_limits(query_buffer_furthest_distance);
        }
    }
    return query_node_to_bound_limits_it;
}

template <typename Buffer, typename QueryIndexer, typename ReferenceIndexer>
auto IndicesToBuffersMap<Buffer, QueryIndexer, ReferenceIndexer>::find_max_furthest_distance_in_node(
    const QueryNodePtr& query_node) const -> DistanceType {
    DistanceType queries_max_upper_bound = 0;

    for (const auto& query_index : *query_node) {
        const auto index_to_buffer_it = queries_to_buffers_map_.find(query_index);

        // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by
        // default. We also don't need to iterate further since the next nodes will never be greater than
        // infinity.
        if (index_to_buffer_it == queries_to_buffers_map_.cend()) {
            return common::infinity<DistanceType>();
        }
        // If there's remaining space in the buffers, then candidates might potentially be further than the
        // buffer's current furthest distance.
        else if (index_to_buffer_it->second.remaining_capacity()) {
            return common::infinity<DistanceType>();

        } else {
            const auto query_buffer_furthest_distance = index_to_buffer_it->second.furthest_distance();

            if (query_buffer_furthest_distance > queries_max_upper_bound) {
                queries_max_upper_bound = query_buffer_furthest_distance;
            }
        }
    }
    return queries_max_upper_bound;
}

}  // namespace ffcl::search::buffer