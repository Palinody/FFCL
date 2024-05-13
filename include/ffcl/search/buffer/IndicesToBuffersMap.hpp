#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>

namespace ffcl::search::buffer {

struct NodesCombinationKey {
    bool operator==(const NodesCombinationKey& other) const {
        return address_1 == other.address_1 && address_2 == other.address_2;
    }

    const void* address_1;
    const void* address_2;
};

}  // namespace ffcl::search::buffer

template <>
struct std::hash<ffcl::search::buffer::NodesCombinationKey> {
    std::size_t operator()(const ffcl::search::buffer::NodesCombinationKey& key) const noexcept {
        std::size_t hash_1 = std::hash<const void*>{}(key.address_1);
        std::size_t hash_2 = std::hash<const void*>{}(key.address_2);

        if constexpr (sizeof(std::size_t) >= 8) {
            hash_1 ^= hash_2 + 0x517cc1b727220a95 + (hash_1 << 6) + (hash_1 >> 2);
        } else {
            hash_1 ^= hash_2 + 0x9e3779b9 + (hash_1 << 6) + (hash_1 >> 2);
        }
        return hash_1;
    }
};

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

template <typename Buffer, typename QuerySamplesIterator, typename ReferenceSamplesIterator>
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

    IndicesToBuffersMap(const QuerySamplesIterator&     query_samples_range_first,
                        const QuerySamplesIterator&     query_samples_range_last,
                        std::size_t                     query_n_features,
                        const ReferenceSamplesIterator& reference_samples_range_first,
                        const ReferenceSamplesIterator& reference_samples_range_last,
                        std::size_t                     reference_n_features)
      : query_samples_range_first_{query_samples_range_first}
      , query_samples_range_last_{query_samples_range_last}
      , query_n_features_{query_n_features}
      , reference_samples_range_first_{reference_samples_range_first}
      , reference_samples_range_last_{reference_samples_range_last}
      , reference_n_features_{reference_n_features} {}

    auto tightest_edge() const {
        return make_edge(tightest_query_to_buffer_it_->first,
                         tightest_query_to_buffer_it_->second.furthest_index(),
                         tightest_query_to_buffer_it_->second.furthest_distance());
    }

    template <typename FeaturesRangeIterator, typename... BufferArgs>
    constexpr auto find_or_emplace_buffer(const IndexType&             index,
                                          const FeaturesRangeIterator& features_range_first,
                                          const FeaturesRangeIterator& features_range_last,
                                          BufferArgs&&... buffer_args) -> IndexToBufferMapIterator {
        // Attempt to find the buffer associated with the current index in the buffer map.
        auto index_to_buffer_it = query_to_buffer_map_.find(index);
        // If the current index does not have an associated buffer in the map,
        if (index_to_buffer_it == query_to_buffer_map_.end()) {
            auto buffer = BufferType(/**/ features_range_first,
                                     /**/ features_range_last,
                                     /**/ std::forward<BufferArgs>(buffer_args)...);
            // Attempt to insert the newly created buffer into the map. If an element with the same
            // index already exists, emplace does nothing. Otherwise, it inserts the new element.
            // The method returns a pair, where the first element is an iterator to the inserted element
            // (or to the element that prevented the insertion) and the second element is a boolean
            // indicating whether the insertion took place.
            // We are only interested in the first element of the pair.
            index_to_buffer_it = query_to_buffer_map_.emplace(index, std::move(buffer)).first;
        }
        return index_to_buffer_it;
    }

    constexpr auto emplace(const IndexType& index, BufferType&& buffer) -> std::pair<IndexToBufferMapIterator, bool> {
        return query_to_buffer_map_.emplace(index, std::move(buffer));
    }

    template <typename QueryIndicesIterator, typename ReferenceIndicesIterator, typename... BufferArgs>
    void partial_search_for_each_query(const QueryIndicesIterator&     query_indices_range_first,
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

            // Update if no tightest buffer has been initialised yet.
            // Or, update if the buffer's max capacity has been reached and its furthest distance is less than the one
            // of the current tightest buffer.
            if (tightest_query_to_buffer_it_ == query_to_buffer_map_.end() ||
                (!query_to_buffer_it->second.remaining_capacity() &&
                 query_to_buffer_it->second.furthest_distance() <
                     tightest_query_to_buffer_it_->second.furthest_distance())) {
                tightest_query_to_buffer_it_ = query_to_buffer_it;
            }
        }
    }

    template <typename QueryNodePtr, typename ReferenceNodePtr>
    auto cost(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node) -> std::optional<DistanceType> {
        const auto min_distance = cache_.nodes_combination_min_distance(query_node, reference_node);

        return (cache_.queries_furthest_distance(query_node) < min_distance) ? std::nullopt
                                                                             : std::make_optional(min_distance);
    }

  private:
    class Cache {
        using IndicesToBuffersMapType = IndicesToBuffersMap<BufferType, QuerySamplesIterator, ReferenceSamplesIterator>;

      public:
        constexpr Cache(const IndicesToBuffersMapType& wrapper_class)
          : this_{wrapper_class} {}

        template <typename QueryNodePtr>
        auto queries_furthest_distance(const QueryNodePtr& query_node) -> DistanceType {
            const auto* raw_query_node = extract_raw_ptr(query_node);

            auto query_node_furthest_distance_it = query_node_to_furthest_distance_map_.find(raw_query_node);

            // If the current query_node is already cached.
            if (query_node_furthest_distance_it != query_node_to_furthest_distance_map_.end()) {
                if (!query_node->is_leaf()) {
                    query_node_furthest_distance_it->second =
                        std::max(query_node_furthest_distance_it->second,
                                 std::max(queries_furthest_distance(query_node->left_),
                                          queries_furthest_distance(query_node->right_)));
                }
                return query_node_furthest_distance_it->second;
            }
            // Base case: will only perform this calculation the node is leaf.
            auto furthest_distance = find_max_furthest_distance(query_node);

            if (!query_node->is_leaf()) {
                furthest_distance = std::max(furthest_distance,
                                             std::max(queries_furthest_distance(query_node->left_),
                                                      queries_furthest_distance(query_node->right_)));
            }
            query_node_to_furthest_distance_map_.emplace(raw_query_node, furthest_distance);

            return furthest_distance;
        }

        template <typename QueryNodePtr, typename ReferenceNodePtr>
        auto nodes_combination_min_distance(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node)
            -> DistanceType {
            const auto* raw_query_node        = extract_raw_ptr(query_node);
            const auto* raw_reference_node    = extract_raw_ptr(reference_node);
            const auto  nodes_combination_key = NodesCombinationKey{raw_query_node, raw_reference_node};

            auto nodes_combination_to_min_distance_it =
                nodes_combination_to_min_distance_map_.find(nodes_combination_key);

            // Checks if the result is cached and returns early if so.
            if (nodes_combination_to_min_distance_it != nodes_combination_to_min_distance_map_.end()) {
                return nodes_combination_to_min_distance_it->second;
            }
            // Base case: will only perform this calculation if the current node combination is not cached.
            auto min_distance = find_min_distance(query_node, reference_node);
            // Cache the results.
            nodes_combination_to_min_distance_map_.emplace(nodes_combination_key, min_distance);

            return min_distance;
        }

      private:
        template <typename QueryNodePtr>
        static constexpr auto* extract_raw_ptr(const QueryNodePtr& query_node) {
            if constexpr (std::is_pointer_v<QueryNodePtr>) {
                // For raw pointers.
                return query_node;
            } else {
                // For smart pointers.
                return query_node.get();
            }
        }

        template <typename QueryNodePtr>
        DistanceType find_max_furthest_distance(const QueryNodePtr& query_node,
                                                const DistanceType& init_queries_max_upper_bound = 0) const {
            auto queries_max_upper_bound = init_queries_max_upper_bound;

            for (const auto& query_index : *query_node) {
                const auto index_to_buffer_it = this_.query_to_buffer_map_.find(query_index);

                // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by
                // default We also don't need to recurse further since the lower-level nodes will never be greater than
                // infinity.
                if (index_to_buffer_it == this_.query_to_buffer_map_.cend()) {
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

        template <typename QueryNodePtr, typename ReferenceNodePtr>
        DistanceType find_min_distance(const QueryNodePtr& query_node, const ReferenceNodePtr& reference_node) const {
            auto min_distance = common::infinity<DistanceType>();

            for (const auto& query_index : *query_node) {
                const auto query_to_reference_min_distance = find_min_distance(query_index, reference_node);

                if (query_to_reference_min_distance < min_distance) {
                    min_distance = query_to_reference_min_distance;
                }
            }
            return min_distance;
        }

        template <typename ReferenceNodePtr>
        DistanceType find_min_distance(std::size_t query_index, const ReferenceNodePtr& reference_node) const {
            auto min_distance = common::infinity<DistanceType>();

            const auto query_features_range_first =
                this_.query_samples_range_first_ + query_index * this_.query_n_features_;

            const auto query_features_range_last =
                this_.query_samples_range_first_ + query_index * this_.query_n_features_ + this_.query_n_features_;

            for (const auto& reference_index : *reference_node) {
                const auto reference_features_range_first =
                    this_.reference_samples_range_first_ + reference_index * this_.reference_n_features_;

                const auto reference_features_range_last = this_.reference_samples_range_first_ +
                                                           reference_index * this_.reference_n_features_ +
                                                           this_.reference_n_features_;

                const auto query_to_reference_distance =
                    common::math::heuristics::auto_distance(query_features_range_first,
                                                            query_features_range_last,
                                                            reference_features_range_first,
                                                            reference_features_range_last);

                if (query_to_reference_distance < min_distance) {
                    min_distance = query_to_reference_distance;
                }
            }
            return min_distance;
        }

        const IndicesToBuffersMapType& this_;

        std::unordered_map<const void*, DistanceType> query_node_to_furthest_distance_map_;

        std::unordered_map<NodesCombinationKey, DistanceType> nodes_combination_to_min_distance_map_;
    };

    const QuerySamplesIterator& query_samples_range_first_;
    const QuerySamplesIterator& query_samples_range_last_;
    std::size_t                 query_n_features_;

    const ReferenceSamplesIterator& reference_samples_range_first_;
    const ReferenceSamplesIterator& reference_samples_range_last_;
    std::size_t                     reference_n_features_;

    IndexToBufferMapType query_to_buffer_map_ = IndexToBufferMapType{};

    IndexToBufferMapIterator tightest_query_to_buffer_it_ = query_to_buffer_map_.end();

    Cache cache_{*this};
};

template <typename Buffer, typename QuerySamplesIterator, typename ReferenceSamplesIterator>
IndicesToBuffersMap(const QuerySamplesIterator&,
                    const QuerySamplesIterator&,
                    std::size_t,
                    const ReferenceSamplesIterator&,
                    const ReferenceSamplesIterator&,
                    std::size_t) -> IndicesToBuffersMap<Buffer, QuerySamplesIterator, ReferenceSamplesIterator>;

}  // namespace ffcl::search::buffer