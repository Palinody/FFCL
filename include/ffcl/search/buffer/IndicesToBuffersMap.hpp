#pragma once

#include "ffcl/common/Utils.hpp"
#include "ffcl/common/math/heuristics/Distances.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>

namespace ffcl::search::buffer {

template <typename Index, typename Distance>
using Edge = std::tuple<Index, Index, Distance>;

template <typename Index, typename Distance>
constexpr auto make_edge(const Index& index_1, const Index& index_2, const Distance& distance) {
    return std::make_tuple(index_1, index_2, distance);
}

template <typename Buffer>
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

    constexpr auto&& tightest_edge() && {
        return std::move(tightest_edge_);
    }

    constexpr auto begin() -> IndexToBufferMapIterator {
        return index_to_buffer_map_.begin();
    }

    constexpr auto end() -> IndexToBufferMapIterator {
        return index_to_buffer_map_.end();
    }

    constexpr auto begin() const -> IndexToBufferMapConstIterator {
        return index_to_buffer_map_.begin();
    }

    constexpr auto end() const -> IndexToBufferMapConstIterator {
        return index_to_buffer_map_.end();
    }

    constexpr auto cbegin() const -> IndexToBufferMapConstIterator {
        return index_to_buffer_map_.cbegin();
    }

    constexpr auto cend() const -> IndexToBufferMapConstIterator {
        return index_to_buffer_map_.cend();
    }

    constexpr auto find(const IndexType& query_index) -> IndexToBufferMapIterator {
        return index_to_buffer_map_.find(query_index);
    }

    constexpr auto find(const IndexType& query_index) const -> IndexToBufferMapConstIterator {
        return index_to_buffer_map_.find(query_index);
    }

    template <typename FeaturesRangeIterator, typename... BufferArgs>
    constexpr auto find_or_make_buffer_at(const IndexType&             index,
                                          const FeaturesRangeIterator& features_range_first,
                                          const FeaturesRangeIterator& features_range_last,
                                          BufferArgs&&... buffer_args) -> IndexToBufferMapIterator {
        // Attempt to find the buffer associated with the current index in the buffer map.
        auto index_to_buffer_it = this->find(index);
        // If the current index does not have an associated buffer in the map,
        if (index_to_buffer_it == this->end()) {
            auto buffer = BufferType(/**/ features_range_first,
                                     /**/ features_range_last,
                                     /**/ std::forward<BufferArgs>(buffer_args)...);
            // Attempt to insert the newly created buffer into the map. If an element with the same
            // index already exists, emplace does nothing. Otherwise, it inserts the new element.
            // The method returns a pair, where the first element is an iterator to the inserted element
            // (or to the element that prevented the insertion) and the second element is a boolean
            // indicating whether the insertion took place.
            // We are only interested in the first element of the pair.
            index_to_buffer_it = this->emplace(index, std::move(buffer)).first;
        }
        return index_to_buffer_it;
    }

    constexpr auto emplace(const IndexType& index, BufferType&& buffer) -> std::pair<IndexToBufferMapIterator, bool> {
        return index_to_buffer_map_.emplace(index, std::move(buffer));
    }

    template <typename QueryIndicesIterator,
              typename QuerySamplesIterator,
              typename ReferenceIndicesIterator,
              typename ReferenceSamplesIterator,
              typename... BufferArgs>
    void partial_search_for_each_query(const QueryIndicesIterator&     query_indices_range_first,
                                       const QueryIndicesIterator&     query_indices_range_last,
                                       const QuerySamplesIterator&     query_samples_range_first,
                                       const QuerySamplesIterator&     query_samples_range_last,
                                       std::size_t                     query_n_features,
                                       const ReferenceIndicesIterator& reference_indices_range_first,
                                       const ReferenceIndicesIterator& reference_indices_range_last,
                                       const ReferenceSamplesIterator& reference_samples_range_first,
                                       const ReferenceSamplesIterator& reference_samples_range_last,
                                       std::size_t                     reference_n_features,
                                       BufferArgs&&... buffer_args) {
        common::ignore_parameters(query_samples_range_last);

        // Iterate through all query indices within the specified range of the query node.
        for (auto query_index_it = query_indices_range_first; query_index_it != query_indices_range_last;
             ++query_index_it) {
            // find_or_make_buffer_at returns a query_to_buffer_it and we are only interested in the buffer
            auto& buffer_at_query_index_reference =
                this->find_or_make_buffer_at(
                        *query_index_it,
                        query_samples_range_first + (*query_index_it) * query_n_features,
                        query_samples_range_first + (*query_index_it) * query_n_features + query_n_features,
                        std::forward<BufferArgs>(buffer_args)...)
                    ->second;

            // Regardless of whether the buffer was just inserted or already existed, perform a partial search
            // operation on the buffer. This operation updates the buffer based on a range of reference samples.
            buffer_at_query_index_reference.partial_search(reference_indices_range_first,
                                                           reference_indices_range_last,
                                                           reference_samples_range_first,
                                                           reference_samples_range_last,
                                                           reference_n_features);
            // Keep track of the current closest edge. Skip when the buffer is empty because the furthest distance
            // is initialized to zero by default. This makes sure that a sample doesn't connect to itself.
            if (buffer_at_query_index_reference.furthest_distance() < std::get<2>(tightest_edge_)) {
                tightest_edge_ = make_edge(*query_index_it,
                                           buffer_at_query_index_reference.furthest_index(),
                                           buffer_at_query_index_reference.furthest_distance());
            }
        }
    }

    template <typename QueryNodePtr,
              typename QuerySamplesIterator,
              typename ReferenceNodePtr,
              typename ReferenceSamplesIterator>
    auto cost(const QueryNodePtr&             query_node,
              const QuerySamplesIterator&     query_samples_range_first,
              const QuerySamplesIterator&     query_samples_range_last,
              std::size_t                     query_n_features,
              const ReferenceNodePtr&         reference_node,
              const ReferenceSamplesIterator& reference_samples_range_first,
              const ReferenceSamplesIterator& reference_samples_range_last,
              std::size_t                     reference_n_features) -> std::optional<DistanceType> {
        common::ignore_parameters(query_samples_range_last, reference_samples_range_last);

        const auto min_distance = find_min_distance(query_node,
                                                    query_samples_range_first,
                                                    query_samples_range_last,
                                                    query_n_features,
                                                    reference_node,
                                                    reference_samples_range_first,
                                                    reference_samples_range_last,
                                                    reference_n_features);

        if (reference_node->is_leaf() &&
            min_distance > /*queries_nodes_to_furthest_distance_cache_.get_cached_furthest_distance(query_node)*/
                search_query_node_furthest_distance(query_node)) {
            return std::nullopt;
        }
        // if (min_distance <= search_query_node_furthest_distance(query_node))
        return std::make_optional(min_distance);
    }

  private:
    template <typename QueryNodePtr>
    auto search_query_node_furthest_distance(const QueryNodePtr& query_node) const -> DistanceType {
        auto queries_max_upper_bound = DistanceType{0};

        for (auto query_index_it = query_node->indices_range_.first;
             query_index_it != query_node->indices_range_.second;
             ++query_index_it) {
            const auto index_to_buffer_it = this->find(*query_index_it);

            // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by default
            // We also don't need to recurse further since the lower-level nodes will never be greater than infinity.
            if (index_to_buffer_it == this->cend()) {
                return common::infinity<DistanceType>();

            }
            // If there's remaining space in the buffers, then candidates might potentially be further than the buffer's
            // current furthest distance.
            else if (index_to_buffer_it->second.remaining_capacity()) {
                return common::infinity<DistanceType>();

            } else {
                const auto query_buffer_furthest_distance = index_to_buffer_it->second.furthest_distance();

                queries_max_upper_bound = std::max(queries_max_upper_bound, query_buffer_furthest_distance);
            }
        }
        // For non-leaf nodes, recursively find the maximum upper bound in child nodes
        if (!query_node->is_leaf()) {
            queries_max_upper_bound = std::max({queries_max_upper_bound,
                                                search_query_node_furthest_distance(query_node->left_),
                                                search_query_node_furthest_distance(query_node->right_)});
        }
        return queries_max_upper_bound;
    }

    template <typename QueryNodePtr,
              typename QuerySamplesIterator,
              typename ReferenceNodePtr,
              typename ReferenceSamplesIterator>
    static auto find_min_distance(const QueryNodePtr&             query_node,
                                  const QuerySamplesIterator&     query_samples_range_first,
                                  const QuerySamplesIterator&     query_samples_range_last,
                                  std::size_t                     query_n_features,
                                  const ReferenceNodePtr&         reference_node,
                                  const ReferenceSamplesIterator& reference_samples_range_first,
                                  const ReferenceSamplesIterator& reference_samples_range_last,
                                  std::size_t                     reference_n_features) -> DistanceType {
        common::ignore_parameters(query_samples_range_last, reference_samples_range_last);

        auto min_distance = common::infinity<DistanceType>();

        for (auto query_index_it = query_node->indices_range_.first;
             query_index_it != query_node->indices_range_.second;
             ++query_index_it) {
            for (auto reference_index_it = reference_node->indices_range_.first;
                 reference_index_it != reference_node->indices_range_.second;
                 ++reference_index_it) {
                const auto query_to_reference_distance = common::math::heuristics::auto_distance(
                    query_samples_range_first + (*query_index_it) * query_n_features,
                    query_samples_range_first + (*query_index_it) * query_n_features + query_n_features,
                    reference_samples_range_first + (*reference_index_it) * reference_n_features,
                    reference_samples_range_first + (*reference_index_it) * reference_n_features +
                        reference_n_features);

                min_distance = std::min(min_distance, query_to_reference_distance);
            }
        }
        return min_distance;
    }

    class QueriesNodesToFurthestDistanceCache {
      public:
        QueriesNodesToFurthestDistanceCache(const IndicesToBuffersMap<BufferType>& parent_map)
          : parent_map_(parent_map) {}

        // Method to retrieve or calculate the furthest distance for a given node.
        template <typename QueryNodePtr>
        DistanceType get_cached_furthest_distance(const QueryNodePtr& query_node) {
            const auto* raw_query_node = extract_raw_ptr(query_node);

            auto it = cache_.find(raw_query_node);

            if (it == cache_.end()) {
                return search_and_try_cache_max_furthest_distance(query_node);

            } else {
                if (!query_node->is_leaf()) {
                    return std::max({it->second,
                                     get_cached_furthest_distance(query_node->left_),
                                     get_cached_furthest_distance(query_node->right_)});

                } else {
                    return it->second;
                }
            }
        }

      private:
        template <typename QueryNodePtr>
        static const auto* extract_raw_ptr(const QueryNodePtr& query_node) {
            if constexpr (std::is_pointer_v<QueryNodePtr>) {
                // For raw pointers.
                return query_node;
            } else {
                // For smart pointers.
                return query_node.get();
            }
        }

        template <typename QueryNodePtr>
        DistanceType search_and_try_cache_max_furthest_distance(const QueryNodePtr& query_node) {
            auto queries_max_upper_bound = DistanceType{0};

            for (auto query_index_it = query_node->indices_range_.first;
                 query_index_it != query_node->indices_range_.second;
                 ++query_index_it) {
                const auto index_to_buffer_it = parent_map_.find(*query_index_it);

                // If the buffer at the current index wasn't initialized, then its furthest distance is infinity by
                // default We also don't need to recurse further since the lower-level nodes will never be greater than
                // infinity.
                if (index_to_buffer_it == parent_map_.cend()) {
                    return common::infinity<DistanceType>();

                }
                // If there's remaining space in the buffers, then candidates might potentially be further than the
                // buffer's current furthest distance.
                else if (index_to_buffer_it->second.remaining_capacity()) {
                    return common::infinity<DistanceType>();

                } else {
                    const auto query_buffer_furthest_distance = index_to_buffer_it->second.furthest_distance();

                    queries_max_upper_bound = std::max(queries_max_upper_bound, query_buffer_furthest_distance);
                }
            }
            cache_.emplace(query_node.get(), queries_max_upper_bound);
            return queries_max_upper_bound;
        }

        const IndicesToBuffersMap<BufferType>& parent_map_;

        std::unordered_map<const void*, DistanceType> cache_;
    };

    QueriesNodesToFurthestDistanceCache queries_nodes_to_furthest_distance_cache_{*this};

    IndexToBufferMapType index_to_buffer_map_;

    // query_index in the query set, reference_index in the reference set and their distance
    Edge<IndexType, DistanceType> tightest_edge_ =
        make_edge(common::infinity<IndexType>(), common::infinity<IndexType>(), common::infinity<DistanceType>());
};

}  // namespace ffcl::search::buffer