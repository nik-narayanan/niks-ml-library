//
// Created by nik on 3/31/2024.
//

#ifndef NML_APPROXIMATE_NEAREST_NEIGHBOR_H
#define NML_APPROXIMATE_NEAREST_NEIGHBOR_H

#include <cmath>
#include <thread>

#include "similarity_distance.h"
#include "../primitives/span.h"
#include "../primitives/random.h"
#include "../primitives/bitset.h"
#include "../primitives/grid_span.h"
#include "../primitives/matrix_span.h"
#include "../primitives/heap_min_max.h"

namespace nml::approximate_nearest_neighbor_internal
{
    struct Node;

    class NodeContainer
    {
        char* _data;

    public:

        unsigned length;
        unsigned node_size;

        explicit NodeContainer(MemorySpan& memory, unsigned node_size);
        Node& operator[](unsigned offset) noexcept;
    };
}

namespace nml
{
    using namespace approximate_nearest_neighbor_internal;

    struct ApproximateNearestNeighborIndex
    {
        NodeContainer _tree;
        unsigned _row_count;
        unsigned _node_children_count;
        unsigned _max_descendant_count;
        Span<unsigned> _tree_root_indexes;

        struct BuildRequest
        {
            const MatrixSpan& matrix; const unsigned tree_count; const unsigned seed; const unsigned thread_count;

            explicit BuildRequest(const MatrixSpan& matrix, unsigned tree_count, unsigned thread_count, unsigned seed) noexcept
                : matrix(matrix), tree_count(tree_count), seed(seed)
                , thread_count(thread_count == 0 ? std::max(1u, std::thread::hardware_concurrency()) : thread_count)
            { }
        };

        [[nodiscard]] static RequiredMemory build_required_memory(const BuildRequest& request) noexcept;
        static NMLResult<ApproximateNearestNeighborIndex> build(const BuildRequest& request, const RequestMemory& memory) noexcept;

        struct FindRequest
        {
            unsigned index; const unsigned neighbor_count; const unsigned nodes_to_check;

            explicit FindRequest(unsigned index, unsigned neighbor_count, unsigned nodes_to_check) noexcept
                : index(index), neighbor_count(neighbor_count), nodes_to_check(nodes_to_check)
            { }
        };

        [[nodiscard]] RequiredMemory find_required_memory(FindRequest& request) const noexcept;
        [[nodiscard]] static RequiredMemory find_required_memory(FindRequest& request, unsigned tree_count, unsigned row_count) noexcept;
        NMLResult<Span<ScoredValue<unsigned>>> find_nearest_neighbors(FindRequest& request, RequestMemory& memory) noexcept;

        struct BulkFindRequest
        {
            const unsigned neighbor_count; const unsigned nodes_to_check; const unsigned thread_count;

            explicit BulkFindRequest(unsigned neighbor_count, unsigned nodes_to_check, unsigned thread_count) noexcept
                : neighbor_count(neighbor_count), nodes_to_check(nodes_to_check)
                , thread_count(thread_count == 0 ? std::max(1u, std::thread::hardware_concurrency()) : thread_count)
            { }
        };

        [[nodiscard]] RequiredMemory bulk_find_required_memory(const BulkFindRequest& request) const noexcept;
        [[nodiscard]] static RequiredMemory bulk_find_required_memory(const BulkFindRequest& request, unsigned tree_count, unsigned row_count) noexcept;
        NMLResult<GridSpan<ScoredValue<unsigned>>> bulk_find_nearest_neighbors(const BulkFindRequest& request, RequestMemory& memory) noexcept;
    };
}

namespace nml::approximate_nearest_neighbor_internal
{
    using Index = ApproximateNearestNeighborIndex;

    struct Node
    {
        Node() noexcept : _values(0), left_index(0), right_index(0), plane_offset(0), descendants_count(1) { }

        Node(const Node& other) noexcept = delete;
        Node& operator=(const Node &) noexcept = delete;
        Node& operator=(const Node &&) noexcept = delete;

        inline void initialize_shallow() noexcept
        {
            left_index = 0;
            right_index = 0;
            plane_offset = 0;
            descendants_count = 1;
        }

        void initialize_deep(const VectorSpan& values) noexcept
        {
            initialize_shallow();

            auto local_values = get_values(values.length);
            local_values.copy_from_unsafe(values);
        }

        void copy(Node& other, const unsigned children_count) noexcept
        {
            left_index = other.left_index;
            right_index = other.right_index;
            plane_offset = other.plane_offset;
            descendants_count = other.descendants_count;

            auto local_values = get_values(children_count);
            auto other_values = other.get_values(children_count);
            local_values.copy_from_unsafe(other_values);
        }

        static inline unsigned get_size(const unsigned children_count) noexcept
        {
            return sizeof(Node) + sizeof(float) * (children_count - 1);
        }

        VectorSpan get_values(unsigned children_count) noexcept
        {
            return VectorSpan(reinterpret_cast<float*>(&_values), children_count);
        }

        Span<unsigned> get_leaf_indexes() noexcept
        {
            return Span<unsigned>(&left_index, descendants_count);
        }

        void fill_leaf_indexes(Span<unsigned>& indexes) noexcept
        {
            memcpy(&left_index, indexes.get_pointer(0), indexes.bytes());
        }

        unsigned descendants_count;
        float plane_offset;
        unsigned left_index;
        unsigned right_index;

    private:
        float _values;
    };

    NodeContainer::NodeContainer(MemorySpan& memory, unsigned node_size)
        : node_size(node_size)
        , length(memory.bytes / node_size)
        , _data(memory.get_pointer<char>(0))
    {}

    Node& NodeContainer::operator[](const unsigned offset) noexcept
    {
        return *reinterpret_cast<Node*>(&_data[offset * node_size]);
    }

    /*******************************************************************************************************************
     BUILD APPROXIMATE NEAREST NEIGHBOR INDEX
     ******************************************************************************************************************/

    struct IndexBuilder
    {
        unsigned _row_count;
        NodeContainer _tree;
        NodeContainer _split_nodes;
        unsigned _node_children_count;
        Span<unsigned> _tree_root_indexes;
        const unsigned _max_descendant_count;
        std::atomic<unsigned> _tree_node_count; // TODO make atomic

        struct WorkingMemory { NodeContainer split_nodes; Span<unsigned> index_buffer; };
        struct ResultMemory { NodeContainer tree; Span<unsigned> tree_root_indexes; };

        [[nodiscard]] static RequiredMemory build_required_memory(const Index::BuildRequest& request) noexcept;
        static NMLResult<Index> build_index(const Index::BuildRequest& request, const RequestMemory& memory) noexcept;
        IndexBuilder(const Index::BuildRequest& request, const WorkingMemory& working, const ResultMemory& result) noexcept;

    private:

        unsigned get_next_working_node_offset() noexcept;

        void build_tree(unsigned root_offset, Span<unsigned>& index_span, Random& random, unsigned thread) noexcept;
        unsigned make_tree(Span<unsigned>& tree_offsets, bool is_root, Random& random, unsigned thread) noexcept;
        void create_split_plane(Span<unsigned>& tree_offsets, Node& parent, Random& random, unsigned thread) noexcept;
        unsigned split_tree_inplace(Span<unsigned>& tree_offsets, Node& split_node, Random& random, unsigned thread) noexcept;

        bool check_side(Node& parent, Node& next, Random& random) const noexcept;
        static void update_mean(VectorSpan& mean_node_values, VectorSpan& new_node_values, float count) noexcept;
    };

    RequiredMemory IndexBuilder::build_required_memory(const Index::BuildRequest& request) noexcept
    {
        auto _indexes = sizeof(unsigned) * request.matrix.row_ct * request.thread_count;
        auto _tree_root_indexes_size = sizeof(unsigned) * request.tree_count;

        auto _split_nodes_size = Node::get_size(request.matrix.column_ct) * 2 * request.thread_count;
        auto _tree_size = Node::get_size(request.matrix.column_ct) * request.matrix.row_ct * (request.tree_count + 1);

        return RequiredMemory
        {
            .result_required_bytes = static_cast<unsigned>(_tree_root_indexes_size + _tree_size),
            .working_required_bytes = static_cast<unsigned>(_split_nodes_size + _indexes)
        };
    }

    NMLResult<Index> IndexBuilder::build_index(const Index::BuildRequest& request, const RequestMemory& memory) noexcept
    {
        if (!memory.is_sufficient(IndexBuilder::build_required_memory(request)))
        {
            return NMLResult<ApproximateNearestNeighborIndex>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        auto node_size = Node::get_size(request.matrix.column_ct);

        unsigned result_memory_byte_offset = 0;

        auto required_tree_nodes = (1 + request.tree_count) * request.matrix.row_ct;
        auto tree_memory = memory.result_memory.to_subspan_unsafe(result_memory_byte_offset, node_size * required_tree_nodes);

        result_memory_byte_offset += tree_memory.bytes;

        const ResultMemory result_memory =
        {
            .tree = NodeContainer(tree_memory, node_size),
            .tree_root_indexes = Span<unsigned>(memory.result_memory.get_pointer<unsigned>(result_memory_byte_offset), request.tree_count)
        };

        unsigned working_memory_byte_offset = 0;

        auto split_nodes_memory = memory.working_memory.to_subspan_unsafe(working_memory_byte_offset, node_size * 2 * request.thread_count);

        working_memory_byte_offset += split_nodes_memory.bytes;

        const WorkingMemory working_memory =
        {
            .split_nodes = NodeContainer(split_nodes_memory, node_size),
            .index_buffer = Span<unsigned>(memory.working_memory.get_pointer<unsigned>(working_memory_byte_offset), request.matrix.row_ct * request.thread_count)
        };

        auto builder = IndexBuilder(request, working_memory, result_memory);

        unsigned thread_count = request.thread_count;
        // todo for each thread;
        for (unsigned root = 0; root < request.tree_count; ++root)
        {
            auto thread = 0;
            auto random = Random(request.seed + root);
            auto index_buffer = working_memory.index_buffer.to_subspan_unsafe(
                request.matrix.row_ct * thread,
                request.matrix.row_ct
            );

            builder.build_tree(root, index_buffer, random, thread);
        }

        return NMLResult<Index>::ok({
            ._tree = builder._tree,
            ._row_count = builder._row_count,
            ._node_children_count = builder._node_children_count,
            ._max_descendant_count = builder._max_descendant_count,
            ._tree_root_indexes = builder._tree_root_indexes,
        });
    }

    IndexBuilder::IndexBuilder(const Index::BuildRequest& request, const WorkingMemory& working, const ResultMemory& result) noexcept
        : _tree(result.tree)
        , _split_nodes(working.split_nodes)
        , _tree_root_indexes(result.tree_root_indexes)
        , _row_count(request.matrix.row_ct)
        , _tree_node_count(request.matrix.row_ct) // TODO make atomic
        , _node_children_count(request.matrix.column_ct)
        , _max_descendant_count(request.matrix.column_ct + 2)
    {
        for (unsigned row = 0; row < request.matrix.row_ct; ++row)
        {
            VectorSpan matrix_span = request.matrix.to_vector_subspan_unsafe(request.matrix.column_ct, row, 0);

            _tree[row].initialize_deep(matrix_span);
        }
    }

    void IndexBuilder::build_tree(unsigned root_offset, Span<unsigned>& index_span, Random& random, const unsigned thread) noexcept
    {
        for (unsigned index = 0; index < _row_count; ++index)
        {
            index_span[index] = index;
        }

        _tree_root_indexes[root_offset] = make_tree(index_span, true, random, thread);
    }

    unsigned IndexBuilder::make_tree(Span<unsigned>& tree_offsets, bool is_root, Random& random, const unsigned thread) noexcept
    {
        if (tree_offsets.length == 1 && !is_root) return tree_offsets[0];

        if (tree_offsets.length <= _max_descendant_count && (!is_root || _row_count <= _max_descendant_count || tree_offsets.length == 1))
        {
            unsigned next_index = get_next_working_node_offset();

            auto& leaf_node = _tree[next_index];

            leaf_node.descendants_count = tree_offsets.length;

            leaf_node.fill_leaf_indexes(tree_offsets);

            return next_index;
        }

        unsigned next_index = get_next_working_node_offset();

        auto& split_node = _tree[next_index];

        split_node.descendants_count = tree_offsets.length;

        unsigned split_index = split_tree_inplace(tree_offsets, split_node, random, thread);

        Span<unsigned> left_children = tree_offsets.to_subspan_unsafe(0, split_index),
                       right_children = tree_offsets.to_subspan_unsafe(split_index);

        if (left_children.length <= right_children.length)
        {
            split_node.left_index = make_tree(left_children, false, random, thread);
            split_node.right_index = make_tree(right_children, false, random, thread);
        }
        else
        {
            split_node.right_index = make_tree(right_children, false, random, thread);
            split_node.left_index = make_tree(left_children, false, random, thread);
        }

        return next_index;
    }

    unsigned IndexBuilder::split_tree_inplace(Span<unsigned>& tree_offsets, Node& split_node, Random& random, unsigned thread) noexcept
    {
        unsigned last_left, last_left_checked, left_index, right_index;

        float split_balance;

        for (unsigned attempt = 0; attempt < 3; ++attempt)
        {
            last_left = 0, last_left_checked = 0;
            left_index = 0, right_index = tree_offsets.length - 1;

            create_split_plane(tree_offsets, split_node, random, thread);

            while (left_index < right_index)
            {
                bool continue_left = true, continue_right = true;

                while (left_index < right_index && continue_left)
                {
                    last_left_checked = left_index;
                    unsigned next_index = tree_offsets[left_index];
                    auto& next = _tree[next_index];

                    bool side = check_side(split_node, next, random);

                    if (!side) last_left = left_index++;
                    else continue_left = false;
                }

                while (left_index < right_index && continue_right)
                {
                    unsigned next_index = tree_offsets[right_index];
                    auto& next = _tree[next_index];

                    bool side = check_side(split_node, next, random);

                    if (side) --right_index;
                    else continue_right = false;
                }

                if (left_index < right_index)
                {
                    last_left_checked = last_left = left_index;
                    std::swap(tree_offsets[left_index++], tree_offsets[right_index--]);
                }
            }

            if (left_index < tree_offsets.length && last_left_checked != left_index)
            {
                unsigned next_index = tree_offsets[left_index];
                auto& next = _tree[next_index];

                bool side = check_side(split_node, next, random);
                if (!side) last_left = left_index;
            }

            auto balance = static_cast<float>(last_left) / static_cast<float>(tree_offsets.length);

            split_balance = std::max(balance, 1 - balance);

            if (split_balance < 0.95) break;
        }

        while (split_balance > 0.99)
        {
            last_left = 0, last_left_checked = 0;
            left_index = 0, right_index = tree_offsets.length - 1;

            split_node.get_values(_node_children_count).fill(0);

            while (left_index < right_index)
            {
                bool continue_left = true, continue_right = true;

                while (left_index < right_index && continue_left)
                {
                    if (!random.flip()) ++left_index;
                    else continue_left = false;
                }

                while (left_index < right_index && continue_right)
                {
                    if (random.flip()) --right_index;
                    else continue_right = false;
                }

                if (left_index < right_index)
                {
                    std::swap(tree_offsets[left_index++], tree_offsets[right_index--]);
                }
            }

            if (left_index < tree_offsets.length && last_left_checked != left_index)
            {
                bool side = random.flip();
                if (!side) last_left = left_index;
            }

            auto balance = static_cast<float>(last_left) / static_cast<float>(tree_offsets.length);

            split_balance = std::max(balance, 1 - balance);
        }

        return last_left + 1;
    }

    bool IndexBuilder::check_side(Node& parent, Node& next, Random& random) const noexcept
    {
        float dot = 0;

        VectorSpan next_values = next.get_values(_node_children_count);
        VectorSpan parent_values = parent.get_values(_node_children_count);

        for (unsigned column = 0; column < parent_values.length; ++column) // TODO vector_cuda
        {
            dot += parent_values[column] * next_values[column];
        }

        float margin = parent.plane_offset + dot;

        if (margin != 0) return margin > 0;

        return random.flip();
    }

    void IndexBuilder::create_split_plane(Span<unsigned>& tree_offsets, Node& parent, Random& random, const unsigned thread) noexcept
    {
        unsigned left_index = random.index(tree_offsets.length);
        unsigned right_index = random.index(tree_offsets.length - 1);
        right_index += (right_index >= left_index);

        Node& left = _split_nodes[0 + thread];
        Node& right = _split_nodes[1 + thread];

        left.copy(_tree[tree_offsets[left_index]], _node_children_count);
        right.copy(_tree[tree_offsets[right_index]], _node_children_count);

        VectorSpan left_values = left.get_values(_node_children_count);
        VectorSpan right_values = right.get_values(_node_children_count);
        VectorSpan parent_values = parent.get_values(_node_children_count);

        const unsigned max_iterations = 200;

        float left_count = 1, right_count = 1;

        for (int iteration = 0; iteration < max_iterations; ++iteration)
        {
            unsigned split_index = random.index(tree_offsets.length);

            auto& split_node = _tree[tree_offsets[split_index]];

            VectorSpan split_node_values = split_node.get_values(_node_children_count);

            float left_distance = left_count * euclidean_distance(left_values, split_node_values),
                    right_distance = right_count * euclidean_distance(right_values, split_node_values);

            if (left_distance < right_distance)
            {
                update_mean(left_values, split_node_values, left_count++);
            }
            else if (right_distance < left_distance)
            {
                update_mean(right_values, split_node_values, right_count++);
            }
        }

        for (unsigned column = 0; column < parent_values.length; ++column)
        {
            parent_values[column] = left_values[column] - right_values[column];
        }

        parent_values.normalize();
        parent.plane_offset = 0.0;

        for (unsigned column = 0; column < parent_values.length; ++column)
        {
            parent.plane_offset += -parent_values[column] * (left_values[column] + right_values[column]) / 2;
        }
    }

    void IndexBuilder::update_mean(VectorSpan& mean_node_values, VectorSpan& new_node_values, const float count) noexcept
    {
        for (unsigned column = 0; column < mean_node_values.length; ++column) // todo vector_cuda
        {
            float mean_node_value = mean_node_values[column];
            float new_node_value = new_node_values[column];

            float new_value = count * mean_node_value + new_node_value;
            mean_node_values[column] = new_value / (count + 1);
        }
    }

    unsigned IndexBuilder::get_next_working_node_offset() noexcept
    {
        unsigned next_index = _tree_node_count++; // TODO make atomic
        _tree[next_index].initialize_shallow();
        return next_index;
    }

    /*******************************************************************************************************************
     FIND APPROXIMATE NEAREST NEIGHBORS
     ******************************************************************************************************************/

    struct NodeFinder
    {
        Node& _node;
        Index& _index;
        Bitset _bitset;
        unsigned _scored_counter;
        MinMaxHeap<ScoredValue<unsigned>> _node_queue;
        MinMaxHeap<ScoredValue<unsigned>> _scored_queue;

        void enqueue_scored_node(unsigned next_node_tree_offset) noexcept;

        static NodeFinder initialize(const Index::FindRequest& request, const RequestMemory& memory, Index& index) noexcept;
        static RequiredMemory required_memory(Index::FindRequest& request, unsigned tree_count, unsigned row_count) noexcept;
        static NMLResult<Span<ScoredValue<unsigned>>> find_nearest_neighbors(Index::FindRequest& request, RequestMemory& memory, Index& index) noexcept;
    };

    NodeFinder NodeFinder::initialize(const Index::FindRequest &request, const RequestMemory &memory, Index &index) noexcept
    {
        unsigned working_bytes_offset = 0;

        auto bitset_required_bytes= Bitset::required_bytes(index._row_count);
        auto bitset_memory = memory.working_memory.to_subspan_unsafe(0, bitset_required_bytes);

        working_bytes_offset += bitset_required_bytes;

        auto node_check_queue_required_bytes = MinMaxHeap<ScoredValue<unsigned>>::required_bytes(index._tree_root_indexes.length + index._row_count);
        auto node_check_queue_memory = memory.working_memory.to_subspan_unsafe(working_bytes_offset, node_check_queue_required_bytes);

        working_bytes_offset += node_check_queue_required_bytes;

        auto scored_queue_required_bytes = MinMaxHeap<ScoredValue<unsigned>>::required_bytes(request.neighbor_count);
        auto scored_queue_memory = memory.working_memory.to_subspan_unsafe(working_bytes_offset, scored_queue_required_bytes);

        return NodeFinder
        {
            ._node = index._tree[request.index],
            ._index = index,
            ._bitset = Bitset(bitset_memory),
            ._scored_counter = 0,
            ._node_queue = MinMaxHeap<ScoredValue<unsigned>>(node_check_queue_memory),
            ._scored_queue = MinMaxHeap<ScoredValue<unsigned>>(scored_queue_memory),
        };
    }

    RequiredMemory NodeFinder::required_memory(Index::FindRequest &request, unsigned tree_count, unsigned row_count) noexcept
    {
        return RequiredMemory
        {
            .result_required_bytes = static_cast<unsigned>(sizeof(ScoredValue<unsigned>)) * request.neighbor_count,
            .working_required_bytes = Bitset::required_bytes(row_count)
                                      + MinMaxHeap<ScoredValue<unsigned>>::required_bytes(tree_count + row_count)
                                      + MinMaxHeap<ScoredValue<unsigned>>::required_bytes(request.neighbor_count)
        };
    }

    void NodeFinder::enqueue_scored_node(unsigned int next_node_tree_offset) noexcept
    {
        if (_bitset.check_unsafe(next_node_tree_offset)) return;
        _bitset.set_unsafe(next_node_tree_offset);

        _scored_counter++;

        auto& next_node = _index._tree[next_node_tree_offset];

        auto node_values = _node.get_values(_index._node_children_count);
        auto next_node_values = next_node.get_values(_index._node_children_count);

        float distance = euclidean_distance(node_values, next_node_values);
        _scored_queue.insert_min({distance, next_node_tree_offset});
    }

    NMLResult<Span<ScoredValue<unsigned>>> NodeFinder::find_nearest_neighbors(Index::FindRequest& request, RequestMemory& memory, Index& index) noexcept
    {
        auto _required_memory = required_memory(request, index._tree_root_indexes.length, index._row_count);

        bool insufficient_result_memory = memory.result_memory.bytes < _required_memory.result_required_bytes;
        bool insufficient_working_memory = memory.working_memory.bytes < _required_memory.working_required_bytes;

        if (insufficient_result_memory || insufficient_working_memory)
        {
            return NMLResult<Span<ScoredValue<unsigned>>>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        auto& node = index._tree[request.index];

        unsigned nodes_to_check = request.nodes_to_check > 0 ?
                                  request.nodes_to_check : request.neighbor_count * index._tree_root_indexes.length;

        auto find_state = NodeFinder::initialize(request, memory, index);

        find_state._bitset.set_unsafe(request.index);

        for (unsigned root = 0; root < index._tree_root_indexes.length; ++root)
        {
            const unsigned tree_root_index = index._tree_root_indexes[root];

            find_state._node_queue.insert_min({std::numeric_limits<float>::max(), tree_root_index});
        }

        while (find_state._scored_counter < nodes_to_check && !find_state._node_queue.is_empty())
        {
            const auto next = find_state._node_queue.remove_max_unsafe();
            auto& next_node = index._tree[next.value];

            if (next_node.descendants_count == 1 && next.value < index._row_count)
            {
                find_state.enqueue_scored_node(next.value);
            }
            else if (next_node.descendants_count <= index._max_descendant_count)
            {
                auto leaf_indexes = next_node.get_leaf_indexes();

                for (unsigned leaf_index = 0; leaf_index < leaf_indexes.length; ++leaf_index)
                {
                    find_state.enqueue_scored_node(leaf_indexes[leaf_index]);
                }
            }
            else
            {
                auto node_values = node.get_values(index._node_children_count);
                auto next_node_values = next_node.get_values(index._node_children_count);

                float margin = next_node.plane_offset + next_node_values.dot_product_unsafe(node_values);

                find_state._node_queue.insert_max({std::min(next.score, margin), next_node.right_index});
                find_state._node_queue.insert_max({std::min(next.score, -margin), next_node.left_index});
            }
        }

        unsigned result_count = 0;

        auto result = Span<ScoredValue<unsigned>>(memory.result_memory, 0);

        while (!find_state._scored_queue.is_empty())
        {
            result[result_count++] = find_state._scored_queue.remove_min_unsafe();
        }

        return NMLResult<Span<ScoredValue<unsigned>>>::ok(result);
    }
}

namespace nml
{
    RequiredMemory Index::find_required_memory(FindRequest& request) const noexcept
    {
        return Index::find_required_memory(request, _tree_root_indexes.length, _row_count);
    }

    RequiredMemory Index::find_required_memory(FindRequest& request, unsigned tree_count, unsigned row_count) noexcept
    {
        return NodeFinder::required_memory(request, tree_count, row_count);
    }

    NMLResult<Span<ScoredValue<unsigned>>> Index::find_nearest_neighbors(FindRequest& request, RequestMemory& memory) noexcept
    {
        return NodeFinder::find_nearest_neighbors(request, memory, *this);
    }

    RequiredMemory Index::build_required_memory(const BuildRequest& request) noexcept
    {
        return IndexBuilder::build_required_memory(request);
    }

    NMLResult<Index> Index::build(const BuildRequest& request, const RequestMemory& memory) noexcept
    {
        return IndexBuilder::build_index(request, memory);
    }

    RequiredMemory Index::bulk_find_required_memory(const BulkFindRequest& request) const noexcept
    {
        return Index::bulk_find_required_memory(request, _tree_root_indexes.length, _row_count);
    }

    RequiredMemory Index::bulk_find_required_memory(const BulkFindRequest& request, unsigned tree_count, unsigned row_count) noexcept
    {
        auto find_request = Index::FindRequest(0, request.neighbor_count, 0);

        auto required_memory = find_required_memory(find_request, tree_count, row_count);

        return RequiredMemory
        {
            .result_required_bytes = required_memory.result_required_bytes * row_count,
            .working_required_bytes = required_memory.working_required_bytes * request.thread_count
        };
    }

    NMLResult<GridSpan<ScoredValue<unsigned>>> Index::bulk_find_nearest_neighbors(const BulkFindRequest& request, RequestMemory& memory) noexcept
    {
        if (!memory.is_sufficient(bulk_find_required_memory(request, _tree_root_indexes.length, _row_count)))
        {
            return NMLResult<GridSpan<ScoredValue<unsigned>>>::err(NMLErrorCode::INSUFFICIENT_MEMORY);
        }

        auto result = GridSpan<ScoredValue<unsigned>>(memory.result_memory.get_pointer<ScoredValue<unsigned>>(0), _row_count, request.neighbor_count);

        auto find_request = Index::FindRequest(0, request.neighbor_count, 0);

        unsigned thread_count = request.thread_count;
        // todo for each thread;

        unsigned thread = 0;
        unsigned memory_block_size = memory.working_memory.bytes / thread_count;

        auto thread_working_memory = memory.working_memory.to_subspan_unsafe(memory_block_size * thread, memory_block_size);

        for (unsigned row = 0; row < _row_count; ++row)
        {
            find_request.index = row;

            auto find_request_memory = RequestMemory
            {
                .result_memory = result[row].to_memory_unsafe(),
                .working_memory = thread_working_memory,
            };

            auto nearest_neighbors_result = find_nearest_neighbors(find_request, find_request_memory);

            if (nearest_neighbors_result.is_err())
            {
                return NMLResult<GridSpan<ScoredValue<unsigned>>>::err(nearest_neighbors_result.err());
            }
        }

        return NMLResult<GridSpan<ScoredValue<unsigned>>>::ok(result);
    }
}

#endif //NML_APPROXIMATE_NEAREST_NEIGHBOR_H