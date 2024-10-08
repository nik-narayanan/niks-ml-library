//
// Created by nik on 4/1/2024.
//

#ifndef NML_HEAP_MIN_MAX_H
#define NML_HEAP_MIN_MAX_H

#include "memory_span.h"
#include "scored_value.h"
#include "matrix_result.h"

namespace nml
{
    template<typename TValue>
    class MinMaxHeap
    {
        TValue* _heap;
        uint64_t _size{};
        uint64_t _capacity{};

    public:

        explicit MinMaxHeap(MemorySpan memory, uint64_t capacity = 0) noexcept
            : _size(0)
            , _heap(memory.get_pointer<TValue>(0))
            , _capacity(capacity == 0 ? memory.bytes / sizeof(TValue) : capacity)
        { }

        static inline uint64_t required_bytes(uint64_t queue_size) noexcept { return queue_size * sizeof(TValue); }

        void print() const;
        void clear() noexcept { _size = 0; }
        [[nodiscard]] uint64_t size() const noexcept { return _size; }
        [[nodiscard]] uint64_t memory_size() const noexcept { return sizeof(TValue) * _capacity; }

        void insert_min(TValue value) noexcept;
        void insert_max(TValue value) noexcept;

        TValue remove_min_unsafe() noexcept;
        TValue remove_max_unsafe() noexcept;
        NMLResult<TValue> remove_min() noexcept;
        NMLResult<TValue> remove_max() noexcept;

        bool contains(TValue value) const noexcept;

        [[nodiscard]] NMLResult<TValue> peek_min() const noexcept;
        [[nodiscard]] NMLResult<TValue> peek_max() const noexcept;
        [[nodiscard]] inline TValue peek_max_unsafe() const noexcept;
        [[nodiscard]] inline TValue peek_min_unsafe() const noexcept;

        [[nodiscard]] bool is_empty() const noexcept { return _size == 0; }

    private:

        void pull_up(uint64_t index) noexcept;
        void push_down(uint64_t index) noexcept;
        void delete_element(uint64_t index) noexcept;

        uint64_t find_min_index() noexcept;
        static bool is_min_level(uint64_t index) noexcept;
        static inline uint64_t left_child_index(uint64_t index) noexcept { return 2 * index + 1; }
        static inline uint64_t right_child_index(uint64_t index) noexcept { return 2 * index + 2; }
        static inline uint64_t parent_index(uint64_t index) noexcept { return index == 0 ? 0 : (index - 1) / 2; }
        static inline uint64_t grandparent_index(uint64_t index) noexcept { return parent_index(parent_index(index)); }
    };

    template<typename TValue>
    void MinMaxHeap<TValue>::insert_min(TValue value) noexcept
    {
        if (_size + 1 > _capacity)
        {
            auto max_score = _heap[0];

            if (value >= max_score) return;

            delete_element(0);
        }

        _heap[_size] = value;
        pull_up(_size++);
    }

    template<typename TValue>
    void MinMaxHeap<TValue>::insert_max(TValue value) noexcept
    {
        if (_size + 1 > _capacity)
        {
            auto min_index = find_min_index();
            auto min_score = _heap[min_index];

            if (value <= min_score) return;

            delete_element(min_index);
        }

        _heap[_size] = value;
        pull_up(_size++);
    }

    template<typename TValue>
    NMLResult<TValue> MinMaxHeap<TValue>::peek_min() const noexcept
    {
        if (_size == 0) return NMLResult<ScoredValue<TValue>>::err(NMLErrorCode::OUT_OF_BOUNDS);

        return NMLResult<ScoredValue<TValue>>::ok(peek_min_unsafe());
    }

    template<typename TValue>
    NMLResult<TValue> MinMaxHeap<TValue>::peek_max() const noexcept
    {
        if (_size == 0) return NMLResult<ScoredValue<TValue>>::err(NMLErrorCode::OUT_OF_BOUNDS);

        return NMLResult<ScoredValue<TValue>>::ok(peek_max_unsafe());
    }

    template<typename TValue>
    TValue MinMaxHeap<TValue>::peek_max_unsafe() const noexcept
    {
        return _heap[0];
    }

    template<typename TValue>
    TValue MinMaxHeap<TValue>::peek_min_unsafe() const noexcept
    {
        return _heap[find_min_index()];
    }

    template<typename TValue>
    NMLResult<TValue> MinMaxHeap<TValue>::remove_min() noexcept
    {
        if (_size == 0) return NMLResult<ScoredValue<TValue>>::err(NMLErrorCode::OUT_OF_BOUNDS);

        return NMLResult<ScoredValue<TValue>>::ok(remove_min_unsafe());
    }

    template<typename TValue>
    NMLResult<TValue> MinMaxHeap<TValue>::remove_max() noexcept
    {
        if (_size == 0) return NMLResult<ScoredValue<TValue>>::err(NMLErrorCode::OUT_OF_BOUNDS);

        return NMLResult<ScoredValue<TValue>>::ok(remove_max_unsafe());
    }

    template<typename TValue>
    TValue MinMaxHeap<TValue>::remove_min_unsafe() noexcept
    {
        auto min_index = find_min_index();
        auto min_value = _heap[min_index];

        delete_element(min_index);

        return min_value;
    }

    template<typename TValue>
    TValue MinMaxHeap<TValue>::remove_max_unsafe() noexcept
    {
        auto max_value = _heap[0];

        delete_element(0);

        return max_value;
    }

    template<typename TValue>
    void MinMaxHeap<TValue>::delete_element(uint64_t index) noexcept
    {
        if(index >= _size) return;

        if (index == _size - 1)
        {
            --_size; return;
        }

        std::swap(_heap[index], _heap[--_size]);

        push_down(index);
    }

    template<typename TValue>
    void MinMaxHeap<TValue>::pull_up(uint64_t index) noexcept
    {
        if (index == 0) return;

        uint64_t parent = parent_index(index);

        bool is_max_level = !is_min_level(index);

        if (!is_max_level && _heap[index] > _heap[parent])
        {
            std::swap(_heap[index], _heap[parent]);
            is_max_level = true;
            index = parent;
        }
        else if (is_max_level && _heap[index] < _heap[parent])
        {
            std::swap(_heap[index], _heap[parent]);
            is_max_level = false;
            index = parent;
        }

        while (index >= 3 && (_heap[index] < _heap[grandparent_index(index)]) ^ is_max_level)
        {
            std::swap(_heap[index], _heap[grandparent_index(index)]);
            index = grandparent_index(index);
        }
    }

    template<typename TValue>
    void MinMaxHeap<TValue>::push_down(uint64_t index) noexcept
    {
        bool is_max_level = !is_min_level(index);

        while (true)
        {
            uint64_t smallest = index;
            uint64_t left = left_child_index(index);
            uint64_t right = right_child_index(index);
            uint64_t grandchild_index = left_child_index(left);

            if (left < _size && (_heap[left] < _heap[smallest]) ^ is_max_level)
            {
                smallest = left;
            }

            if (right < _size && (_heap[right] < _heap[smallest]) ^ is_max_level)
            {
                smallest = right;
            }

            for (uint64_t i = 0; i < 4 && grandchild_index + i < _size; ++i)
            {
                if ((_heap[grandchild_index + i] < _heap[smallest]) ^ is_max_level)
                {
                    smallest = grandchild_index + i;
                }
            }

            if (smallest == index) return;

            std::swap(_heap[index], _heap[smallest]);

            if (smallest - left > 1)
            {
                if ((_heap[parent_index(smallest)] < _heap[smallest]) ^ is_max_level)
                {
                    std::swap(_heap[parent_index(smallest)], _heap[smallest]);
                }

                index = smallest;

                continue;
            }

            return;
        }
    }

    static inline uint64_t log2(uint64_t index) noexcept
    {
        if (index == 0) return 0;

        uint64_t result = 0;

        while (index)
        {
            index >>= 1;
            ++result;
        }

        return result - 1;
    }

    template<typename TValue>
    uint64_t MinMaxHeap<TValue>::find_min_index() noexcept
    {
        return _size == 1 ? 0 : (_size == 2 || _heap[1] < _heap[2] ? 1 : 2);
    }

    template<typename TValue>
    bool MinMaxHeap<TValue>::is_min_level(uint64_t index) noexcept
    {
        return log2(index + 1) % 2 == 1;
    }

    template<typename TValue>
    void MinMaxHeap<TValue>::print() const
    {
        std::cout << "{";

        for (uint64_t i = 0; i < _size; ++i)
        {
            if (i > 0) std::cout << ", ";
            std::cout << _heap[i];
        }

        std::cout << "}" << std::endl;
    }

    template<typename TValue>
    bool MinMaxHeap<TValue>::contains(TValue value) const noexcept // TODO
    {
        for (uint64_t i = 0; i < _heap; ++i)
        {
            if (_heap[i] == value) return true;
        }

        return false;
    }
}

#endif //NML_HEAP_MIN_MAX_H