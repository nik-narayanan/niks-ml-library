//
// Created by nik on 4/8/2024.
//

#ifndef NML_HEAP_H
#define NML_HEAP_H

#include "memory_span.h"

namespace nml
{
    template<typename TValue, bool MinHeap = false>
    class Heap
    {
        TValue* _heap;
        uint64_t _size{};
        uint64_t _capacity{};

    public:

        Heap() noexcept : _heap(nullptr), _size(0), _capacity(0) { };
        explicit Heap(MemorySpan memory, uint64_t size = 0) noexcept;
        [[nodiscard]] inline uint64_t size() const noexcept { return _size; }
        inline constexpr uint64_t bytes() noexcept { return _capacity * sizeof(TValue); }
        static inline constexpr uint64_t required_bytes(uint64_t capacity) noexcept { return capacity * sizeof(TValue); }

        TValue pop() noexcept;
        TValue peek() noexcept;
        void push(TValue value) noexcept;

        inline void reset() noexcept { _size = 0; }
        [[nodiscard]] inline bool is_empty() const noexcept { return _size == 0; }
        [[nodiscard]] inline bool is_full() const noexcept { return _size + 1 > _capacity; }

        void print() const;

    private:

        static inline uint64_t parent_index(uint64_t index) noexcept { return (index - 1) / 2; }
        static inline uint64_t left_child_index(uint64_t index) noexcept { return 2 * index + 1; }
        static inline uint64_t right_child_index(uint64_t index) noexcept { return 2 * index + 2; }

        void heapify(uint64_t index);
    };

    template<typename TValue, bool MinHeap>
    Heap<TValue, MinHeap>::Heap(MemorySpan memory, uint64_t size) noexcept
        : _size(size)
        , _capacity(memory.bytes / sizeof(TValue))
        , _heap(memory.get_pointer<TValue>(0))
    { }

    template<typename TValue, bool MinHeap>
    void Heap<TValue, MinHeap>::heapify(uint64_t index)
    {
        while (true)
        {
            uint64_t left = left_child_index(index);
            uint64_t right = right_child_index(index);
            uint64_t largest = index;

            if (left < _size && (_heap[left] > _heap[largest]) ^ MinHeap)
            {
                largest = left;
            }

            if (right < _size && (_heap[right] > _heap[largest]) ^ MinHeap)
            {
                largest = right;
            }

            if (index == largest) return;

            std::swap(_heap[index], _heap[largest]);
            index = largest;
        }
    }

    template<typename TValue, bool MinHeap>
    TValue Heap<TValue, MinHeap>::pop() noexcept
    {
        TValue max = _heap[0];
        _heap[0] = _heap[--_size];
        heapify(0);

        return max;
    }

    template<typename TValue, bool MinHeap>
    TValue Heap<TValue, MinHeap>::peek() noexcept
    {
        return _heap[0];
    }


    template<typename TValue, bool MinHeap>
    void Heap<TValue, MinHeap>::push(TValue value) noexcept
    {
        if (_size + 1 > _capacity) return;

        _heap[_size] = value;

        uint64_t index = _size++;

        while (index > 0 && (_heap[parent_index(index)] < _heap[index]) ^ MinHeap)
        {
            std::swap(_heap[index], _heap[parent_index(index)]);
            index = parent_index(index);
        }
    }

    template<typename TValue, bool MinHeap>
    void Heap<TValue, MinHeap>::print() const
    {
        std::cout << "{";

        for (uint64_t i = 0; i < _size; ++i)
        {
            if (i > 0) std::cout << ", ";
            std::cout << _heap[i];
        }

        std::cout << "}" << std::endl;
    }

}

#endif //NML_HEAP_H
