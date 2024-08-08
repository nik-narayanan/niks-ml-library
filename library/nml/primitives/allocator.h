//
// Created by nik on 6/17/2024.
//

#ifndef NML_ALLOCATOR_H
#define NML_ALLOCATOR_H

#include "list.h"
#include "heap.h"

namespace nml
{
    template<typename T>
    struct Allocator
    {
        [[nodiscard]] inline T& get_element(uint64_t index) noexcept
        {
            return _elements[index - 1];
        }

        [[nodiscard]] inline uint64_t claim_next_index() noexcept
        {
            if (_available_indexes.is_empty())
            {
                _resize();
            }

            return _available_indexes.pop();
        }

        inline void return_index(const uint64_t index) noexcept
        {
            if (index == 0) return;

            get_element(index).initialize();

            _available_indexes.push(index);
        }

        inline void reset() noexcept
        {
            _available_indexes.reset();

            _initialize_allocation_range(0, _elements.length);
        }

        explicit Allocator(uint64_t initial_capacity = 0) noexcept
            : _elements(Span<T>())
            , _available_indexes(Heap<uint64_t, true>())
            , _capacity(initial_capacity == 0 ? 10 : initial_capacity)
        {
            MemoryLayout layout = _get_memory_layout(_capacity);

            uint64_t total_bytes = layout.total_bytes();

            _memory = static_cast<char*>(std::malloc(total_bytes));
            auto memory_span = MemorySpan(_memory, total_bytes);

            _elements = Span<T>(memory_span, _capacity);

            _available_indexes = Heap<uint64_t, true>(memory_span.offset(layout.element_bytes));

            _initialize_allocation_range(0, _capacity);
        }

        ~Allocator() noexcept
        {
            std::free(_memory);
        }

        Allocator(Allocator&& other) noexcept
            : _memory(other._memory)
            , _elements(other._elements)
            , _capacity(other._capacity)
            , _available_indexes(other._available_indexes)
        {
            other._memory = nullptr;
        }

        Allocator& operator=(Allocator&& other) noexcept
        {
            if (this != &other)
            {
                this->~Allocator();
                new(this) Allocator(std::move(other));
            }

            return *this;
        }

        Allocator(const Allocator& other) = delete;
        Allocator& operator=(const Allocator& other) = delete;

    private:

        char* _memory;
        uint64_t _capacity;

        Span<T> _elements;
        Heap<uint64_t, true> _available_indexes;

        struct MemoryLayout
        {
            uint64_t heap_bytes;
            uint64_t element_bytes;

            inline uint64_t total_bytes() noexcept { return heap_bytes + element_bytes; }
        };

        static inline MemoryLayout _get_memory_layout(uint64_t capacity) noexcept
        {
            return MemoryLayout
            {
                .heap_bytes = Heap<uint64_t, true>::required_bytes(capacity),
                .element_bytes = sizeof(T) * capacity
            };
        }

        void _resize() noexcept
        {
            uint64_t new_capacity = _capacity * 2;

            MemoryLayout layout = _get_memory_layout(new_capacity);

            uint64_t total_bytes = layout.total_bytes();

            auto new_memory = static_cast<char*>(std::malloc(total_bytes));
            auto memory_span = MemorySpan(new_memory, total_bytes);

            uint64_t old_offset = 0, new_offset = 0;

            memcpy(new_memory, _memory, _elements.bytes());

            old_offset += _elements.bytes(); new_offset += layout.element_bytes;

            memcpy(_memory + old_offset, new_memory + new_offset, _available_indexes.bytes());

            uint64_t old_capacity = _capacity;

            std::free(_memory);
            _memory = new_memory;
            _capacity = new_capacity;

            _elements = Span<T>(memory_span, new_capacity);

            _available_indexes = Heap<uint64_t, true>(
                memory_span.offset(layout.element_bytes),
                _available_indexes.size()
            );

            _initialize_allocation_range(old_capacity, _elements.length);
        }

        inline void _initialize_allocation_range(uint64_t start, uint64_t end) noexcept
        {
            for (uint64_t i = start; i < end; ++i)
            {
                _elements[i].initialize();

                _available_indexes.push(i + 1);
            }
        }
    };
}

#endif //NML_ALLOCATOR_H