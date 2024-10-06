//
// Created by nik on 6/17/2024.
//

#ifndef NML_ALLOCATOR_H
#define NML_ALLOCATOR_H

#include "heap.h"
#include "span.h"

namespace nml
{
    template<typename T, bool ThreadSafe = false>
    struct Allocator
    {
        explicit Allocator(uint64_t initial_capacity = 0) noexcept;

        ~Allocator() noexcept;
        Allocator(Allocator&& move) noexcept;
        Allocator& operator=(Allocator&& move) noexcept;

        Allocator(Allocator&) = delete;
        Allocator& operator=(Allocator&) = delete;

        void reset() noexcept;
        void return_index(uint64_t index) noexcept;
        [[nodiscard]] uint64_t claimed_ct() const noexcept;
        [[nodiscard]] uint64_t claim_next_index() noexcept;
        [[nodiscard]] T& get_element(uint64_t index) noexcept;

        template<std::enable_if_t<ThreadSafe, bool> = true>
        void lock() { _lock.lock(); }

        template<std::enable_if_t<ThreadSafe, bool> = true>
        void unlock() { _lock.unlock(); }

        template<std::enable_if_t<ThreadSafe, bool> = true>
        [[nodiscard]] uint64_t claim_next_index_unsafe() noexcept;

        template<std::enable_if_t<ThreadSafe, bool> = true>
        void return_index_unsafe(uint64_t index) noexcept;

    private:

        struct MemoryLayout;

        char* _memory{};
        uint64_t _capacity{};

        Span<T> _elements{};
        Heap<uint64_t, true> _available_indexes{};

        std::conditional_t<ThreadSafe, std::mutex, std::nullptr_t> _lock{};

        void _resize() noexcept;
        static MemoryLayout _get_memory_layout(uint64_t capacity) noexcept;
        void _initialize_allocation_range(uint64_t start, uint64_t end) noexcept;
    };

    template<typename T, bool ThreadSafe>
    template<std::enable_if_t<ThreadSafe, bool>>
    void Allocator<T, ThreadSafe>::return_index_unsafe(uint64_t index) noexcept
    {
        if (index == 0) return;

        T& element = get_element(index);

        element.~T(), element = {};

        _available_indexes.push(index);
    }

    template<typename T, bool ThreadSafe>
    template<std::enable_if_t<ThreadSafe, bool>>
    uint64_t Allocator<T, ThreadSafe>::claim_next_index_unsafe() noexcept
    {
        if (_available_indexes.is_empty()) _resize();

        return _available_indexes.pop();
    }

    template<typename T, bool ThreadSafe>
    uint64_t Allocator<T, ThreadSafe>::claimed_ct() const noexcept
    {
        return _capacity - _available_indexes.size();
    }

    template<typename T, bool ThreadSafe>
    void Allocator<T, ThreadSafe>::return_index(uint64_t index) noexcept
    {
        if (index == 0) return;

        if constexpr (ThreadSafe) _lock.lock();

        T& element = get_element(index);

        element.~T(), element = {};

        _available_indexes.push(index);

        if constexpr (ThreadSafe) _lock.unlock();
    }

    template<typename T, bool ThreadSafe>
    uint64_t Allocator<T, ThreadSafe>::claim_next_index() noexcept
    {
        if constexpr (ThreadSafe) _lock.lock();

        if (_available_indexes.is_empty()) _resize();

        uint64_t index = _available_indexes.pop();

        if constexpr (ThreadSafe) _lock.unlock();

        return index;
    }

    template<typename T, bool ThreadSafe>
    T& Allocator<T, ThreadSafe>::get_element(uint64_t index) noexcept
    {
        return _elements[index - 1];
    }

    template<typename T, bool ThreadSafe>
    void Allocator<T, ThreadSafe>::_initialize_allocation_range(uint64_t start, uint64_t end) noexcept
    {
        for (uint64_t i = start; i < end; ++i)
        {
            _elements[i] = {};

            _available_indexes.push(i + 1);
        }
    }

    template<typename T, bool ThreadSafe>
    void Allocator<T, ThreadSafe>::_resize() noexcept
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

    template<typename T, bool ThreadSafe>
    struct Allocator<T, ThreadSafe>::MemoryLayout
    {
        uint64_t heap_bytes{};
        uint64_t element_bytes{};

        inline uint64_t total_bytes() noexcept { return heap_bytes + element_bytes; }
    };

    template<typename T, bool ThreadSafe>
    typename Allocator<T, ThreadSafe>::MemoryLayout Allocator<T, ThreadSafe>::_get_memory_layout(uint64_t capacity) noexcept
    {
        return MemoryLayout
        {
            .heap_bytes = Heap<uint64_t, true>::required_bytes(capacity),
            .element_bytes = sizeof(T) * capacity
        };
    }

    template<typename T, bool ThreadSafe>
    Allocator<T, ThreadSafe>& Allocator<T, ThreadSafe>::operator=(Allocator&& move) noexcept
    {
        if (this != &move)
        {
            this->~Allocator();
            new(this) Allocator(std::move(move));
        }

        return *this;
    }

    template<typename T, bool ThreadSafe>
    Allocator<T, ThreadSafe>::Allocator(Allocator&& move) noexcept
        : _memory(move._memory)
        , _elements(move._elements)
        , _capacity(move._capacity)
        , _available_indexes(move._available_indexes)
    {
        move._memory = nullptr;
    }

    template<typename T, bool ThreadSafe>
    Allocator<T, ThreadSafe>::~Allocator() noexcept
    {
        std::free(_memory);
    }

    template<typename T, bool ThreadSafe>
    Allocator<T, ThreadSafe>::Allocator(uint64_t initial_capacity) noexcept
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

    template<typename T, bool ThreadSafe>
    void Allocator<T, ThreadSafe>::reset() noexcept
    {
        _available_indexes.reset();

        _initialize_allocation_range(0, _elements.length);
    }
}

#endif //NML_ALLOCATOR_H