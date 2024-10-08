//
// Created by nik on 4/6/2024.
//

#ifndef NML_LIST_H
#define NML_LIST_H

#include <cstdlib>

#include "span.h"
#include "memory_span.h"

namespace nml
{
    template<typename T>
    class StaticList
    {
        T* _values;

    public:

        uint64_t count{};
        const uint64_t capacity{};

        Span<T>::Iterator begin() const { return to_span(0, count).begin(); }
        Span<T>::Iterator end() const { return to_span(0, count).end(); }

        explicit StaticList(uint64_t capacity = 10) noexcept
            : count(0)
            , capacity(capacity)
            , _values(static_cast<T*>(std::malloc(capacity * sizeof(T))))
        { }

        ~StaticList() noexcept
        {
            std::free(_values);
        }

        StaticList(const StaticList&) = delete;
        StaticList& operator=(const StaticList&) = delete;
        StaticList& operator=(const StaticList&&) = delete;

        StaticList(StaticList&& other) noexcept
            : _values(other._values)
            , count(other.count)
            , capacity(other.capacity)
        {
            other._values = nullptr;
            other.count = 0;
        }

        T& operator[](uint64_t index) noexcept
        {
            return _values[index];
        }

        const T& operator[](uint64_t index) const noexcept
        {
            return _values[index];
        }

        T* add(const T& element) noexcept
        {
            _values[count] = element;
            return &_values[count++];
        }

        Span<T> to_span() const noexcept
        {
            return Span<T>(_values, capacity);
        }

        Span<T> to_span(uint64_t start, uint64_t length) const noexcept
        {
            return Span<T>(_values + start * sizeof(T), length);
        }

        [[nodiscard]] MemorySpan to_memory() const noexcept
        {
            return MemorySpan(_values, count * sizeof(T));
        }

        [[nodiscard]] MemorySpan to_memory(uint64_t length) const noexcept
        {
            return MemorySpan(_values, length * sizeof(T));
        }

        [[nodiscard]] bool is_full() const noexcept
        {
            return count == capacity;
        }

        [[nodiscard]] bool can_add() const noexcept
        {
            return count < capacity;
        }
    };

    template<typename T>
    class StaticOwnerList
    {
        T* _values;

    public:

        uint64_t count{};
        const uint64_t capacity{};

        Span<T>::Iterator begin() const { return to_span(0, count).begin(); }
        Span<T>::Iterator end() const { return to_span(0, count).end(); }

        explicit StaticOwnerList(uint64_t capacity = 10) noexcept
            : count(0)
            , capacity(capacity)
            , _values(static_cast<T*>(std::malloc(capacity * sizeof(T))))
        { }

        ~StaticOwnerList() noexcept
        {
            for (uint64_t i = 0; i < count; ++i)
            {
                _values[i].~T();
            }
            
            std::free(_values);
        }

        StaticOwnerList(const StaticOwnerList&) = delete;
        StaticOwnerList& operator=(const StaticOwnerList&) = delete;
        StaticOwnerList& operator=(const StaticOwnerList&&) = delete;

        StaticOwnerList(StaticOwnerList&& other) noexcept
            : count(other.count)
            , _values(other._values)
            , capacity(other.capacity)
        {
            other._values = nullptr;
            other.count = 0;
        }

        T& operator[](uint64_t index) noexcept
        {
            return _values[index];
        }

        const T& operator[](uint64_t index) const noexcept
        {
            return _values[index];
        }

        T* add(const T& element) noexcept
        {
            new (_values + count) T(element);
            return &_values[count++];
        }

        T* add(T&& element) noexcept
        {
            new (&_values[count]) T(std::move(element));
            return &_values[count++];
        }

        Span<T> to_span() const noexcept
        {
            return Span<T>(_values, capacity);
        }

        Span<T> to_span(uint64_t start, uint64_t length) const noexcept
        {
            return Span<T>(_values + start * sizeof(T), length);
        }

        [[nodiscard]] MemorySpan to_memory() const noexcept
        {
            return MemorySpan(_values, count * sizeof(T));
        }

        [[nodiscard]] MemorySpan to_memory(uint64_t length) const noexcept
        {
            return MemorySpan(_values, length * sizeof(T));
        }

        [[nodiscard]] bool is_full() const noexcept
        {
            return count == capacity;
        }

        [[nodiscard]] bool can_add() const noexcept
        {
            return count < capacity;
        }
    };

    template<typename T>
    class ResizableList
    {
        T* _values;

    public:

        uint64_t count{};
        uint64_t capacity{};

        Span<T>::Iterator begin() const { return to_span(0, count).begin(); }
        Span<T>::Iterator end() const { return to_span(0, count).end(); }

        explicit ResizableList(uint64_t capacity = 10) noexcept
            : count(0)
            , capacity(capacity)
            , _values(static_cast<T*>(std::malloc(capacity * sizeof(T))))
        { }

        ~ResizableList() noexcept
        {
            std::free(_values);
        }

        ResizableList(const ResizableList&) = delete;
        ResizableList& operator=(const ResizableList&) = delete;
        ResizableList& operator=(const ResizableList&&) = delete;

        ResizableList(ResizableList&& other) noexcept
            : _values(other._values)
            , count(other.count)
            , capacity(other.capacity)
        {
            other._values = nullptr;
            other.count = 0;
            other.capacity = 0;
        }

        void resize() noexcept
        {
            capacity *= 2;
            _values = static_cast<T*>(std::realloc(_values, capacity * sizeof(T)));
        }

        T& operator[](uint64_t index) noexcept
        {
            return _values[index];
        }

        const T& operator[](uint64_t index) const noexcept
        {
            return _values[index];
        }

        uint64_t add(const T& element) noexcept
        {
            if (count == capacity) resize();
            _values[count] = element;
            return count++;
        }

        void pop() noexcept
        {
            count--;
        }

        void reset() noexcept
        {
            count = 0;
        }

        Span<T> to_span() const noexcept
        {
            return Span<T>(_values, count);
        }

        Span<T> to_span(uint64_t start, uint64_t length) const noexcept
        {
            return Span(_values + start * sizeof(T), length);
        }

        [[nodiscard]] MemorySpan to_memory() const noexcept
        {
            return MemorySpan(_values, count * sizeof(T));
        }

        [[nodiscard]] MemorySpan to_memory(uint64_t length) const noexcept
        {
            return MemorySpan(_values, length * sizeof(T));
        }
    };

    template<typename T>
    class ResizableOwnerList
    {
        T* _values;

    public:

        uint64_t count{};
        uint64_t capacity{};

        Span<T>::Iterator begin() const { return to_span(0, count).begin(); }
        Span<T>::Iterator end() const { return to_span(0, count).end(); }

        explicit ResizableOwnerList(uint64_t capacity = 10) noexcept
            : count(0)
            , capacity(capacity)
            , _values(static_cast<T*>(std::malloc(capacity * sizeof(T))))
        { }

        ~ResizableOwnerList() noexcept
        {
            for (uint64_t i = 0; i < count; ++i)
            {
                _values[i].~T();
            }

            std::free(_values);
        }

        ResizableOwnerList(const ResizableOwnerList&) = delete;
        ResizableOwnerList& operator=(const ResizableOwnerList&) = delete;
        ResizableOwnerList& operator=(const ResizableOwnerList&&) = delete;

        ResizableOwnerList(ResizableOwnerList&& other) noexcept
            : count(other.count)
            , _values(other._values)
            , capacity(other.capacity)
        {
            other._values = nullptr;
            other.count = 0;
            other.capacity = 0;
        }

        void resize() noexcept
        {
            uint64_t new_capacity = capacity * 2;

            T* new_values = static_cast<T*>(std::malloc(new_capacity * sizeof(T)));

            for (uint64_t i = 0; i < count; ++i)
            {
                new (new_values + i) T(std::move_if_noexcept(_values[i]));
                _values[i].~T();
            }

            std::free(_values);

            _values = new_values, capacity = new_capacity;
        }

        T& operator[](uint64_t index) noexcept
        {
            return _values[index];
        }

        const T& operator[](uint64_t index) const noexcept
        {
            return _values[index];
        }

        uint64_t add(const T& element) noexcept
        {
            if (count == capacity) resize();
            new (_values + count) T(element);
            return count++;
        }

        T* add(T&& element) noexcept
        {
            new (&_values[count]) T(std::move(element));
            return &_values[count++];
        }

        void pop() noexcept
        {
            if (count == 0) return;
            _values[--count].~T();
        }

        void reset() noexcept
        {
            for (uint64_t i = 0; i < count; ++i)
            {
                _values[i].~T();
            }

            count = 0;
        }

        Span<T> to_span() const noexcept
        {
            return Span<T>(_values, count);
        }

        Span<T> to_span(uint64_t start, uint64_t length) const noexcept
        {
            return Span(_values + start * sizeof(T), length);
        }

        [[nodiscard]] MemorySpan to_memory() const noexcept
        {
            return MemorySpan(_values, count * sizeof(T));
        }

        [[nodiscard]] MemorySpan to_memory(uint64_t length) const noexcept
        {
            return MemorySpan(_values, length * sizeof(T));
        }
    };
}

#endif //NML_LIST_H