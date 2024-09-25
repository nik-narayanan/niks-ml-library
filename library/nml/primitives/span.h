//
// Created by nik on 3/31/2024.
//

#ifndef NML_SPAN_H
#define NML_SPAN_H

#include <algorithm>
#include "memory_span.h"

namespace nml
{
    template<typename T>
    class Span
    {
        class Iterator;
        T* _values;

    public:

        uint64_t length{};

        Span() noexcept : _values(nullptr), length(0) { }

        explicit Span(T* values, const unsigned length) noexcept
            : _values(values), length(length)
        { }

        explicit Span(const MemorySpan memory, const unsigned length = 0) noexcept
            : _values(memory.get_pointer<T>(0))
            , length(length == 0 ? memory.bytes / sizeof(T) : length)
        { }

        [[nodiscard]] inline constexpr unsigned bytes() const noexcept { return length * sizeof(T); }
        [[nodiscard]] static inline unsigned required_bytes(unsigned length) noexcept { return length * sizeof(T); }

        inline void zero() noexcept { memset( _values, 0, bytes()); }
        inline void fill(const T value) noexcept { for (unsigned i = 0; i < length; ++i) _values[i] = value; }

        [[nodiscard]] bool scan_contains(const T value) { for (int i = 0; i < length; ++i) { if (_values[i] == value) return true; } return false; }

        void sort_ascending() { std::sort(_values, _values + length); }
        void sort_descending() { std::sort(_values, _values + length, std::greater<>()); }

        T& operator[](unsigned offset) noexcept { return _values[offset]; }
        const T& operator[](unsigned offset) const noexcept { return _values[offset]; }

        [[nodiscard]] T* get_pointer(const unsigned offset = 0) const { return &_values[offset]; }

        template<typename TPtr> [[nodiscard]] TPtr* get_pointer(const unsigned offset) const { return reinterpret_cast<TPtr*>(&_values[offset]); }

        [[nodiscard]] MemorySpan to_memory_unsafe(const unsigned offset = 0) const { return MemorySpan(get_pointer(offset), (length - offset) * sizeof(T)); }

        [[nodiscard]] Span<T> to_subspan_unsafe(const unsigned offset) const { return Span<T>(_values + offset, length - offset); }
        [[nodiscard]] Span<T> to_subspan_unsafe(const unsigned start, const unsigned sub_length) const { return Span<T>(_values + start, sub_length); }

        [[nodiscard]] Iterator begin() const noexcept;
        [[nodiscard]] Iterator end() const noexcept;

        [[nodiscard]] uint64_t hash() const noexcept
        {
            std::hash<T> hasher{};
            uint64_t hash = 0xCBF29CE484222325;

            for (uint64_t offset = 0; offset < length; ++offset)
            {
                hash ^= hasher(_values[offset]);
                hash *= 0x100000001B3;
            }

            return hash;
        }

        bool operator==(const Span<T>& other) const noexcept
        {
            if (length != other.length) return false;
            if (_values == other._values) return true;

            for (uint64_t offset = 0; offset < length; ++offset)
            {
                if (_values[offset] != other[offset]) return false;
            }

            return true;
        }

        bool operator!=(const Span<T>& other) const
        {
            return !(*this == other);
        }

        void print(const char* separator = ", ", bool terminate_line = true) const
        {
            for (unsigned i = 0; i < length; ++i)
            {
                if (i > 0) std::cout << separator;
                std::cout << _values[i];
            }

            if (terminate_line) std::cout << std::endl;
        }
    };

    template<typename T>
    class Span<T>::Iterator
    {
        Span<T> _span;
        uint64_t _position{};

    public:

        explicit Iterator(Span<T> span, uint64_t position = 0)
            : _span(span), _position(position)
        { }

        Iterator begin() const { return Iterator(_span, 0); }
        Iterator end() const { return Iterator(_span, _span.bytes); }

        Iterator& operator++()
        {
            ++_position;

            return *this;
        }

        const T& operator*() const
        {
            return *_span.get_pointer(_position);
        }

        bool operator==(const Iterator& rhs) const
        {
            return _position == rhs._position;
        }

        bool operator!=(const Iterator& rhs) const
        {
            return _position != rhs._position;
        }
    };

    template<typename T>
    typename Span<T>::Iterator Span<T>::begin() const noexcept
    {
        return Span<T>::Iterator(*this);
    }

    template<typename T>
    typename Span<T>::Iterator Span<T>::end() const noexcept
    {
        return Span<T>::Iterator(*this, length);
    }
}

namespace std { template<typename T> struct hash<nml::Span<T>> { size_t operator()(const nml::Span<T>& span) const noexcept { return span.hash(); } }; }

#endif //NML_SPAN_H