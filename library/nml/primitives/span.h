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
        T* _values;

    public:

        uint64_t length;

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

        [[nodiscard]] T* get_end() const { return &_values[length]; }
        [[nodiscard]] T* get_pointer(const unsigned offset = 0) const { return &_values[offset]; }

        template<typename TPtr> [[nodiscard]] TPtr* get_pointer(const unsigned offset) const { return reinterpret_cast<TPtr*>(&_values[offset]); }

        [[nodiscard]] MemorySpan to_memory_unsafe(const unsigned offset = 0) const { return MemorySpan(get_pointer(offset), (length - offset) * sizeof(T)); }

        [[nodiscard]] Span<T> to_subspan_unsafe(const unsigned offset) const { return Span<T>(_values + offset, length - offset); }
        [[nodiscard]] Span<T> to_subspan_unsafe(const unsigned start, const unsigned sub_length) const { return Span<T>(_values + start, sub_length); }

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
}

#endif //NML_SPAN_H