//
// Created by nik on 4/14/2024.
//

#ifndef NML_GRID_SPAN_H
#define NML_GRID_SPAN_H

#include "span.h"
#include "memory_span.h"

namespace nml
{
    template<typename T>
    class GridSpan
    {
        T* _values;

    public:

        const unsigned row_ct;
        const unsigned column_ct;

        explicit GridSpan(T* values, unsigned row_ct, unsigned column_ct) noexcept
            : _values(values), row_ct(row_ct), column_ct(column_ct)
        { }

        explicit GridSpan(const MemorySpan memory, unsigned row_ct, unsigned column_ct) noexcept
            : _values(memory.get_pointer<T>(0)), row_ct(row_ct), column_ct(column_ct)
        { }

        [[nodiscard]] static inline unsigned required_bytes(unsigned row_ct, unsigned column_ct) noexcept { return row_ct * column_ct * sizeof(T); }

        ~GridSpan() noexcept = default;
        GridSpan(GridSpan &) noexcept = default;
        GridSpan(GridSpan &&other) noexcept = default;

        [[nodiscard]] T* get_pointer(const unsigned offset) const { return &_values[offset]; }
        [[nodiscard]] inline constexpr unsigned bytes() const noexcept { return row_ct * column_ct * sizeof(T); }
        [[nodiscard]] inline constexpr unsigned length() const noexcept { return row_ct * column_ct; }

        void print() const;

        void fill(T value) noexcept;
        void copy_into_unsafe(T* buffer) const noexcept;

        [[nodiscard]] inline unsigned get_linear_index_unsafe(unsigned row, unsigned column) const noexcept;

        inline T& get_unsafe_ref(unsigned row, unsigned column) noexcept;
        [[nodiscard]] inline T get_unsafe(unsigned row, unsigned column) const noexcept;

        inline void set_unsafe(unsigned row, unsigned column, T value) noexcept;

        [[nodiscard]] MemorySpan to_memory() const noexcept;
        Span<T> operator[](unsigned row) const noexcept;
    };

    template<typename T>
    MemorySpan GridSpan<T>::to_memory() const noexcept
    {
        return MemorySpan(_values, bytes());
    }

    template<typename T>
    Span<T> GridSpan<T>::operator[](unsigned int row) const noexcept
    {
        return Span<T>(&_values[get_linear_index_unsafe(row, 0)], column_ct);
    }

    template<typename T>
    void GridSpan<T>::set_unsafe(unsigned row, unsigned column, T value) noexcept
    {
        _values[get_linear_index_unsafe(row, column)] = value;
    }

    template<typename T>
    T GridSpan<T>::get_unsafe(unsigned int row, unsigned int column) const noexcept
    {
        return _values[get_linear_index_unsafe(row, column)];
    }

    template<typename T>
    T& GridSpan<T>::get_unsafe_ref(unsigned row, unsigned column) noexcept
    {
        return _values[get_linear_index_unsafe(row, column)];
    }

    template<typename T>
    unsigned GridSpan<T>::get_linear_index_unsafe(unsigned row, unsigned column) const noexcept
    {
        return row * column_ct + column;
    }

    template<typename T>
    void GridSpan<T>::copy_into_unsafe(T* buffer) const noexcept
    {
        memcpy(buffer, _values, bytes());
    }

    template<typename T>
    void GridSpan<T>::fill(T value) noexcept
    {
        if (value == 0)
        {
            memset(_values, 0, bytes()); return;
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                set_unsafe(row, column, value);
            }
        }
    }

    template<typename T>
    void GridSpan<T>::print() const
    {
        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                if (column > 0) std::cout << ", ";
                std::cout << get_unsafe(row, column);
            }

            std::cout << std::endl;
        }
    }
}

#endif //NML_GRID_SPAN_H
