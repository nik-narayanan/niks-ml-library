//
// Created by nik on 3/24/2024.
//

#ifndef NML_MATRIX_SPAN_H
#define NML_MATRIX_SPAN_H

#include <cmath>
#include <iomanip>
#include <iostream>

#include "hash.h"
#include "list.h"
#include "random.h"
#include "memory_span.h"
#include "vector_span.h"
#include "matrix_result.h"

namespace nml
{
    class MatrixSpan
    {
        float* _values;

    public:

        const unsigned row_ct;
        const unsigned column_ct;

        const unsigned row_start;
        const unsigned column_start;

        const unsigned column_end;

        explicit MatrixSpan(float *values, unsigned row_ct, unsigned column_ct,
            unsigned row_start = 0, unsigned column_start = 0, unsigned column_end = 0) noexcept
            : _values(values), row_ct(row_ct), column_ct(column_ct), row_start(row_start), column_start(column_start)
            , column_end(std::max(column_ct + column_start, column_end))
        { }

        explicit MatrixSpan(const MemorySpan memory, unsigned row_ct, unsigned column_ct) noexcept
            : _values(memory.get_pointer<float>(0)), row_ct(row_ct), column_ct(column_ct)
            , row_start(0), column_start(0), column_end(column_ct)
        { }

        ~MatrixSpan() noexcept = default;
        MatrixSpan(MatrixSpan &) noexcept = default;
        MatrixSpan(MatrixSpan &&other) noexcept = default;

        void print() const;

        [[nodiscard]] inline bool is_symmetric(float threshold) const noexcept;
        [[nodiscard]] inline bool is_row_echelon(float zero_threshold) const noexcept;
        [[nodiscard]] inline constexpr bool is_square() const noexcept { return row_ct == column_ct; }
        [[nodiscard]] inline constexpr unsigned element_count() const noexcept { return row_ct * column_ct; }
        [[nodiscard]] inline constexpr unsigned bytes() const noexcept { return row_ct * column_ct * sizeof(float); }
        [[nodiscard]] static inline unsigned required_bytes(unsigned row_ct, unsigned column_ct) noexcept { return row_ct * column_ct * sizeof(float); }
        [[nodiscard]] inline constexpr bool is_memory_root() const noexcept { return row_start == 0 && column_start == 0 && column_end == column_ct; }

        [[nodiscard]] inline bool can_fit_inside(MatrixSpan rhs) const noexcept { return rhs.row_ct >= row_ct && rhs.column_ct >= column_ct; }
        [[nodiscard]] inline bool has_same_dimensions_as(MatrixSpan rhs) const noexcept { return rhs.row_ct == row_ct && rhs.column_ct == column_ct; }

        void center() noexcept;
        void set_identity() noexcept;
        void fill(float value) noexcept;
        void transpose_inplace() noexcept;
        void standardize(float zero_threshold) noexcept;
        void copy_into_unsafe(float *buffer) const noexcept;
        void copy_from_unsafe(const MatrixSpan& span) noexcept;
        void fill_random_uniform(float min, float max, unsigned seed = 42) noexcept;
        void fill_random_gaussian(float standard_deviation = 1, unsigned seed = 42) noexcept;

        void swap_rows(unsigned row_1, unsigned row_2);
        void swap_columns(unsigned column_1, unsigned column_2);

        [[nodiscard]] NMLResult<unsigned> get_linear_index(unsigned row, unsigned column) const noexcept;
        [[nodiscard]] inline unsigned get_linear_index_unsafe(unsigned row, unsigned column) const noexcept;

        float* get_pointer() noexcept { return _values; }
        NMLResult<float> get(unsigned row, unsigned column) noexcept;
        inline float& get_unsafe_ref(unsigned row, unsigned column) noexcept;
        [[nodiscard]] inline float get_unsafe(unsigned row, unsigned column) const noexcept;

        NMLResult<bool> set(unsigned row, int column, float value) noexcept;
        inline void set_unsafe(unsigned row, unsigned column, float value) noexcept;

        [[nodiscard]] VectorSpan operator[](unsigned row) const noexcept;
        [[nodiscard]] VectorSpan to_vector_subspan_unsafe(unsigned _column_ct, unsigned _row_start, unsigned _column_start) const noexcept;
        NMLResult<MatrixSpan> to_subspan(unsigned _row_ct, unsigned _column_ct, unsigned _row_start, unsigned _column_start) noexcept;
        [[nodiscard]] MatrixSpan to_subspan_unsafe(unsigned _row_ct, unsigned _column_ct, unsigned _row_start = 0, unsigned _column_start = 0) const noexcept;

        NMLResult<MatrixSpan> covariance(MatrixSpan& output) const noexcept;
        [[nodiscard]] bool equals(const MatrixSpan& rhs, float threshold) const noexcept;
        NMLResult<MatrixSpan> multiply(const MatrixSpan& rhs, MatrixSpan& output) const noexcept;
        NMLResult<MatrixSpan> transpose_multiply(const MatrixSpan& rhs, MatrixSpan& output) const noexcept;

        [[nodiscard]] StaticList<float> distinct(unsigned column) const noexcept; // WARNING: allocates
    };

    NMLResult<float> MatrixSpan::get(const unsigned row, const unsigned column) noexcept
    {
        auto index = get_linear_index(row, column);

        if (index.is_err()) return NMLResult<float>(index.err());

        return NMLResult<float>(_values[index.ok()]);
    }

    inline float& MatrixSpan::get_unsafe_ref(const unsigned row, const unsigned column) noexcept
    {
        return _values[get_linear_index_unsafe(row, column)];
    }

    inline float MatrixSpan::get_unsafe(const unsigned row, const unsigned column) const noexcept
    {
        return _values[get_linear_index_unsafe(row, column)];
    }

    NMLResult<bool> MatrixSpan::set(const unsigned row, const int column, float value) noexcept
    {
        auto index = get_linear_index(row, column);

        if (index.is_err()) return NMLResult<bool>(index.err());

        _values[index.ok()] = value;

        return NMLResult<bool>(true);
    }

    inline void MatrixSpan::set_unsafe(const unsigned row, const unsigned column, const float value) noexcept
    {
        _values[get_linear_index_unsafe(row, column)] = value;
    }

    inline VectorSpan MatrixSpan::operator[](unsigned row) const noexcept
    {
        return VectorSpan(_values + get_linear_index_unsafe(row, 0), column_ct);
    }

    inline VectorSpan MatrixSpan::to_vector_subspan_unsafe(const unsigned _column_ct, const unsigned _row_start, const unsigned _column_start) const noexcept
    {
        return VectorSpan(_values + get_linear_index_unsafe(_row_start, _column_start), _column_ct);
    }

    inline MatrixSpan MatrixSpan::to_subspan_unsafe(const unsigned _row_ct, const unsigned _column_ct,
                                                    const unsigned _row_start, const unsigned _column_start) const noexcept
    {
        return MatrixSpan(_values, _row_ct, _column_ct,
                          row_start + _row_start, column_start + _column_start, column_end);
    }

    NMLResult<MatrixSpan> MatrixSpan::to_subspan(const unsigned _row_ct, const unsigned _column_ct,
                                                 const unsigned _row_start, const unsigned _column_start) noexcept
    {
        if (_row_ct + _row_start + row_start > row_ct || _column_ct + _column_start + column_start > column_ct)
        {
            return NMLResult<MatrixSpan>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        return NMLResult<MatrixSpan>(MatrixSpan(_values, _row_ct,
                                                _column_ct, row_start, column_start));
    }

    void MatrixSpan::center() noexcept
    {
        auto float_rows = static_cast<float>(row_ct);

        auto working_memory = (float*)alloca(column_ct * sizeof(float)); // TODO remove alloca

        float* means = working_memory;

        for (unsigned column = 0; column < column_ct; ++column)
        {
            means[column] = 0;
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                means[column] += get_unsafe(row, column);
            }
        }

        for (unsigned column = 0; column < column_ct; ++column)
        {
            means[column] /= float_rows;
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                get_unsafe_ref(row, column) -= means[column];
            }
        }
    }

    void MatrixSpan::standardize(const float zero_threshold) noexcept
    {
        auto float_rows = static_cast<float>(row_ct);

        auto working_memory = (float*)alloca(2 * column_ct * sizeof(float));

        float* means = working_memory, *standard_deviations = working_memory + column_ct;
//        float means[column_ct], standard_deviations[column_ct];

        for (unsigned column = 0; column < column_ct; ++column)
        {
            means[column] = 0; standard_deviations[column] = 0;
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                means[column] += get_unsafe(row, column);
            }
        }

        for (unsigned column = 0; column < column_ct; ++column)
        {
            means[column] /= float_rows;
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                float value = get_unsafe(row, column) - means[column];
                standard_deviations[column] += value * value;
            }
        }

        for (unsigned column = 0; column < column_ct; ++column)
        {
            if (standard_deviations[column] < zero_threshold) continue;
            standard_deviations[column] = std::sqrt(standard_deviations[column] / float_rows);
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                if (standard_deviations[column] < zero_threshold) continue;

                float value = (get_unsafe(row, column) - means[column]) / standard_deviations[column];
                set_unsafe(row, column, value);
            }
        }
    }

    void MatrixSpan::transpose_inplace() noexcept
    {
        if (!is_square()) return;

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = row + 1; column < column_ct; ++column)
            {
                float swap = get_unsafe(row, column);
                set_unsafe(row, column, get_unsafe(column, row));
                set_unsafe(column, row, swap);
            }
        }
    }

    void MatrixSpan::fill_random_uniform(float min, float max, unsigned seed) noexcept
    {
        auto random = Random(seed);

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                float value = random.value(min, max);
                set_unsafe(row, column, value);
            }
        }
    }

    void MatrixSpan::fill_random_gaussian(float standard_deviation, unsigned seed) noexcept
    {
        auto random = Random(seed);

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                float value = random.gauss(standard_deviation);
                set_unsafe(row, column, value);
            }
        }
    }

    void MatrixSpan::set_identity() noexcept
    {
        if (is_memory_root())
        {
            memset(_values, 0, bytes());

            unsigned diagonals = std::min(row_ct, column_ct);

            for (unsigned diagonal = 0; diagonal < diagonals; ++diagonal)
            {
                set_unsafe(diagonal, diagonal, 1);
            }

            return;
        }

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                set_unsafe(row, column, row == column ? 1 : 0);
            }
        }
    }

    void MatrixSpan::fill(const float value) noexcept
    {
        if (is_memory_root() && value == 0)
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

    void MatrixSpan::copy_into_unsafe(float* buffer) const noexcept
    {
        if (is_memory_root())
        {
            memcpy(buffer, _values, bytes()); return;
        }

        unsigned offset = 0;

        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                buffer[offset++] = get_unsafe(row, column);
            }
        }
    }

    void MatrixSpan::copy_from_unsafe(const MatrixSpan& span) noexcept
    {
        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                set_unsafe(row, column, span.get_unsafe(row, column));
            }
        }
    }

    [[nodiscard]] NMLResult<unsigned> MatrixSpan::get_linear_index(const unsigned row, const unsigned column) const noexcept
    {
        const bool is_inbounds = (row < row_ct) && (column < column_ct);

        if (!is_inbounds) return NMLResult<unsigned>(NMLErrorCode::OUT_OF_BOUNDS);

        return NMLResult<unsigned>(get_linear_index_unsafe(row, column));
    }

    [[nodiscard]] inline unsigned MatrixSpan::get_linear_index_unsafe(const unsigned row, const unsigned column) const noexcept
    {
        return ((row_start + row) * (column_end)) + column_start + column;
    }

    void MatrixSpan::swap_rows(const unsigned row_1, const unsigned row_2)
    {
        for (int column = 0; column < column_ct; ++column)
        {
            float temp = get_unsafe(row_1, column);
            set_unsafe(row_1, column, get_unsafe(row_2, column));
            set_unsafe(row_2, column, temp);
        }
    }

    void MatrixSpan::swap_columns(unsigned column_1, unsigned column_2)
    {
        for (int row = 0; row < row_ct; ++row)
        {
            float temp = get_unsafe(row, column_1);
            set_unsafe(row, column_1, get_unsafe(row, column_2));
            set_unsafe(row, column_2, temp);
        }
    }

    void MatrixSpan::print() const
    {
        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                if (column > 0) std::cout << ", ";
                std::cout << std::setw(10) << std::fixed << get_unsafe(row, column);
            }

            std::cout << std::endl;
        }
    }

    NMLResult<MatrixSpan> MatrixSpan::covariance(MatrixSpan& output) const noexcept
    {
        if (column_ct != output.column_ct || column_ct != output.row_ct)
        {
            return NMLResult<MatrixSpan>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        output.fill(0);

        for (unsigned column_1 = 0; column_1 < column_ct; ++column_1)
        {
            for (unsigned column_2 = column_1; column_2 < column_ct; ++column_2)
            {
                float sum = 0.0;

                for (unsigned row = 0; row < row_ct; ++row)
                {
                    sum += get_unsafe(row, column_1) * get_unsafe(row, column_2);
                }

                auto covariance = sum / static_cast<float>(row_ct - 1);
                output.set_unsafe(column_1, column_2, covariance);
                output.set_unsafe(column_2, column_1, covariance);
            }
        }

        return NMLResult<MatrixSpan>(output);
    }

    NMLResult<MatrixSpan> MatrixSpan::multiply(const MatrixSpan& rhs, MatrixSpan& output) const noexcept
    {
        if (column_ct != rhs.row_ct || output.row_ct != row_ct || output.column_ct != rhs.column_ct)
        {
            return NMLResult<MatrixSpan>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        output.fill(0);

        for (unsigned lhs_row = 0; lhs_row < row_ct; ++lhs_row)
        {
            for (unsigned rhs_col = 0; rhs_col < rhs.column_ct; ++rhs_col)
            {
                for (unsigned lhs_col = 0; lhs_col < column_ct; ++lhs_col)
                {
                    float lhs_value = get_unsafe(lhs_row, lhs_col);
                    float rhs_value = rhs.get_unsafe(lhs_col, rhs_col);
                    output.get_unsafe_ref(lhs_row, rhs_col) += lhs_value * rhs_value;
                }
            }
        }

        return NMLResult<MatrixSpan>(output);
    }

    NMLResult<MatrixSpan> MatrixSpan::transpose_multiply(const MatrixSpan& rhs, MatrixSpan& output) const noexcept
    {
        if (row_ct != rhs.row_ct || output.row_ct != column_ct || output.column_ct != rhs.column_ct)
        {
            return NMLResult<MatrixSpan>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        output.fill(0);

        for (unsigned lhs_col = 0; lhs_col < column_ct; ++lhs_col)
        {
            for (unsigned rhs_col = 0; rhs_col < rhs.column_ct; ++rhs_col)
            {
                for (unsigned lhs_row = 0; lhs_row < row_ct; ++lhs_row)
                {
                    float lhs_value = get_unsafe(lhs_row, lhs_col);
                    float rhs_value = rhs.get_unsafe(lhs_row, rhs_col);
                    output.get_unsafe_ref(lhs_col, rhs_col) += lhs_value * rhs_value;
                }
            }
        }

        return NMLResult<MatrixSpan>(output);
    }

    bool MatrixSpan::equals(const MatrixSpan& rhs, const float threshold) const noexcept
    {
        if (row_ct != rhs.row_ct || column_ct != rhs.column_ct)
        {
            return false;
        }

//        float max_diff = 0;
        for (unsigned row = 0; row < row_ct; ++row)
        {
            for (unsigned column = 0; column < column_ct; ++column)
            {
                float lhs_value = get_unsafe(row, column);
                float rhs_value = rhs.get_unsafe(row, column);
                float relative_diff = std::abs(lhs_value - rhs_value);

//                max_diff = std::max(max_diff, relative_diff);
                if (relative_diff > threshold)
                {
                    return false;
                }
            }
        }

//        std::cout << max_diff;

        return true;
    }

    inline bool MatrixSpan::is_row_echelon(const float zero_threshold) const noexcept
    {
        for (unsigned row = 0; row < row_ct - 1; ++row)
        {
            for (unsigned column = row + 1; column < column_ct; ++column)
            {
                float value = std::abs(get_unsafe(row, column));

                if (value > zero_threshold)
                {
                    return false;
                }
            }
        }

        return true;
    }

    inline bool MatrixSpan::is_symmetric(const float threshold) const noexcept
    {
        if (!is_square()) return false;

        for (unsigned row = 0; row < row_ct - 1; ++row)
        {
            for (unsigned column = row + 1; column < column_ct; ++column)
            {
                float difference = get_unsafe(row, column) - get_unsafe(column, row);

                if (std::abs(difference) > threshold)
                {
                    return false;
                }
            }
        }

        return true;
    }

    StaticList<float> MatrixSpan::distinct(unsigned column) const noexcept
    {
        auto set = HashSet<float>();

        for (uint32_t row = 0; row < row_ct; ++row)
        {
            set.insert(get_unsafe(row, column));
        }

        auto labels = StaticList<float>(set.count());

        auto iterator = set.to_iterator();

        while (iterator.has_next())
        {
            labels.add(iterator.next());
        }

        return labels;
    }
}

#endif //NML_MATRIX_SPAN_H