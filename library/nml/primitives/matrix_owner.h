//
// Created by nik on 3/24/2024.
//

#ifndef NML_MATRIX_OWNER_H
#define NML_MATRIX_OWNER_H

#include <algorithm>
#include "matrix_result.h"
#include "../primitives/span.h"
#include "../primitives/file.h"
#include "../primitives/matrix_span.h"

namespace nml
{
    class MatrixOwner
    {
        float* _values;

    public:

        const unsigned row_ct;
        const unsigned column_ct;

        ~MatrixOwner();
        MatrixOwner(const MatrixOwner& other);
        MatrixOwner(MatrixOwner&& other) noexcept;
        explicit MatrixOwner(unsigned row_ct, unsigned column_ct) noexcept;
        explicit MatrixOwner(unsigned row_ct, unsigned column_ct, std::initializer_list<float> values);

        static MatrixOwner from_delimited(const char* path, bool has_headers = false, char delimiter = ',');
        static MatrixOwner from_delimited(const std::string& path, bool has_headers = false, char delimiter = ',');

        MatrixOwner& operator=(const MatrixOwner&) = delete;
        MatrixOwner& operator=(const MatrixOwner&&) = delete;

        inline float* get_pointer() { return _values; };

        void copy_from_unsafe(MatrixSpan span);
        NMLResult<bool> copy_from(MatrixSpan span);

        MatrixSpan to_span() const noexcept;
        MatrixSpan to_span_unsafe(unsigned _row_ct, unsigned _column_ct, unsigned row_start = 0, unsigned column_start = 0) noexcept;
        NMLResult<MatrixSpan> to_span(unsigned _row_ct, unsigned _column_ct, unsigned row_start, unsigned column_start) noexcept;

        [[nodiscard]] VectorSpan operator[](unsigned row) const noexcept;
        VectorSpan to_vector_span_unsafe(unsigned length, unsigned row_start, unsigned column_start) noexcept;
    };

    MatrixOwner::MatrixOwner(const unsigned row_ct, const unsigned column_ct) noexcept
        : row_ct(row_ct), column_ct(column_ct)
    {
        _values = static_cast<float*>(std::malloc(row_ct * column_ct * sizeof(float)));
        if (!_values) std::abort();
    }

    MatrixOwner::MatrixOwner(const unsigned row_ct, const unsigned column_ct, std::initializer_list<float> values)
        : row_ct(row_ct), column_ct(column_ct)
    {
        _values = static_cast<float*>(std::malloc(row_ct * column_ct * sizeof(float)));
        if (!_values) std::abort();
        std::copy(values.begin(), values.end(), _values);
    }

    MatrixOwner::MatrixOwner(const MatrixOwner& other)
        : row_ct(other.row_ct), column_ct(other.column_ct)
    {
        _values = static_cast<float*>(std::malloc(row_ct * column_ct * sizeof(float)));
        if (!_values) std::abort();
        memcpy(_values, other._values, row_ct * column_ct * sizeof(float));
    }

    MatrixOwner::MatrixOwner(MatrixOwner&& other) noexcept
        : row_ct(other.row_ct), column_ct(other.column_ct), _values(other._values)
    {
        other._values = nullptr;
    }

    MatrixOwner::~MatrixOwner()
    {
        std::free(_values);
    }

    NMLResult<bool> MatrixOwner::copy_from(MatrixSpan span)
    {
        if (span.row_ct * span.row_ct != row_ct * column_ct)
        {
            return NMLResult<bool>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        span.copy_into_unsafe(_values);

        return NMLResult<bool>(true);
    }

    void MatrixOwner::copy_from_unsafe(MatrixSpan span)
    {
        span.copy_into_unsafe(_values);
    }

    MatrixSpan MatrixOwner::to_span() const noexcept
    {
        return MatrixSpan(_values, row_ct, column_ct, 0, 0);
    }

    MatrixSpan MatrixOwner::to_span_unsafe(const unsigned _row_ct, const unsigned _column_ct,
                                           const unsigned row_start, const unsigned column_start) noexcept
    {
        return MatrixSpan(_values, _row_ct, _column_ct, row_start, column_start, column_ct);
    }

    NMLResult<MatrixSpan> MatrixOwner::to_span(const unsigned _row_ct, const unsigned _column_ct, const unsigned row_start, const unsigned column_start) noexcept
    {
        if (_row_ct + row_start > row_ct || _column_ct + column_start > column_ct)
        {
            return NMLResult<MatrixSpan>(NMLErrorCode::OUT_OF_BOUNDS);
        }

        return NMLResult<MatrixSpan>(MatrixSpan(_values, _row_ct,
                                                _column_ct, row_start, column_start));
    }

    VectorSpan MatrixOwner::to_vector_span_unsafe(unsigned length, unsigned row_start, unsigned column_start) noexcept
    {
        return VectorSpan(_values + (row_start * column_ct) + column_start, length);
    }

    VectorSpan MatrixOwner::operator[](unsigned int row) const noexcept
    {
        return to_span()[row]; // no
    }

    namespace matrix_owner_internal
    {
        static inline bool is_whitespace(const char ch) noexcept
        {
            return ch ==  ' ' || ch == '\t' ||
                   ch == '\n' || ch == '\r' ||
                   ch == '\f' || ch == '\v' ;
        }

        static inline bool is_number(const char ch) noexcept
        {
            return ch >= '0' && ch <= '9';
        }

        static inline double parse_float(Span<const char> span)
        {
            uint16_t offset = 0;

            double parsed = 0, adjustment = 0.1;

            bool is_negative = offset < span.length
                && !is_number(span[offset]) && span[offset] == '-';

            if (is_negative) offset += 1;

            while (offset < span.length && is_whitespace(span[offset]))
            {
                offset += 1;
            }

            while (offset < span.length && is_number(span[offset]))
            {
                parsed = parsed * 10.0f + static_cast<double>(span[offset++] - '0');
            }

            if (offset < span.length && span[offset] == '.')
            {
                offset += 1;
            }

            while (offset < span.length && is_number(span[offset]))
            {
                parsed += static_cast<double>(span[offset++] - '0') * adjustment;

                adjustment *= 0.1;
            }

            if (offset < span.length && (span[offset] == 'e' || span[offset] == 'E'))
            {
                offset += 1;

                int32_t power = 0;

                bool is_negative_power = offset < span.length
                    && !is_number(span[offset]) && span[offset++] == '-';

                while (offset < span.length && is_number(span[offset]))
                {
                    power = power * 10 + (span[offset++] - '0');
                }

                if (is_negative_power) power *= -1;

                parsed *= static_cast<double>(std::pow(10.0f, power));
            }

            while (offset < span.length && is_whitespace(span[offset]))
            {
                offset += 1;
            }

            if (offset < span.length)
            {
                throw std::invalid_argument("Unable to parse float from string: "
                    + std::string(span.get_pointer(), span.length)
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            return is_negative ? -parsed : parsed;
        }
    }

    MatrixOwner MatrixOwner::from_delimited(const std::string& path, bool has_headers, char delimiter)
    {
        return from_delimited(path.c_str(), has_headers, delimiter);
    }

    MatrixOwner MatrixOwner::from_delimited(const char* path, bool has_headers, char delimiter)
    {
        using namespace matrix_owner_internal;

        auto iterator = DataReader::delimited_iterator(path, delimiter);

        auto summary = iterator.summary();

        auto mx = MatrixOwner(summary.row_ct - has_headers, summary.column_ct);

        uint64_t counter = 0;

        while (iterator.has_next())
        {
            auto next = iterator.next();

            if (next.type == DataReader::DelimitedTokenType::VALUE)
            {
                auto span = Span<const char>(next.start, next.length);

                mx._values[counter++] = static_cast<float>(parse_float(span));
            }
        }

        return mx;
    }
}

#endif //NML_MATRIX_OWNER_H