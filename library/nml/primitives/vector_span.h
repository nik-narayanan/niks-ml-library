//
// Created by nik on 3/24/2024.
//

#ifndef NML_VECTOR_SPAN_H
#define NML_VECTOR_SPAN_H

#include <cmath>
#include <algorithm>

#include "random.h"

namespace nml
{
    class VectorSpan
    {
        float* _values;

    public:

        uint64_t length;

        explicit VectorSpan(float* values, uint64_t length) noexcept
            : _values(values), length(length)
        { }

        explicit VectorSpan(const MemorySpan memory, const uint64_t length = 0) noexcept
            : _values(memory.get_pointer<float>(0))
            , length(length == 0 ? memory.bytes / sizeof(float) : length)
        { }

        [[nodiscard]] inline constexpr uint64_t bytes() const noexcept { return length * sizeof(float); }
        [[nodiscard]] static inline uint64_t required_bytes(uint64_t length) noexcept { return length * sizeof(float); }

        VectorSpan(const VectorSpan& original) noexcept = default;
        VectorSpan& operator=(const VectorSpan& copy) noexcept = default;

        float operator[](const uint64_t offset) const noexcept { return _values[offset]; }
        float& operator[](const uint64_t offset) noexcept { return _values[offset]; }
        [[nodiscard]] float* get_pointer(const uint64_t offset) const noexcept { return &_values[offset]; }

        [[nodiscard]] VectorSpan to_subspan_unsafe(const uint64_t offset) const noexcept { return VectorSpan(_values + offset, length - offset); }
        [[nodiscard]] VectorSpan to_subspan_unsafe(const uint64_t start, const uint64_t sub_length) const noexcept { return VectorSpan(_values + start, sub_length); }

        void sort_ascending() noexcept { std::sort(_values, _values + length, std::less<>()); }
        void sort_descending() noexcept { std::sort(_values, _values + length, std::greater<>()); }

        void copy_from_unsafe(const VectorSpan& original) noexcept;
        void normalize(float zero_threshold = 1e-06, bool flip_sign = false) noexcept;

        void zero() noexcept { memset(_values, 0, bytes()); }
        void fill_random_gaussian(float standard_deviation = 1, uint64_t seed = 42) noexcept;
        void fill(const float value) noexcept { for (uint64_t i = 0; i < length; ++i) _values[i] = value; }

        float dot_product_unsafe(VectorSpan& rhs) const noexcept;

        [[nodiscard]] bool equals(const VectorSpan& rhs, float threshold) const noexcept;

        void print() const noexcept { for (uint64_t i = 0; i < length; ++i) { if (i > 0) std::cout << ", "; std::cout << _values[i]; } std::cout << std::endl; }
    };

    void VectorSpan::normalize(const float zero_threshold, const bool flip_sign) noexcept
    {
        float factor = 0;

        for (uint64_t i = 0; i < length; ++i) // todo vector_cuda
        {
            factor += _values[i] * _values[i];
        }

        if (factor < zero_threshold) return;
        factor = std::sqrt(factor);
        if (flip_sign) factor *= -1;

        for (uint64_t i = 0; i < length; ++i) // todo vector_cuda
        {
            _values[i] /= factor;
        }
    }

    void VectorSpan::copy_from_unsafe(const VectorSpan& original) noexcept
    {
        auto source = original.get_pointer(0);
        memcpy(_values, source, bytes());
    }

    float VectorSpan::dot_product_unsafe(VectorSpan& rhs) const noexcept
    {
        float dot_product = 0;

        for (uint64_t i = 0; i < length; ++i) // todo vectorize
        {
            dot_product += _values[i] * rhs[i];
        }

        return dot_product;
    }

    bool VectorSpan::equals(const VectorSpan& rhs, const float threshold) const noexcept
    {
        if (length != rhs.length)
        {
            return false;
        }

        for (int i = 0; i < length; ++i)
        {
            float lhs_value = _values[i];
            float rhs_value = rhs[i];
            float relative_diff = std::abs(lhs_value - rhs_value);

            if (relative_diff > threshold)
            {
                return false;
            }
        }

        return true;
    }

    void VectorSpan::fill_random_gaussian(float standard_deviation, uint64_t seed) noexcept
    {
        auto random = Random(seed);

        for (uint64_t i = 0; i < length; ++i)
        {
            _values[i] = random.gauss(standard_deviation);
        }
    }
}
#endif //NML_VECTOR_SPAN_H
