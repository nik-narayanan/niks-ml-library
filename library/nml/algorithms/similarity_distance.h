//
// Created by nik on 3/24/2024.
//

#ifndef NML_SIMILARITY_DISTANCE_H
#define NML_SIMILARITY_DISTANCE_H

#include <cmath>
#include "../primitives/vector_span.h"

// TODO vectorize

namespace nml
{
    enum class DistanceType : char
    {
        EUCLIDEAN,
        MANHATTAN,
        ANGULAR,
        HAMMING,
    };

    static inline float l2_norm(const VectorSpan& nums) noexcept
    {
        float result = 0;

        for (uint64_t i = 0; i < nums.length; ++i)
        {
            result += nums[i] * nums[i];
        }

        return sqrtf(result);
    }

    static inline float squared_euclidean_distance(const VectorSpan& left, const VectorSpan& right) noexcept
    {
        float result = 0;

        for (uint64_t i = 0; i < left.length; ++i)
        {
            float diff = left[i] - right[i];
            result += diff * diff;
        }

        return result;
    }

    static inline float euclidean_distance(const VectorSpan& left, const VectorSpan& right) noexcept
    {
        float result = 0;

        for (uint64_t i = 0; i < left.length; ++i)
        {
            float diff = left[i] - right[i];
            result += diff * diff;
        }

        return std::sqrt(result);
    }

    static inline float manhattan_distance(const VectorSpan& left, const VectorSpan& right) noexcept
    {
        float result = 0;

        for (uint64_t i = 0; i < left.length; ++i)
        {
            result += std::fabs(left[i] - right[i]);
        }

        return result;
    }

    static inline float hamming_distance(const VectorSpan& left, const VectorSpan& right) noexcept
    {
        float result = 0;

        for (uint64_t i = 0; i < left.length; ++i)
        {
            if (left[i] != right[i]) ++result;
        }

        return result;
    }

    static inline float angular_distance(const VectorSpan& left, const VectorSpan& right) noexcept
    {
        float result = 0;

        float left_norm = std::max(l2_norm(left), static_cast<float>(1e-20));
        float right_norm = std::max(l2_norm(right), static_cast<float>(1e-20));

        for (uint64_t i = 0; i < left.length; ++i)
        {
            result += left[i] * right[i];
        }

        return std::sqrt(2.0f - 2.0f * result / left_norm / right_norm);
    }

    static inline float calculate_distance(const VectorSpan& left, const VectorSpan& right, const DistanceType distance_type) noexcept
    {
        switch (distance_type)
        {
            case DistanceType::ANGULAR: return angular_distance(left, right);
            case DistanceType::HAMMING: return hamming_distance(left, right);
            case DistanceType::MANHATTAN: return manhattan_distance(left, right);
            case DistanceType::EUCLIDEAN: default: return euclidean_distance(left, right);
        }
    }
}

#endif //NML_SIMILARITY_DISTANCE_H
