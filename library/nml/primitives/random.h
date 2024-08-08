//
// Created by nik on 4/7/2024.
//

#ifndef NML_RANDOM_H
#define NML_RANDOM_H

#include <cmath>

struct Random
{
    unsigned long long x;
    unsigned long long y;
    unsigned long long z;
    unsigned long long c;

    static const unsigned long long default_seed = 1234567890987654321ULL;

    explicit Random(unsigned long long seed = default_seed)
    {
        x = seed == 0 ? default_seed : seed;
        y = 362436362436362436ULL;
        z = 1066149217761810ULL;
        c = 123456123456123456ULL;
    }

    unsigned long long kiss()
    {
        z = 6906969069LL*z+1234567;

        y ^= (y<<13);
        y ^= (y>>17);
        y ^= (y<<43);

        unsigned long long t = (x << 58) + c;

        c = x >> 6;
        x += t;
        c += x < t;

        return x + y + z;
    }

    inline bool flip()
    {
        return kiss() & 1;
    }

    inline unsigned long long index(unsigned long long n)
    {
        return kiss() % n;
    }

    inline float value(float min, float max)
    {
        const double factor = 1e8;

        auto large_value = kiss() % static_cast<unsigned long long>((max - min) * factor);

        double truncated_value = static_cast<double>(large_value) / factor;

        return min + static_cast<float>(truncated_value);
    }

    inline float gauss(float standard_deviation)
    {
        const float factor = 1e8;
        const float pi = 3.1415926535897932384626433832795028841971693993751;

        float rand_1 = static_cast<float>(kiss() % static_cast<unsigned>(factor)) / factor;
        float rand_2 = static_cast<float>(kiss() % static_cast<unsigned>(factor)) / factor;

        float value = std::sqrt(-2 * std::log(rand_1)) * std::cos(2 * pi * rand_2);

        return value * standard_deviation;
    }

    inline void set_seed(unsigned long long seed)
    {
        x = seed;
    }
};

#endif //NML_RANDOM_H