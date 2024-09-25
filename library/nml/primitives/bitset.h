//
// Created by nik on 4/6/2024.
//

#ifndef NML_BITSET_H
#define NML_BITSET_H

#include "span.h"
#include "memory_span.h"

#if defined(_MSC_VER)
#include <intrin.h>
    #define COUNT_LEADING_ZEROS(x) _lzcnt_u32(x)
#elif defined(__GNUC__) || defined(__clang__)
    #define COUNT_LEADING_ZEROS(x) __builtin_clz(x)
#endif

#if defined(_MSC_VER)
#include <intrin.h>
    #define COUNT_TRAILING_ZEROS(x) _tzcnt_u32(x)
#elif defined(__GNUC__) || defined(__clang__)
    #define COUNT_TRAILING_ZEROS(x) __builtin_ctz(x)
#endif

#if defined(_MSC_VER)
#include <intrin.h>
    #define POP_COUNT(x) __popcnt64(x)
#elif defined(__GNUC__) || defined(__clang__)
    #define POP_COUNT(x) __builtin_popcountll(x)
#endif

namespace nml
{
    class Bitset
    {
        Span<char> _memory;

    public:

        Bitset() noexcept : _memory(Span<char>(nullptr, 0)) { }

        explicit Bitset(MemorySpan memory, bool initialize = true) noexcept : _memory(Span<char>(memory, 0))
        {
            if (initialize)
            {
                reset_all();
            }
        }

        static inline unsigned required_bytes(unsigned length) noexcept
        {
            return (length + 7) / 8;
        }

        inline bool any() noexcept
        {
            for (const char byte : _memory)
            {
                if (byte != 0) return true;
            }

            return false;
        }

        void set(unsigned offset) noexcept
        {
            if (offset < _memory.length * 8) set_unsafe(offset);
        }

        void inline set_unsafe(unsigned offset) noexcept
        {
            _memory[offset / 8] |= (1 << (offset % 8));
        }

        void inline set_all() noexcept
        {
            memset(_memory.get_pointer(0), 255, _memory.bytes());
        }

        void reset(unsigned offset) noexcept
        {
            if (offset < _memory.length * 8) reset_unsafe(offset);
        }

        void inline reset_unsafe(unsigned offset) noexcept
        {
            _memory[offset / 8] &= ~(1 << (offset % 8));
        }

        void inline reset_all() noexcept
        {
            memset(_memory.get_pointer(0), 0, _memory.bytes());
        }

        [[nodiscard]] bool check(unsigned offset) const noexcept
        {
            if (offset < _memory.length * 8)
            {
                return check_unsafe(offset);
            }

            return false;
        }

        [[nodiscard]] bool inline check_unsafe(unsigned offset) const noexcept
        {
            return (_memory[offset / 8] & (1 << (offset % 8))) != 0;
        }
    };
}

#endif //NML_BITSET_H