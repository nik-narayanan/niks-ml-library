//
// Created by nik on 4/6/2024.
//

#ifndef NML_BITSET_H
#define NML_BITSET_H

#include "span.h"
#include "memory_span.h"

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
            char* start = _memory.get_pointer();
            const char* end = _memory.get_end();

            while (start < end)
            {
                if (*start != 0)
                {
                    return true;
                }

                start++;
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