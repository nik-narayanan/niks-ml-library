//
// Created by nik on 4/1/2024.
//

#ifndef NML_MEMORY_SPAN_H
#define NML_MEMORY_SPAN_H

#ifndef alloca
#define alloca(upper_limit) __builtin_alloca((upper_limit))
#endif // !alloca

#include <cstring>
#include <string_view>

namespace nml
{
    class MemorySpan
    {
        class Iterator;
        char* _memory;

    public:

        const uint64_t bytes;

        explicit MemorySpan(const uint64_t bytes) noexcept
            : bytes(bytes), _memory(static_cast<char*>(std::malloc(bytes)))
        { }

        explicit MemorySpan(void* memory, const uint64_t bytes) noexcept
            : bytes(bytes), _memory(static_cast<char*>(memory))
        { }

        Iterator begin() const noexcept;
        Iterator end() const noexcept;

        [[nodiscard]] void* get_pointer(const uint64_t byte_offset = 0) const noexcept
        {
            return _memory + byte_offset;
        }

        template<typename T> [[nodiscard]] T* get_pointer(const uint64_t byte_offset) const noexcept
        {
            return reinterpret_cast<T*>(&_memory[byte_offset]);
        }

        [[nodiscard]] MemorySpan to_subspan_unsafe(const uint64_t byte_offset, const uint64_t subspan_bytes) const noexcept
        {
            return MemorySpan(get_pointer(byte_offset), subspan_bytes);
        }

        [[nodiscard]] MemorySpan offset(const uint64_t byte_offset) const noexcept
        {
            return MemorySpan(get_pointer(byte_offset), bytes - byte_offset);
        }

        inline void zero() noexcept
        {
            memset(_memory, 0, bytes);
        }

        inline void fill(char value) noexcept
        {
            memset(_memory, value, bytes);
        }

        bool operator==(const MemorySpan& other) const noexcept
        {
            if (bytes != other.bytes) return false;
            if (_memory == other._memory) return true;

            return std::equal(_memory, _memory + bytes, other._memory);
        }

        bool operator!=(const MemorySpan& other) const noexcept
        {
            return !(*this == other);
        }

        uint64_t hash() const noexcept;
    };

    class MemorySpan::Iterator
    {
        uint64_t _position;
        MemorySpan _memory;

    public:

        explicit Iterator(MemorySpan memory, uint64_t position = 0)
            : _memory(memory), _position(position)
        { }

        Iterator begin() const { return Iterator(_memory, 0); }
        Iterator end() const { return Iterator(_memory, _memory.bytes); }

        Iterator& operator++()
        {
            ++_position;

            return *this;
        }

        const unsigned char operator*() const
        {
            return *_memory.get_pointer<unsigned char>(_position);
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

    MemorySpan::Iterator MemorySpan::begin() const noexcept
    {
        return MemorySpan::Iterator(*this);
    }

    MemorySpan::Iterator MemorySpan::end() const noexcept
    {
        return MemorySpan::Iterator(*this, bytes);
    }

    uint64_t MemorySpan::hash() const noexcept
    {
        return std::hash<std::string_view>{}(std::string_view(_memory, bytes));
    }

    struct RequiredMemory
    {
        uint64_t result_required_bytes;
        uint64_t working_required_bytes;

        [[nodiscard]] uint64_t total_bytes() const noexcept
        {
            return result_required_bytes + working_required_bytes;
        }
    };

    struct RequestMemory
    {
        MemorySpan result_memory;
        MemorySpan working_memory;

        [[nodiscard]] inline bool is_sufficient(const RequiredMemory& required) const noexcept
        {
            return required.working_required_bytes <= working_memory.bytes
                && required.result_required_bytes <= result_memory.bytes;
        }

        static RequestMemory from_required_unsafe(RequiredMemory required, const MemorySpan memory, uint64_t byte_offset = 0) noexcept
        {
            return RequestMemory
            {
                .result_memory = memory.to_subspan_unsafe(byte_offset, required.result_required_bytes),
                .working_memory = memory.to_subspan_unsafe(byte_offset + required.result_required_bytes, required.working_required_bytes),
            };
        }
    };
}

namespace std { template<> struct hash<nml::MemorySpan> { size_t operator()(const nml::MemorySpan& ms) const noexcept { return ms.hash(); } }; }

#endif //NML_MEMORY_SPAN_H