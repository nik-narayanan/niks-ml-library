//
// Created by nik on 4/1/2024.
//

#ifndef NML_MEMORY_OWNER_H
#define NML_MEMORY_OWNER_H


#include "span.h"
#include "vector_span.h"

namespace nml
{
    class MemoryOwner
    {
        void* _memory;

    public:

        const unsigned bytes;

        explicit MemoryOwner(const unsigned bytes) : bytes(bytes)
        {
            _memory = std::malloc(bytes);
            if (!_memory) std::abort();
        }

        ~MemoryOwner()
        {
            std::free(_memory);
        }

        MemoryOwner(MemoryOwner&& other) = delete;
        MemoryOwner(const MemoryOwner& other) = delete;
        MemoryOwner& operator=(const MemoryOwner&) = delete;
        MemoryOwner& operator=(const MemoryOwner&&) = delete;

        [[nodiscard]] void* get_pointer(const unsigned byte_offset) const
        {
            return static_cast<char*>(_memory) + byte_offset;
        }

        template<typename T> [[nodiscard]] T* get_pointer(const unsigned byte_offset) const
        {
            return reinterpret_cast<T*>(static_cast<char*>(_memory) + byte_offset);
        }

        template<typename T> [[nodiscard]] Span<T> to_span_unsafe(const unsigned byte_offset, const unsigned t_length) const
        {
            return Span<T>(get_pointer<T>(byte_offset), t_length);
        }

        [[nodiscard]] VectorSpan to_vector_span_unsafe(const unsigned byte_offset, const unsigned float_length) const
        {
            return VectorSpan(get_pointer<float>(byte_offset), float_length);
        }

        [[nodiscard]] MemorySpan to_memory_span(const unsigned byte_offset = 0, const unsigned memory_size = 0) const
        {
            return MemorySpan(get_pointer(byte_offset), memory_size == 0 ? (bytes - byte_offset) : memory_size);
        }

        [[nodiscard]] RequestMemory to_request_memory(const RequiredMemory& required, unsigned byte_offset = 0) const
        {
            return RequestMemory
            {
                .result_memory = to_memory_span(byte_offset, required.result_required_bytes),
                .working_memory = to_memory_span(byte_offset + required.result_required_bytes, required.working_required_bytes),
            };
        }
    };
}

#endif //NML_MEMORY_OWNER_H
