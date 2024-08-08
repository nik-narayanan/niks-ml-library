//
// Created by nik on 3/30/2024.
//

#ifndef NML_DATASETS_H
#define NML_DATASETS_H

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif // NOMINMAX
#include <windows.h>
#define WSMAN_API_VERSION_1_0
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define INVALID_HANDLE_VALUE -1
#endif // _WIN32

#include <fstream>
#include <sstream>
#include <filesystem>
#include <wsman.h>

//#include "matrix_owner.h"

namespace nml
{
    struct DataReader
    {
        struct DelimitedFileIterator;
        enum struct DelimitedTokenType : char;

        static DelimitedFileIterator delimited_iterator(const char* path, char delimiter = ',');
    };
}

namespace nml::file_internal
{
    #ifdef _WIN32
        using FileHandle = HANDLE;
    #else
        using FileHandle = int;
    #endif

    struct MemoryMappedInstance
    {
        char* data;
        uint64_t length;
        FileHandle handle;
        #ifdef _WIN32
            FileHandle mapped_handle;
        #endif
    };

    static inline uint64_t get_os_page_size() noexcept
    {
        #ifdef _WIN32
            SYSTEM_INFO SystemInfo;
            GetSystemInfo(&SystemInfo);
            return SystemInfo.dwAllocationGranularity;
        #else
            return sysconf(_SC_PAGE_SIZE);
        #endif
    }

    static inline uint64_t page_size() noexcept
    {
        static const uint64_t page_size = get_os_page_size();

        return page_size;
    }

    static inline uint64_t make_offset_page_aligned(uint64_t offset) noexcept
    {
        const uint64_t _page_size = page_size();

        return offset / _page_size * _page_size;
    }

    static inline FileHandle open_file(const char* path, bool read_only = true)
    {
        #ifdef _WIN32
            const auto handle = ::CreateFileA(
                path,
                read_only ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                nullptr,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                nullptr
            );
        #else // POSIX
            const auto handle = ::open(c_str(path), read_only ? O_RDONLY : O_RDWR);
        #endif

        if (handle == INVALID_HANDLE_VALUE)
        {
            throw std::filesystem::filesystem_error(
                "Failed to open file.",
                std::filesystem::path(path),
                std::make_error_code(std::errc::io_error)
            );
        }

        return handle;
    }

    static inline uint64_t query_file_size(FileHandle handle)
    {
        #ifdef _WIN32
            LARGE_INTEGER file_size;

            if (::GetFileSizeEx(handle, &file_size) == 0)
            {
                throw std::system_error(
                    std::error_code(static_cast<int32_t>(::GetLastError()), std::system_category()),
                    "Failed to get file size."
                );
            }

            return static_cast<int64_t>(file_size.QuadPart);
        #else // POSIX
            struct stat sbuf;

            if (::fstat(handle, &sbuf) == -1)
            {
                throw std::system_error(
                    std::error_code(errno, std::system_category()),
                    "Failed to get file size."
                );
            }

            return sbuf.st_size;
        #endif
    }

    static inline MemoryMappedInstance open_memory_map_instance(FileHandle handle, uint64_t file_size, bool read_only = true)
    {
        #ifdef _WIN32
            const auto windows_handle = ::CreateFileMapping(
                handle,
                nullptr,
                read_only ? PAGE_READONLY : PAGE_READWRITE,
                static_cast<DWORD>(file_size >> 32),
                static_cast<DWORD>(file_size & 0xffffffff),
                nullptr
            );

            if (windows_handle == INVALID_HANDLE_VALUE)
            {
                throw std::system_error(
                    std::error_code(static_cast<int32_t>(::GetLastError()), std::system_category()),
                    "Failed to memory map file."
                );
            }

            auto mapping_start = static_cast<char*>(::MapViewOfFile(
                windows_handle,
                read_only ? FILE_MAP_READ : FILE_MAP_WRITE,
                0,
                0,
                file_size
            ));

            if (mapping_start == nullptr)
            {
                ::CloseHandle(windows_handle);

                throw std::system_error(
                    std::error_code(static_cast<int32_t>(::GetLastError()), std::system_category()),
                    "Failed to claim memory map start."
                );
            }
        #else // POSIX
            auto mapping_start = static_cast<char >(::mmap(
                0,
                file_size,
                read_only ? PROT_READ : PROT_WRITE,
                MAP_SHARED,
                handle,
                0
            ));

            if (mapping_start == MAP_FAILED)
            {
                throw std::system_error(
                    std::error_code(errno, std::system_category()),
                    "Failed to claim memory map start."
                );
            }
        #endif

        return MemoryMappedInstance
        {
            .data = mapping_start,
            .length = file_size,
            .handle = handle,
            #ifdef _WIN32
                .mapped_handle = windows_handle,
            #endif
        };
    }

    static inline MemoryMappedInstance open_memory_map_instance(const char* path, bool read_only = true)
    {
        FileHandle handle = open_file(path, read_only);
        uint64_t file_size = query_file_size(handle);

        return open_memory_map_instance(handle, file_size, read_only);
    }

    static inline void close_memory_map_instance(MemoryMappedInstance& instance)
    {
        if (instance.data != nullptr)
        {
            #ifdef _WIN32
                ::UnmapViewOfFile(instance.data);

                if (instance.mapped_handle != INVALID_HANDLE_VALUE)
                {
                    ::CloseHandle(instance.mapped_handle);
                }
            #else
                ::munmap(instance.data, instance.length);
            #endif

            instance.length = 0;
            instance.data = nullptr;
        }

        if (instance.handle != INVALID_HANDLE_VALUE)
        {
            #ifdef _WIN32
                ::CloseHandle(instance.handle);
            #else
                ::close(instance.handle);
            #endif
        }
    }
}

/*******************************************************************************************************************
 DELIMITED
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::file_internal;

    enum struct DataReader::DelimitedTokenType : char
    {
        VALUE,
        LINE_END,
        STRING_END,
        STRING_START,
    };

    struct DataReader::DelimitedFileIterator
    {
        explicit DelimitedFileIterator(const char* path, char delimiter = ',')
            : _delimiter(delimiter), _offset(0)
            , _memory(open_memory_map_instance(path, true))
            , _next({ .type = DelimitedTokenType::STRING_START, .length = 0, .start = nullptr })
        { }

        ~DelimitedFileIterator()
        {
            close_memory_map_instance(_memory);
        }

        struct Summary
        {
            uint64_t row_ct;
            uint64_t column_ct;
        };

        [[nodiscard]] inline Summary summary() const noexcept
        {
            Summary summary
            {
                .row_ct = 0,
                .column_ct = _memory.length > 0
            };

            // TODO simd

            bool is_escaped_double = false;
            bool is_escaped_single = false;

            for (uint64_t offset = 0; offset < _memory.length; ++offset)
            {
                char ch = _memory.data[offset];

                if (ch == '"')
                {
                    is_escaped_double = !is_escaped_double;
                }
                else if (ch == '\'')
                {
                    is_escaped_single = !is_escaped_single;
                }
                else if (ch == '\n' && !is_escaped_double && !is_escaped_single)
                {
                    summary.row_ct += 1;
                }
                else if (!summary.row_ct && ch == _delimiter && !is_escaped_double && !is_escaped_single)
                {
                    summary.column_ct += 1;
                }
            }

            return summary;
        }

        struct Token
        {
            DelimitedTokenType type;
            uint32_t length;
            const char* start;
        };

        [[nodiscard]] inline int64_t length() const noexcept
        {
            return static_cast<int64_t>(_memory.length);
        }

        [[nodiscard]] inline bool is_end() const noexcept
        {
            return _offset >= _memory.length;
        }

        [[nodiscard]] inline Token next() const noexcept
        {
            return _next;
        }

        [[nodiscard]] inline bool has_next() noexcept
        {
            if (is_end())
            {
                _next.type = DelimitedTokenType::STRING_END;

                return false;
            }

            if (_memory.data[_offset] == '\n')
            {
                _next.type = DelimitedTokenType::LINE_END;

                _offset += 1;

                return true;
            }

            _next.type = DelimitedTokenType::VALUE;
            _next.start = &_memory.data[_offset];

            // TODO simd

            bool is_escaped_double = false;
            bool is_escaped_single = false;

            for (; _offset < _memory.length; ++_offset)
            {
                char ch = _memory.data[_offset];

                if (ch == '"')
                {
                    is_escaped_double = !is_escaped_double;
                }
                else if (ch == '\'')
                {
                    is_escaped_single = !is_escaped_single;
                }
                else if ((ch == '\n' || ch == '\r') && !is_escaped_double && !is_escaped_single)
                {
                    _next.length = &_memory.data[_offset] - _next.start;
                    _offset += ch == '\r';
                    break;
                }
                else if (ch == _delimiter && !is_escaped_double && !is_escaped_single)
                {
                    _next.length = &_memory.data[_offset++] - _next.start;
                    break;
                }
            }

            return true;
        }

        inline void reset() noexcept
        {
            _offset = 0;
            _next = { .type = DelimitedTokenType::STRING_START, .length = 0, .start = nullptr };
        }

    private:

        const char _delimiter;

        Token _next;
        uint64_t _offset;
        MemoryMappedInstance _memory;
    };

    DataReader::DelimitedFileIterator DataReader::delimited_iterator(const char* path, char delimiter)
    {
        return DelimitedFileIterator(path, delimiter);
    }
}

#endif //NML_DATASETS_H