//
// Created by nik on 6/1/2024.
//

#ifndef NML_DATA_FRAME_H
#define NML_DATA_FRAME_H

#include <cmath>
#include <string>
#include <chrono>
#include <iostream>

#include "file.h"
#include "heap.h"
#include "hash.h"
#include "list.h"
#include "bitset.h"
#include "allocator.h"
#include "matrix_owner.h"
#include "matrix_result.h"

#include "../external/date.h"

namespace nml::dataframe_internal
{
    struct Column;

    constexpr uint64_t DEFAULT_RESIZE_FACTOR = 2;
    constexpr uint64_t STARTING_COLUMN_LENGTH = 10;
    constexpr uint32_t MAX_U32 = std::numeric_limits<uint32_t>::max();

    typedef std::chrono::milliseconds Ticks;
    typedef date::sys_time<std::chrono::milliseconds> DateTime;
}

namespace nml
{
    using namespace nml::dataframe_internal;

    struct DataFrame
    {
        struct Value;
        struct RowView;
        struct ColumnView;
        struct ReferenceValue;

        ~DataFrame() noexcept;
        explicit DataFrame() noexcept;
        DataFrame(DataFrame&&) noexcept;
        DataFrame(const DataFrame&) noexcept;
        DataFrame& operator=(DataFrame&&) noexcept;
        DataFrame& operator=(const DataFrame&) noexcept;

        static DataFrame from_delimited(const char* path, bool has_headers = true, char delimiter = ',');
        static DataFrame from_delimited(const std::string& path, bool has_headers = true, char delimiter = ',');

        MatrixOwner to_matrix();
        MatrixOwner to_matrix(Span<uint32_t> columns);
        MatrixOwner to_matrix(std::initializer_list<uint32_t> columns);

        enum class Type : char { INT, FLOAT, STRING, DATETIME };

        [[nodiscard]] inline uint64_t row_count() const noexcept { return _row_count; }
        [[nodiscard]] inline uint32_t column_count() const noexcept { return _column_shifts.count; }

        inline RowView operator[](uint64_t row);
        inline ReferenceValue operator()(uint64_t row, uint32_t column);

        inline RowView get_row_unsafe(uint64_t row) noexcept;
        inline NMLResult<RowView> get_row(uint64_t row) noexcept;
        inline ColumnView get_column_unsafe(uint32_t offset) noexcept;
        inline NMLResult<ColumnView> get_column(uint32_t offset) noexcept;
        inline ColumnView get_column_by_name_unsafe(const char* name) noexcept;
        inline ColumnView get_column_by_name_unsafe(const std::string&) noexcept;
        inline NMLResult<ColumnView> get_column_by_name(const char* name) noexcept;
        inline NMLResult<ColumnView> get_column_by_name(const std::string&) noexcept;

        uint64_t add_row(Span<Value> row);
        uint64_t add_row(std::initializer_list<Value> row);

        // TODO factorize inplace
        // TODO pass in resizable list
        ColumnView factorize_column(ColumnView column, uint32_t offset = MAX_U32);
        ColumnView factorize_column(ColumnView column, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView factorize_column(ColumnView column, Span<const char> column_name, uint32_t offset = MAX_U32);

        ColumnView factorize_column_with_distinct(ColumnView column, ResizableList<int64_t>* distinct, uint32_t offset = MAX_U32);
        ColumnView factorize_column_with_distinct(ColumnView column, const char* column_name, ResizableList<int64_t>* distinct, uint32_t offset = MAX_U32);
        ColumnView factorize_column_with_distinct(ColumnView column, Span<const char> column_name, ResizableList<int64_t>* distinct, uint32_t offset = MAX_U32);

        void remove_column(uint32_t offset) noexcept;
        void remove_column(const char* column_name) noexcept;
        void remove_column(Span<const char> column_name) noexcept;

        ColumnView add_column(Type type, uint32_t offset = MAX_U32); // Adds all nulls
        ColumnView add_column(const Value& value, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<double> values, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<int64_t> values, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<DateTime> dates, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<char*> c_strings, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<std::string> strings, uint32_t offset = MAX_U32);
        ColumnView add_column(ColumnView column_to_copy, uint32_t offset = MAX_U32);

        ColumnView add_column(Type type, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(const Value& value, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<double> values, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<int64_t> values, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<DateTime> dates, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<char*> c_strings, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<std::string> strings, const char* column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(ColumnView column_to_copy, const char* column_name, uint32_t offset = MAX_U32);

        ColumnView add_column(Type type, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(const Value& value, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<double> values, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<int64_t> values, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<DateTime> dates, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<char*> c_strings, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(Span<std::string> strings, Span<const char> column_name, uint32_t offset = MAX_U32);
        ColumnView add_column(ColumnView column_to_copy, Span<const char> column_name, uint32_t offset = MAX_U32);

        void print(uint64_t rows = 0);
        
    private:

        uint64_t _row_count;
        ResizableList<Column> _columns;
        ResizableList<uint32_t> _column_shifts;

        uint32_t find_next_column_offset() noexcept;
        void throw_if_column_name_taken(const char* name);
        void throw_if_column_name_taken(Span<const char>);
        inline void initialize_allocated_columns() noexcept;
        int32_t find_column_offset(const char* name) noexcept;
        int32_t find_column_offset(Span<const char> name) noexcept;
        uint32_t add_column(Type type, bool nullable, uint64_t length, Span<const char> name, uint32_t offset);
        inline Column& get_raw_column_unsafe(uint32_t offset) noexcept { return _columns[_column_shifts[offset]]; }
    };
}

/*******************************************************************************************************************
 STRING
 ******************************************************************************************************************/
namespace nml::dataframe_internal
{
    constexpr uint32_t COLUMN_NAME_LENGTH = 30;
    constexpr uint32_t COLUMN_STRING_LENGTH = 24;
    constexpr uint32_t OVERFLOW_STRING_LENGTH = 60;

    struct String
    {
        uint32_t length;
        uint32_t overflow_index;

        char string[COLUMN_STRING_LENGTH];

        inline void initialize() noexcept
        {
            length = 0;
            overflow_index = 0;
        }
    };

    struct StringOverflow
    {
        uint32_t overflow_index{0};

        char string[OVERFLOW_STRING_LENGTH]{0};
    };

    class StringManager
    {
        Allocator<StringOverflow> _overflow;

    public:

        struct Iterator;
        struct Container;

        explicit StringManager(uint64_t initial_capacity = 0) noexcept
            : _overflow(Allocator<StringOverflow>(initial_capacity))
        { }

        void reset() noexcept;

        void return_node(uint32_t index) noexcept;
        [[nodiscard]] inline uint64_t get_next_available_node_index() noexcept;
        [[nodiscard]] inline StringOverflow& get_node(uint64_t index) noexcept;

        uint32_t store_remaining_string(const char* str) noexcept;
        void string_append(String& base_string, char* append) noexcept;
        void set_string(String& destination, const char* value) noexcept;
        void set_string(String& destination, Span<const char> value) noexcept;

        Container container(String& string) noexcept;
        Iterator get_iterator(String& string) noexcept;
        void string_copy(String& base_string, char* buffer) noexcept;
    };

    struct StringManager::Iterator
    {
        uint64_t _index;
        char _next =  0;
        int32_t _overflow;
        String* _string;
        StringManager* _manager;
        StringOverflow* _current_overflow;

        explicit Iterator(String* string, StringManager* manager, uint64_t index = 0)
            : _string(string), _manager(manager)
            , _index(index), _overflow(-1), _current_overflow(nullptr)
        { }

        [[nodiscard]] inline int64_t length() const noexcept
        {
            return _string->length;
        }

        [[nodiscard]] inline bool is_end() const noexcept
        {
            return _index >= length();
        }

        [[nodiscard]] inline char next() const noexcept
        {
            return _next;
        }

        [[nodiscard]] inline bool has_next() noexcept
        {
            if (is_end())
            {
                _next = 0;
                return false;
            }

            if (_index < COLUMN_STRING_LENGTH)
            {
                _next = _string->string[_index++];

                return true;
            }

            if ((_index - COLUMN_STRING_LENGTH) / OVERFLOW_STRING_LENGTH > _overflow)
            {
                _current_overflow = &_manager->get_node(_current_overflow->overflow_index);
            }

            _next = _current_overflow->string[(_index - 1 - COLUMN_STRING_LENGTH) % OVERFLOW_STRING_LENGTH];

            return true;
        }

        inline void reset() noexcept
        {
            _index = 0, _current_overflow = nullptr;
        }

        Iterator& operator++()
        {
            if (is_end()) return *this;

            if (_index < COLUMN_STRING_LENGTH)
            {
                _next = _string->string[_index++];

                return *this;
            }

            if ((_index - COLUMN_STRING_LENGTH) / OVERFLOW_STRING_LENGTH > _overflow)
            {
                _current_overflow = &_manager->get_node(_current_overflow->overflow_index);
            }

            _next = _current_overflow->string[(_index - 1 - COLUMN_STRING_LENGTH) % OVERFLOW_STRING_LENGTH];

            return *this;
        }

        inline std::string to_std_string() noexcept
        {
            auto copy = *this;

            copy.reset();

            std::string string;

            while (copy.has_next())
            {
                string += copy.next();
            }

            return string;
        }

        const char& operator*() const
        {
            return _next;
        }

        bool operator==(const Iterator& rhs) const
        {
            return _index == rhs._index;
        }

        bool operator!=(const Iterator& rhs) const
        {
            return _index != rhs._index;
        }
    };

    StringManager::Iterator StringManager::get_iterator(String& string) noexcept
    {
        return StringManager::Iterator(&string, this);
    }

    struct StringManager::Container
    {
        String* _string;
        StringManager* _manager;

        explicit Container(String* string, StringManager* manager)
            : _string(string), _manager(manager)
        { }

        [[nodiscard]] Iterator begin() const noexcept
        {
            return StringManager::Iterator(_string, _manager);
        }

        [[nodiscard]] Iterator end() const noexcept
        {
            return StringManager::Iterator(_string, _manager, _string->length);
        }

        bool operator==(const Container& rhs) const
        {
            if (_string->length != rhs._string->length) return false;

            for (Iterator l = begin(), r = rhs.begin(); l != end() && r != rhs.end(); ++l, ++r)
            {
                if (l._next != r._next) return false;
            }

            return true;
        }

        [[nodiscard]] uint64_t hash() const noexcept
        {
            uint64_t hash = 0xCBF29CE484222325;

            for (const unsigned char byte : *this)
            {
                hash ^= static_cast<uint64_t>(byte);
                hash *= 0x100000001B3;
            }

            return hash;
        }
    };

    StringManager::Container StringManager::container(String& string) noexcept
    {
        return StringManager::Container(&string, this);
    }

    void StringManager::string_copy(String& base_string, char* buffer) noexcept
    {
        uint32_t copied_characters = 0;

        uint32_t base_length = std::min(base_string.length, COLUMN_STRING_LENGTH);

        while (copied_characters < base_length)
        {
            buffer[copied_characters] = base_string.string[copied_characters++];
        }

        if (copied_characters >= base_string.length || base_string.overflow_index == 0)
        {
            buffer[copied_characters] = '0';

            return;
        }

        auto next_node = get_node(base_string.overflow_index);

        uint32_t characters_to_copy = std::min(base_string.length - copied_characters, OVERFLOW_STRING_LENGTH);

        for (uint32_t i = 0; i < characters_to_copy; ++i)
        {
            buffer[copied_characters++] = next_node.string[i];
        }

        if (copied_characters >= base_string.length || next_node.overflow_index == 0)
        {
            buffer[copied_characters] = '0';

            return;
        }

        while (copied_characters <= base_length)
        {
            next_node = get_node(next_node.overflow_index);

            characters_to_copy = std::min(base_string.length - copied_characters, OVERFLOW_STRING_LENGTH);

            for (uint32_t i = 0; i < characters_to_copy; ++i)
            {
                buffer[copied_characters++] = next_node.string[i];
            }
        }

        buffer[copied_characters] = '0';
    }

    void StringManager::string_append(String& base_string, char* append) noexcept
    {
        uint32_t append_length = strlen(append);

        uint32_t appended_characters = 0;

        if (base_string.length < COLUMN_STRING_LENGTH)
        {
            uint32_t base_length = strlen(base_string.string);

            uint32_t next_offset = appended_characters + base_length + 1;

            while (next_offset < COLUMN_STRING_LENGTH && appended_characters <= append_length)
            {
                base_string.string[next_offset++] = append[appended_characters++];
            }
        }

        if (appended_characters >= append_length)
        {
            base_string.length += append_length;

            return;
        }

        if (base_string.overflow_index == 0)
        {
            base_string.overflow_index = store_remaining_string(append + appended_characters);

            base_string.length += append_length;

            return;
        }

        auto last_node = get_node(base_string.overflow_index);

        while (last_node.overflow_index > 0)
        {
            last_node = get_node(last_node.overflow_index);
        }

        uint32_t last_node_used_space = (base_string.length - COLUMN_STRING_LENGTH) % OVERFLOW_STRING_LENGTH;

        base_string.length += append_length;

        while (last_node_used_space + 1 <= OVERFLOW_STRING_LENGTH && appended_characters <= append_length)
        {
            last_node.string[++last_node_used_space] = append[appended_characters++];
        }

        if (appended_characters < append_length)
        {
            last_node.overflow_index = store_remaining_string(append + appended_characters);
        }
    }

    uint32_t StringManager::store_remaining_string(const char* str) noexcept
    {
        uint64_t root_node_index = get_next_available_node_index();

        auto storage_node = get_node(root_node_index);

        uint32_t current_node_offset = 0;

        while (str)
        {
            if (current_node_offset + 1 >= OVERFLOW_STRING_LENGTH)
            {
                auto next_node_index = get_next_available_node_index();

                storage_node.overflow_index = next_node_index;

                storage_node = get_node(next_node_index);
            }

            storage_node.string[current_node_offset++] = *str++;
        }

        storage_node.overflow_index = 0;

        return root_node_index;
    }

    void StringManager::return_node(const uint32_t index) noexcept
    {
        if (index == 0) return;

        auto pointer = _overflow.get_element(index).overflow_index;

        _overflow.return_index(index);

        while (pointer > 0)
        {
            auto next = _overflow.get_element(pointer);
            _overflow.return_index(pointer);
            pointer = next.overflow_index;
        }
    }

    void StringManager::set_string(String& destination, Span<const char> value) noexcept
    {
        uint32_t characters_copied = 0;

        uint32_t string_length = value.length;

        destination.length = string_length;

        uint32_t characters_to_copy = std::min(string_length, COLUMN_STRING_LENGTH);

        memcpy(&destination.string, value.get_pointer(0), characters_to_copy);

        characters_copied += characters_to_copy;

        if (characters_copied <= string_length)
        {
            return_node(destination.overflow_index);

            return;
        }

        auto overflow_index = destination.overflow_index > 0 ?
                              destination.overflow_index :
                              get_next_available_node_index();

        destination.overflow_index = overflow_index;

        auto overflow_node = get_node(overflow_index);

        characters_to_copy = std::min(string_length - characters_copied, OVERFLOW_STRING_LENGTH);

        memcpy(&overflow_node.string, value.get_pointer(characters_copied), characters_to_copy);

        characters_copied += characters_to_copy;

        while (characters_copied < string_length)
        {
            overflow_index = overflow_node.overflow_index > 0 ?
                             overflow_node.overflow_index :
                             get_next_available_node_index();

            overflow_node.overflow_index = overflow_index;

            overflow_node = get_node(overflow_index);

            characters_to_copy = std::min(string_length - characters_copied, OVERFLOW_STRING_LENGTH);

            memcpy(&overflow_node.string, value.get_pointer(characters_copied), characters_to_copy);

            characters_copied += characters_to_copy;
        }
    }

    void StringManager::set_string(String &destination, const char *value) noexcept
    {
        auto span = Span<const char>(value, strlen(value) + 1);

        set_string(destination, span);
    }

    uint64_t StringManager::get_next_available_node_index() noexcept
    {
        return _overflow.claim_next_index();
    }

    StringOverflow &StringManager::get_node(uint64_t index) noexcept
    {
        return _overflow.get_element(index);
    }

    inline void StringManager::reset() noexcept
    {
        _overflow.reset();
    }
}

namespace std
{
    template<> struct hash<nml::dataframe_internal::StringManager::Container>
    {
        size_t operator()(const nml::dataframe_internal::StringManager::Container& str) const noexcept
        {
            return str.hash();
        }
    };
}

/*******************************************************************************************************************
 VALUE
 ******************************************************************************************************************/
namespace nml
{
    namespace dataframe_internal
    {
        std::string type_to_string(DataFrame::Type type)
        {
            switch (type)
            {
                case DataFrame::Type::INT: return "INT";
                case DataFrame::Type::FLOAT: return "FLOAT";
                case DataFrame::Type::STRING: return "STRING";
                case DataFrame::Type::DATETIME: return "DATETIME";
                default: throw std::invalid_argument("Unknown Type");
            }
        }

        static inline void imprint_integer(char* buffer, int number, bool terminate = true)
        {
            bool is_negative = number < 0;

            uint32_t width = is_negative ? 9 : 10;

            std::snprintf(buffer, 12, "%-*d", width, number);

            uint32_t used_length = std::snprintf(nullptr, 0, "%-*d", width, number);
            uint32_t remaining_spaces = 11 - used_length - 1;

            for (uint32_t i = used_length; i < remaining_spaces + used_length; ++i)
            {
                buffer[i] = ' ';
            }

            if (!terminate) buffer[used_length] = ' ';
            else buffer[remaining_spaces + used_length] = '\0';
        }

        static inline void imprint_int64_t(char* buffer, int64_t number, bool terminate = true)
        {
            bool is_negative = number < 0;

            uint32_t width = is_negative ? 18 : 19;

            std::snprintf(buffer, 21, "%-*lld", width, number);

            uint32_t used_length = std::snprintf(nullptr, 0, "%-*lld", width, number);
            uint32_t remaining_spaces = 20 - used_length - 1;

            for (uint32_t i = used_length; i < remaining_spaces + used_length; ++i)
            {
                buffer[i] = ' ';
            }

            if (!terminate) buffer[used_length] = ' ';
            else buffer[remaining_spaces + used_length] = '\0';
        }

        static inline void imprint_float(char* buffer, double number, bool terminate = true) noexcept
        {
            double abs = std::fabs(number);

            char starting_end_char = buffer[11];

            if ((abs < 1000 && abs > 0.001) || abs == 0)
            {
                std::snprintf(buffer, 12, "%-11.6g", number);
            }
            else
            {
                std::snprintf(buffer, 12, "%-11.4e", number);
            }

            if (!terminate) buffer[11] = starting_end_char;
        }

        static inline void imprint_double(char* buffer, double number, bool terminate = true) noexcept
        {
            double abs = std::fabs(number);

            char starting_end_char = buffer[19];

            if ((abs < 1000 && abs > 0.001) || abs == 0)
            {
                std::snprintf(buffer, 20, "%-19.6g", number);
            }
            else
            {
                std::snprintf(buffer, 20, "%-19.4e", number);
            }

            if (!terminate) buffer[19] = starting_end_char;
            else buffer[19] = '\0';
        }

        static inline void imprint_date_time(char* buffer, DateTime date_time) noexcept
        {
            std::time_t time = std::chrono::system_clock::to_time_t(date_time);

            std::tm* utc_tm = std::gmtime(&time);

            std::strftime(buffer, 21, "%Y-%m-%dT%H:%M:%SZ", utc_tm);
        }

        static inline bool is_whitespace(const char ch) noexcept
        {
            return ch ==  ' ' || ch == '\t' ||
                   ch == '\n' || ch == '\r' ||
                   ch == '\f' || ch == '\v' ;
        }

        static inline bool is_letter(const char ch) noexcept
        {
            return (ch >= 'a' && ch <= 'z') ||
                   (ch >= 'A' && ch <= 'Z') ;
        }

        static inline bool is_number(const char ch) noexcept
        {
            return ch >= '0' && ch <= '9';
        }

        static inline bool validate_date_time(
            uint16_t year, uint16_t month, uint16_t day,
            uint16_t hour, uint16_t minute, uint16_t second, uint16_t millisecond
        )
        {
            if (day == 0) return false;
            if (hour > 23) return false;
            if (year > 9999) return false;
            if (minute > 59) return false;
            if (second > 59) return false;
            if (millisecond > 999) return false;
            if (month == 0 || month > 12) return false;

            static const int days_in_month[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

            bool is_leap_year = (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0));

            if (!is_leap_year && day > days_in_month[month + 1]) return false;

            if (is_leap_year && month == 2 && day > 29) return false;

            return true;
        }

        static inline DateTime construct_date_time(
            uint16_t year, uint16_t month, uint16_t day,
            uint16_t hour, uint16_t minute, uint16_t second, uint16_t millisecond
        )
        {
            auto dp = date::year{ year } / month / day;

            auto tp = std::chrono::hours{ hour }
                      + std::chrono::minutes{ minute }
                      + std::chrono::seconds{ second }
                      + std::chrono::milliseconds{ millisecond };

            return DateTime(date::sys_days(dp) + tp);
        }

        static inline Span<const char> trim_string_span(Span<const char> string) noexcept
        {
            uint64_t offset = 0, end = string.length;

            while (offset < string.length && (is_whitespace(string[offset]) || string[offset] == '"'))
            {
                offset += 1;
            }

            while (end - 1 > offset && (is_whitespace(string[end - 1]) || string[end - 1] == '"'))
            {
                end -= 1;
            }

            if (end - offset == 0)
            {
                return string;
            }

            return string.to_subspan_unsafe(offset, end - offset);
        }

        // TODO parsing is garbo
        static inline double parse_float(StringManager::Iterator& string)
        {
            double parsed = 0, adjustment = 0.1;

            if (!string.has_next())
            {
                throw std::invalid_argument("Unable to parse float from empty string."
                    "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            while ((is_whitespace(string.next()) || string.next() == '"') && string.has_next()) { }

            bool is_negative = string.next() == '-' && string.has_next();

            do
            {
                if (!is_number(string.next())) break;

                parsed = parsed * 10.0f + static_cast<double>(string.next() - '0');
            }
            while (string.has_next());

            if (string.next() == '.' && string.has_next()) { }

            do
            {
                if (!is_number(string.next())) break;

                parsed += static_cast<double>(string.next() - '0') * adjustment;

                adjustment *= 0.1;
            }
            while (string.has_next());

            if (string.next() == 'e' || string.next() == 'E')
            {
                int32_t power = 0;

                bool is_negative_power = string.next() == '-' && string.has_next();

                do
                {
                    if (!is_number(string.next())) break;

                    power = power * 10 + (string.next() - '0');
                }
                while (string.has_next());

                if (is_negative_power) power *= -1;

                parsed *= static_cast<double>(std::pow(10.0f, power));
            }

            while ((is_whitespace(string.next()) || string.next() == '"') && string.has_next()) { }

            if (!string.is_end())
            {
                throw std::invalid_argument("Unable to parse float from string: " + string.to_std_string()
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            return is_negative ? -parsed : parsed;
        }

        static inline int64_t parse_integer(StringManager::Iterator& string)
        {
            int64_t parsed = 0, offset = 0;

            if (!string.has_next())
            {
                throw std::invalid_argument("Unable to parse integer from empty string."
                    "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            do
            {
                if (!(is_whitespace(string.next()) || string.next() == '"')) break;

                offset += 1;
            }
            while (string.has_next());

            bool is_negative = string.next() == '-' && string.has_next();

            uint64_t max_length = 19 + offset;

            do
            {
                if (!is_number(string.next())) break;

                parsed = parsed * 10 + (string.next() - '0'), offset += 1;
            }
            while (string.has_next() && offset < max_length);

            do
            {
                if (!(is_whitespace(string.next()) || string.next() == '"')) break;

                offset += 1;
            }
            while (string.has_next());

            if (offset < string.length())
            {
                throw std::invalid_argument("Unable to parse integer from string: " + string.to_std_string()
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            return is_negative ? -parsed : parsed;
        }

        static inline Result<DateTime, bool> parse_datetime(StringManager::Iterator& string, int64_t starting = -1)
        {
            uint16_t year = 0, month = 0, day = 0;
            uint16_t hour = 0, minute = 0, second = 0, millisecond = 0;

            if (starting < 0)
            {
                starting = 0;

                while (string.has_next() && is_number(string.next()))
                {
                    starting = starting * 10 + (string.next() - '0');
                }
            }

            if (starting > 31 && starting < 10'000)
            {
                year = starting;

                if (string.is_end() || !(string.next() == '/' || string.next() == '-'))
                {
                    return Result<DateTime, bool>::err(false);
                }

                while (string.has_next() && is_number(string.next()))
                {
                    month = month * 10 + (string.next() - '0');
                }

                if (string.is_end() || !(string.next() == '/' || string.next() == '-'))
                {
                    return Result<DateTime, bool>::err(false);
                }

                while (string.has_next() && is_number(string.next()))
                {
                    day = day * 10 + (string.next() - '0');
                }
            }
            else if (starting <= 12)
            {
                month = starting;

                if (string.is_end() || !(string.next() == '/' || string.next() == '-'))
                {
                    return Result<DateTime, bool>::err(false);
                }

                while (string.has_next() && is_number(string.next()))
                {
                    day = day * 10 + (string.next() - '0');
                }

                if (string.is_end() || !(string.next() == '/' || string.next() == '-'))
                {
                    return Result<DateTime, bool>::err(false);
                }

                while (string.has_next() && is_number(string.next()))
                {
                    year = year * 10 + (string.next() - '0');
                }
            }
            else
            {
                return Result<DateTime, bool>::err(false);
            }

            if (!string.is_end() && !(is_whitespace(string.next()) || string.next() == 'T'))
            {
                return Result<DateTime, bool>::err(false);
            }

            while (string.has_next() && is_number(string.next()))
            {
                hour = hour * 10 + (string.next() - '0');
            }

            if (!string.is_end() && string.next() != ':')
            {
                return Result<DateTime, bool>::err(false);
            }

            while (string.has_next() && is_number(string.next()))
            {
                minute = minute * 10 + (string.next() - '0');
            }

            if (!string.is_end() && string.next() != ':')
            {
                return Result<DateTime, bool>::err(false);
            }

            while (string.has_next() && is_number(string.next()))
            {
                second = second * 10 + (string.next() - '0');
            }

            if (!string.is_end() && !(string.next() == '.' || string.next() == ':' || string.next() == 'Z'))
            {
                return Result<DateTime, bool>::err(false);
            }

            while (string.has_next() && is_number(string.next()))
            {
                millisecond = millisecond * 10 + (string.next() - '0');
            }

            while (!string.is_end() && string.next() == 'Z' && string.has_next()) { }

            while (!string.is_end() && is_whitespace(string.next()) && string.has_next()) { }

            if (!string.is_end())
            {
                return Result<DateTime, bool>::err(false);
            }

            if (!validate_date_time(year, month, day, hour, minute, second, millisecond))
            {
                throw std::invalid_argument("Unable to parse DateTime from string: " + string.to_std_string()
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            return Result<DateTime, bool>::ok(
                construct_date_time(year, month, day, hour, minute, second, millisecond));
        }

        // TODO actually put some thought into this

        using TypeCastFunction = void(*)(void*);
        using StringCastFunction = uint32_t(*)(char*, void*);
        using TypeCastArrayFunction = void(*)(void*, uint64_t);
        using StringIteratorParseFunction = void(*)(StringManager::Iterator&, void*);

        static inline uint32_t null_cast(char*, void*) noexcept { return 0; }

        static inline uint32_t integer_to_string(char* string, void* source) noexcept
        {
            imprint_int64_t(string, *static_cast<int64_t*>(source));

            return static_cast<uint32_t>(strlen(string));
        }

        static inline uint32_t date_time_to_string(char* string, void* source) noexcept
        {
            imprint_date_time(string, *static_cast<DateTime*>(source));

            return static_cast<uint32_t>(strlen(string));
        }

        static inline uint32_t float_to_string(char* string, void* source) noexcept
        {
            imprint_double(string, *static_cast<double*>(source));

            return static_cast<uint32_t>(strlen(string));
        }

        static inline void null_cast(StringManager::Iterator&, void*) noexcept { }

        static inline void string_to_float(StringManager::Iterator& string, void* destination)
        {
            *static_cast<double*>(destination) = parse_float(string);
        }

        static inline void string_to_integer(StringManager::Iterator& string, void* destination)
        {
            *static_cast<int64_t*>(destination) = parse_integer(string);
        }

        static inline void string_to_datetime(StringManager::Iterator& string, void* destination)
        {
            auto result = parse_datetime(string);

            if (result.is_err())
            {
                throw std::invalid_argument("Unable to parse DateTime from string: " + string.to_std_string()
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
            }

            *static_cast<DateTime*>(destination) = result.ok();
        }

        static inline void null_cast(void*) noexcept { }
        static inline void null_cast(void*, uint64_t) noexcept { }

        static inline void int_to_float(void* data) noexcept
        {
            int64_t& integer = *static_cast<int64_t*>(data);
            *reinterpret_cast<double*>(data) = static_cast<double>(integer);
        }

        static inline void int_to_float(void* data, uint64_t length) noexcept
        {
            auto ints = static_cast<int64_t*>(data);
            auto floats = static_cast<double*>(data);

            for (uint64_t i = 0; i < length; ++i)
            {
                floats[i] = static_cast<double>(ints[i]);
            }
        }

        static inline void float_to_int(void* data) noexcept
        {
            double& floating = *static_cast<double*>(data);
            *reinterpret_cast<int64_t*>(data) = static_cast<int64_t>(floating);
        }

        static inline void float_to_int(void* data, uint64_t length) noexcept
        {
            auto ints = static_cast<int64_t*>(data);
            auto floats = static_cast<double*>(data);

            for (uint64_t i = 0; i < length; ++i)
            {
                ints[i] = static_cast<int64_t>(floats[i]);
            }
        }

        static inline void int_to_datetime(void* data) noexcept
        {

        }

        static inline void int_to_datetime(void* data, uint64_t length) noexcept
        {
//            auto ints = static_cast<int64_t*>(data);
//            auto dates = static_cast<DateTime*>(data);
//
//            for (uint64_t i = 0; i < length; ++i)
//            {
//                dates[i] = DateTime(Ticks(ints[i]));
//            }
        }

        static inline void datetime_to_int(void* data) noexcept
        {

        }

        static inline void datetime_to_int(void* data, uint64_t length) noexcept
        {
//            auto ints = static_cast<int64_t*>(data);
//            auto dates = static_cast<DateTime*>(data);
//
//            for (uint64_t i = 0; i < length; ++i)
//            {
//                ints[i] = dates[i].time_since_epoch().count();
//            }
        }

        static inline void double_to_datetime(void* data) noexcept
        {
            double& floating = *static_cast<double*>(data);

            std::chrono::duration<double, std::milli> duration_double(floating);

            *reinterpret_cast<DateTime*>(data) = DateTime(std::chrono::duration_cast<Ticks>(duration_double));
        }

        static inline void double_to_datetime(void* data, uint64_t length) noexcept
        {
            auto floats = static_cast<double*>(data);
            auto dates = static_cast<DateTime*>(data);

            for (uint64_t i = 0; i < length; ++i)
            {
                std::chrono::duration<double, std::milli> duration_double(floats[i]);
                dates[i] = DateTime(std::chrono::duration_cast<Ticks>(duration_double));
            }
        }

        static inline void datetime_to_double(void* data) noexcept
        {
            DateTime& date = *static_cast<DateTime*>(data);

            *reinterpret_cast<double*>(data) = static_cast<double>(date.time_since_epoch().count());
        }

        static inline void datetime_to_double(void* data, uint64_t length) noexcept
        {
            auto floats = static_cast<double*>(data);
            auto dates = static_cast<DateTime*>(data);

            for (uint64_t i = 0; i < length; ++i)
            {
                floats[i] = static_cast<double>(dates[i].time_since_epoch().count());
            }
        }

        static_assert(static_cast<char>(DataFrame::Type::INT) == 0, "DataFrame::Type::INT must be 0");
        static_assert(static_cast<char>(DataFrame::Type::FLOAT) == 1, "DataFrame::Type::FLOAT must be 1");
        static_assert(static_cast<char>(DataFrame::Type::STRING) == 2, "DataFrame::Type::STRING must be 2");
        static_assert(static_cast<char>(DataFrame::Type::DATETIME) == 3, "DataFrame::Type::DATETIME must be 3");

        constexpr static StringCastFunction table_string_cast[4] =
        {
            integer_to_string, float_to_string, null_cast, date_time_to_string
        };

        constexpr static StringIteratorParseFunction table_string_parse[4] =
        {
            string_to_integer, string_to_float, null_cast, string_to_datetime
        };

        constexpr static TypeCastArrayFunction table_type_cast_array[4][4] =
        {
            {null_cast      , int_to_float      , null_cast, int_to_datetime},
            {float_to_int   , null_cast         , null_cast, double_to_datetime},
            {null_cast      , null_cast         , null_cast, null_cast},
            {datetime_to_int, datetime_to_double, null_cast, null_cast}
        };

        constexpr static TypeCastFunction table_type_cast[4][4] =
        {
            {null_cast      , int_to_float      , null_cast, int_to_datetime},
            {float_to_int   , null_cast         , null_cast, double_to_datetime},
            {null_cast      , null_cast         , null_cast, null_cast},
            {datetime_to_int, datetime_to_double, null_cast, null_cast}
        };
    }

    using namespace dataframe_internal;

    struct DataFrame::Value
    {
        union
        {
            int64_t value_int;
            double value_float;
            DateTime value_date_time;
            const char* value_string;
        };

        Type type;
        bool is_null = false;
        uint32_t string_length = 0;

        Value() noexcept : is_null(true), type(Type::INT), value_int(0) { }

        Value(int16_t value) noexcept : type(Type::INT), value_int(value) { }
        Value(int32_t value) noexcept : type(Type::INT), value_int(value) { }
        Value(int64_t value) noexcept : type(Type::INT), value_int(value) { }
        Value(double value)  noexcept : type(Type::FLOAT), value_float(value) { }
        Value(DateTime value) noexcept : type(Type::DATETIME), value_date_time(value) { }

        Value(uint16_t value) noexcept : type(Type::INT), value_int(static_cast<int64_t>(value)) { }
        Value(uint32_t value) noexcept : type(Type::INT), value_int(static_cast<int64_t>(value)) { }
        Value(uint64_t value) noexcept : type(Type::INT), value_int(static_cast<int64_t>(value)) { }

        Value(const char* value) noexcept
            : type(Type::STRING), value_string(value)
            , string_length(static_cast<int32_t>(strlen(value)))
        { }

        Value(Span<const char> value) noexcept
            : type(Type::STRING), value_string(value.get_pointer())
            , string_length(static_cast<int32_t>(value.length))
        { }

        inline void to_string(char* buffer)
        {
            string_length = table_string_cast[static_cast<uint32_t>(type)](buffer, &value_int);

            value_string = buffer;

            type = Type::STRING;
        }

        void cast_as(DataFrame::Type cast_type)
        {
            if (type == cast_type) return;

            if (type == Type::STRING)
            {
                auto str = get_string();

                *this = parse_value(str);

                if (type == Type::STRING)
                {
                    throw std::invalid_argument("Unable to parse: " + std::string(str.get_pointer(), str.length)
                        + " as type: " + type_to_string(cast_type)
                        + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
                }

                if (type == cast_type) return;
            }

            table_type_cast[static_cast<uint32_t>(type)][static_cast<uint32_t>(cast_type)](&value_int);
        }

        Value& operator=(int32_t value) noexcept
        {
            is_null = false;
            type = Type::INT;
            value_int = value;

            return *this;
        }

        Value& operator=(int64_t value) noexcept
        {
            is_null = false;
            type = Type::INT;
            value_int = value;

            return *this;
        }

        Value& operator=(double value) noexcept
        {
            is_null = false;
            type = Type::FLOAT;
            value_float = value;

            return *this;
        }

        Value& operator=(const DateTime& value) noexcept
        {
            is_null = false;
            type = Type::DATETIME;
            value_date_time = value;

            return *this;
        }

        Value& operator=(const char* value) noexcept
        {
            is_null = false;
            type = Type::STRING;
            value_string = value;

            return *this;
        }

        [[nodiscard]] inline int64_t get_integer() const noexcept
        {
            return value_int;
        }

        [[nodiscard]] inline double get_float() const noexcept
        {
            return value_float;
        }

        [[nodiscard]] inline DateTime get_datetime() const noexcept
        {
            return value_date_time;
        }

        [[nodiscard]] Span<const char> get_string() const noexcept
        {
            return Span<const char>(value_string, string_length);
        }

        static inline DataFrame::Value parse_datetime(const char* string, int64_t starting = -1)
        {
            auto span = Span<const char>(string, strlen(string));

            return parse_datetime(span);
        }

        static inline DataFrame::Value parse_datetime(Span<const char> string, int64_t starting = -1)
        {
            uint16_t offset = 0;
            uint16_t year = 0, month = 0, day = 0;
            uint16_t hour = 0, minute = 0, second = 0, millisecond = 0;

            if (starting < 0)
            {
                starting = 0;

                while (offset < string.length && is_number(string[offset]))
                {
                    starting = starting * 10 + (string[offset++] - '0');
                }
            }

            if (starting > 31 && starting < 10'000)
            {
                year = starting;

                if (offset + 3 >= string.length || !is_number(string[++offset]))
                {
                    return {string};
                }

                while (offset < string.length && is_number(string[offset]))
                {
                    month = month * 10 + (string[offset++] - '0');
                }

                if (offset + 1 >= string.length || !(string[offset] == '/' || string[offset] == '-'))
                {
                    return {string};
                }

                while (++offset < string.length && is_number(string[offset]))
                {
                    day = day * 10 + (string[offset] - '0');
                }
            }
            else if (starting <= 12)
            {
                month = starting;

                if (offset + 3 >= string.length || !is_number(string[++offset]))
                {
                    return {string};
                }

                while (offset < string.length && is_number(string[offset]))
                {
                    day = day * 10 + (string[offset++] - '0');
                }

                if (offset + 1 >= string.length || !(string[offset] == '/' || string[offset] == '-'))
                {
                    return {string};
                }

                while (++offset < string.length && is_number(string[offset]))
                {
                    year = year * 10 + (string[offset] - '0');
                }
            }
            else
            {
                return {string};
            }

            while (offset < string.length && (is_whitespace(string[offset]) || string[offset] == 'T'))
            {
                offset++;
            }

            while (offset < string.length && is_number(string[offset]))
            {
                hour = hour * 10 + string[offset++] - '0';
            }

            if (offset < string.length && string[offset] != ':')
            {
                return {string};
            }

            while (++offset < string.length && is_number(string[offset]))
            {
                minute = minute * 10 + string[offset] - '0';
            }

            if (offset < string.length && string[offset] != ':')
            {
                return {string};
            }

            while (++offset < string.length && is_number(string[offset]))
            {
                second = second * 10 + string[offset] - '0';
            }

            if (offset < string.length && !(string[offset] == '.' || string[offset] == ':' || string[offset] == 'Z'))
            {
                return {string};
            }

            while (++offset < string.length && is_number(string[offset]))
            {
                millisecond = millisecond * 10 + string[offset] - '0';
            }

            offset += offset < string.length && string[offset] == 'Z';

            while (offset < string.length && is_whitespace(string[offset]))
            {
                offset += 1;
            }

            if (offset < string.length)
            {
                return {string};
            }

            if (!validate_date_time(year, month, day, hour, minute, second, millisecond))
            {
                return {string};
            }

            return {construct_date_time(year, month, day, hour, minute, second, millisecond)};
        }

        static inline DataFrame::Value parse_number(Span<const char> string, bool is_negative, uint64_t offset = 0)
        {
            double adjustment = 0.1;

            auto token = DataFrame::Value(0LL);

            uint64_t max_int_offset = std::min(19ULL, string.length);

            while (offset < max_int_offset && is_number(string[offset]))
            {
                token.value_int = token.value_int * 10 + (string[offset++] - '0');
            }

            if (offset < max_int_offset && !is_negative && (string[offset] == '-' || string[offset] == '/'))
            {
                return parse_datetime(string.to_subspan_unsafe(offset), token.value_int);
            }

            if (is_negative)
            {
                token.value_int *= -1;
            }

            if (offset >= string.length)
            {
                return token;
            }

            token.type = DataFrame::Type::FLOAT;

            token.value_float = static_cast<double>(token.value_int);

            offset += string[offset] == '.';

            while (offset < string.length && is_number(string[offset]))
            {
                token.value_float += (string[offset++] - '0') * adjustment;

                adjustment *= 0.1;
            }

            if (offset < string.length && (string[offset] == 'e' || string[offset] == 'E'))
            {
                offset += 1;

                int32_t power = 0;

                bool is_negative_power = offset < string.length
                                         && !is_number(string[offset]) && string[offset++] == '-';

                while (offset < string.length && is_number(string[offset]))
                {
                    power += power * 10 + (string[offset++] - '0');
                }

                if (is_negative_power) power *= -1;

                token.value_float *= std::pow(10.0f, power);
            }

            while (offset < string.length && (is_whitespace(string[offset]) || string[offset] == '"'))
            {
                offset += 1;
            }

            return offset >= string.length ? token : DataFrame::Value(string);
        }

        static inline DataFrame::Value parse_value(const char* string)
        {
            auto span = Span<const char>(string, strlen(string));

            return parse_value(span);
        }

        static inline DataFrame::Value parse_value(Span<const char> string)
        {
            uint64_t offset = 0;

            while (offset < string.length && (is_whitespace(string[offset]) || string[offset] == '"'))
            {
                offset += 1;
            }

            if (offset >= string.length)
            {
                return {string};
            }

            if (string[offset] == '-' && offset + 1 < string.length)
            {
                return parse_number(string, true, offset + 1);
            }
            else if (is_number(string[offset]) || string[offset] == '.')
            {
                return parse_number(string, false, offset);
            }

            return {string};
        }
    };
}

/*******************************************************************************************************************
 COLUMN
 ******************************************************************************************************************/
namespace nml::dataframe_internal
{
    struct ColumnMemory
    {
        uint64_t values_bytes;
        uint64_t bitset_bytes;

        [[nodiscard]] uint64_t total_bytes() const noexcept
        {
            return values_bytes + bitset_bytes;
        }
    };

    struct Column
    {
        char* _memory;

        uint64_t _count;
        uint64_t _capacity;

        Bitset _null_index;
        ColumnMemory _memory_layout;
        StringManager* _string_manager;

        bool _is_nullable;
        DataFrame::Type _type;

        char _column_name[COLUMN_NAME_LENGTH];

        [[nodiscard]] inline uint32_t column_name_length() noexcept
        {
            return strlen(_column_name) + 1;
        }

        [[nodiscard]] static inline uint32_t type_size(DataFrame::Type type) noexcept
        {
            switch (type)
            {
                case DataFrame::Type::INT: return sizeof(int64_t);
                case DataFrame::Type::FLOAT: return sizeof(double);
                case DataFrame::Type::STRING: return sizeof(String);
                case DataFrame::Type::DATETIME: return sizeof(DateTime);
                default: return 8;
            }
        };

        [[nodiscard]] static inline ColumnMemory memory_layout(DataFrame::Type type, bool is_nullable, uint64_t length) noexcept
        {
            return ColumnMemory
            {
                .values_bytes = length * type_size(type),
                .bitset_bytes = is_nullable ? Bitset::required_bytes(length) : 0
            };
        }

        [[nodiscard]] static inline uint64_t capacity_as(DataFrame::Type type, bool is_nullable, uint64_t bytes) noexcept
        {
            if (is_nullable)
            {
                return (8 * bytes) / (1 + 8 * type_size(type));
            }
            else
            {
                return bytes / type_size(type);
            }
        }

        [[nodiscard]] MemorySpan get_null_index_memory() const noexcept
        {
            return MemorySpan(_memory + _memory_layout.values_bytes, _memory_layout.bitset_bytes);
        }

        [[nodiscard]] inline char* get_value_pointer(uint64_t row) const noexcept
        {
            return _memory + row * type_size(_type);
        }

        [[nodiscard]] inline double* get_values_float() const noexcept
        {
            return reinterpret_cast<double*>(_memory);
        }

        [[nodiscard]] inline int64_t* get_values_int() const noexcept
        {
            return reinterpret_cast<int64_t*>(_memory);
        }

        [[nodiscard]] inline DateTime* get_values_datetime() const noexcept
        {
            return reinterpret_cast<DateTime*>(_memory);
        }

        [[nodiscard]] inline String* get_values_string() const noexcept
        {
            return reinterpret_cast<String*>(_memory);
        }

        inline void initialize_string_overflow_references(uint64_t start) const noexcept
        {
            if (_type != DataFrame::Type::STRING) return;

            auto pointer = get_values_string();

            for (uint64_t i = start; i < _capacity; ++i)
            {
                pointer[i].initialize();
            }
        }

        [[nodiscard]] inline double& get_float(uint64_t offset) const noexcept
        {
            return get_values_float()[offset];
        }

        [[nodiscard]] inline int64_t& get_int(uint64_t offset) const noexcept
        {
            return get_values_int()[offset];
        }

        [[nodiscard]] inline DateTime& get_datetime(uint64_t offset) const noexcept
        {
            return get_values_datetime()[offset];
        }

        [[nodiscard]] inline String& get_string(uint64_t offset) const noexcept
        {
            return get_values_string()[offset];
        }

        inline void set(const uint64_t offset) noexcept
        {
            if (!_is_nullable) return;

            _null_index.set(offset);
        }

        inline void reset(const uint64_t offset) noexcept
        {
            if (!_is_nullable) return;

            _null_index.reset(offset);
        }

        inline void set(const uint64_t offset, const int64_t value) noexcept
        {
            if (_type != DataFrame::Type::INT) return;

            reset(offset);

            get_int(offset) = value;
        }

        inline void set(const uint64_t offset, const double value) noexcept
        {
            if (_type != DataFrame::Type::FLOAT) return;

            reset(offset);

            get_float(offset) = value;
        }

        inline void set(const uint64_t offset, const DateTime value) noexcept
        {
            if (_type != DataFrame::Type::DATETIME) return;

            reset(offset);

            get_datetime(offset) = value;
        }

        inline void set(const uint64_t offset, Span<const char> value) noexcept
        {
            if (_type != DataFrame::Type::STRING) return;

            reset(offset);

            auto& destination = get_string(offset);

            _string_manager->set_string(destination, value);
        }

        void add() noexcept
        {
            if (!_is_nullable) return;

            if (_count == _capacity) resize_capacity();

            set(_count++);
        }

        void add(const int64_t value) noexcept
        {
            if (_type != DataFrame::Type::INT) return;

            if (_count == _capacity) resize_capacity();

            set(_count++, value);
        }

        void add(const double value) noexcept
        {
            if (_type != DataFrame::Type::FLOAT) return;

            if (_count == _capacity) resize_capacity();

            set(_count++, value);
        }

        void add(const DateTime value) noexcept
        {
            if (_type != DataFrame::Type::DATETIME) return;

            if (_count == _capacity) resize_capacity();

            set(_count++, value);
        }

        void add(Span<const char> value) noexcept
        {
            if (_type != DataFrame::Type::STRING) return;

            if (_count == _capacity) resize_capacity();

            set(_count++, value);
        }

        void add(DataFrame::Value& value)
        {
            if (_type == value.type && !value.is_null)
            {
                switch (_type)
                {
                    case DataFrame::Type::INT: return add(value.get_integer());
                    case DataFrame::Type::FLOAT: return add(value.get_float());
                    case DataFrame::Type::STRING: return add(value.get_string());
                    case DataFrame::Type::DATETIME: return add(value.get_datetime());
                }
            }
            else if (value.is_null)
            {
                if (_is_nullable)
                {
                    return add();
                }

                throw std::invalid_argument("Unable to add NULL into a non-nullable column. "
                    "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
            }
            else if (_type == DataFrame::Type::STRING)
            {
                char* cast_buffer = static_cast<char*>(alloca(COLUMN_STRING_LENGTH));

                value.to_string(cast_buffer);

                value.cast_as(_type);

                switch (_type)
                {
                    case DataFrame::Type::INT: add(value.get_integer()); break;
                    case DataFrame::Type::FLOAT: add(value.get_float()); break;
                    case DataFrame::Type::STRING: add(value.get_string()); break;
                    case DataFrame::Type::DATETIME: add(value.get_datetime()); break;
                    default: throw std::invalid_argument("Unknown Type");
                }
            }
            else
            {
                value.cast_as(_type);

                switch (_type)
                {
                    case DataFrame::Type::INT: add(value.get_integer()); break;
                    case DataFrame::Type::FLOAT: add(value.get_float()); break;
                    case DataFrame::Type::STRING: add(value.get_string()); break;
                    case DataFrame::Type::DATETIME: add(value.get_datetime()); break;
                    default: throw std::invalid_argument("Unknown Type");
                }
            }
        }

        void fill(Span<int64_t> values) const
        {
            if (values.length != _count)
            {
                throw std::out_of_range("Unable to fill column of " + std::to_string(_count)
                                        + " elements with " + std::to_string(values.length) + " values.");
            }

            memcpy(_memory, values.get_pointer(0), values.bytes());
        }

        void fill(Span<double> values) const
        {
            if (values.length != _count)
            {
                throw std::out_of_range("Unable to fill column of " + std::to_string(_count)
                                        + " elements with " + std::to_string(values.length) + " values.");
            }

            memcpy(_memory, values.get_pointer(0), values.bytes());
        }

        void fill(Span<DateTime> values) const
        {
            if (values.length != _count)
            {
                throw std::out_of_range("Unable to fill column of " + std::to_string(_count)
                                        + " elements with " + std::to_string(values.length) + " values.");
            }

            memcpy(_memory, values.get_pointer(0), values.bytes());
        }

        void fill(Span<char*> values)
        {
            if (values.length != _count)
            {
                throw std::out_of_range("Unable to fill column of " + std::to_string(_count)
                                        + " elements with " + std::to_string(values.length) + " values.");
            }

            for (uint64_t i = 0; i < _count; ++i)
            {
                add(Span<const char>(values[i], strlen(values[i])));
            }
        }

        void fill(Span<std::string> values)
        {
            if (values.length != _count)
            {
                throw std::out_of_range("Unable to fill column of " + std::to_string(_count)
                                        + " elements with " + std::to_string(values.length) + " values.");
            }

            for (uint64_t i = 0; i < _count; ++i)
            {
                add(Span<const char>(values[i].c_str(), strlen(values[i].c_str())));
            }
        }

        void fill_value(int64_t value) const noexcept
        {
            int64_t* ints = get_values_int();

            for (uint64_t i = 0; i < _count; ++i)
            {
                ints[i] = value;
            }
        }

        void fill_value(double value) const noexcept
        {
            double* floats = get_values_float();

            for (uint64_t i = 0; i < _count; ++i)
            {
                floats[i] = value;
            }
        }

        void fill_value(DateTime value) const noexcept
        {
            DateTime* date_times = get_values_datetime();

            for (uint64_t i = 0; i < _count; ++i)
            {
                date_times[i] = value;
            }
        }

        void fill_value(const Span<const char> value) const noexcept
        {
            String* strings = get_values_string();

            for (uint64_t i = 0; i < _count; ++i)
            {
                _string_manager->set_string(strings[i], value);
            }
        }

        void fill_value(const char* value) const noexcept
        {
            auto span = Span<const char>(value, strlen(value) + 1);

            fill_value(span);
        }

        void fill_value(const DataFrame::Value& value) const noexcept
        {
            switch (value.type)
            {
                case DataFrame::Type::INT: fill_value(value.value_int); break;
                case DataFrame::Type::FLOAT: fill_value(value.value_float); break;
                case DataFrame::Type::STRING: fill_value(value.get_string()); break;
                case DataFrame::Type::DATETIME: fill_value(value.value_date_time); break;
            }
        }

        void resize(const uint64_t new_capacity, bool copy_nulls = true) noexcept
        {
            ColumnMemory new_layout = memory_layout(_type, _is_nullable, new_capacity);
            uint64_t new_total_bytes = new_layout.total_bytes();
            auto new_memory = static_cast<char*>(std::malloc(new_total_bytes));

            uint64_t old_offset = 0, new_offset = 0;

            memcpy(&new_memory[new_offset], &_memory[old_offset], _memory_layout.values_bytes);

            new_offset += new_layout.values_bytes;
            old_offset += _memory_layout.values_bytes;

            if (_is_nullable)
            {
                auto new_bitset_memory = MemorySpan(&new_memory[new_offset], new_layout.bitset_bytes);

                _null_index = Bitset(new_bitset_memory);

                if (copy_nulls)
                {
                    memcpy(&new_memory[new_offset], &_memory[old_offset], _memory_layout.bitset_bytes);
                }
            }

            std::free(_memory);
            _memory = new_memory;
            _memory_layout = new_layout;
            _capacity = new_capacity;

            if (_type == DataFrame::Type::STRING)
            {
                initialize_string_overflow_references(_count);
            }
        }

        inline void resize_capacity() noexcept
        {
            resize(_capacity * DEFAULT_RESIZE_FACTOR);
        }

        void make_nullable() noexcept
        {
            if (_is_nullable) return;

            auto current_bytes = _memory_layout.total_bytes();
            auto new_capacity = capacity_as(_type, true, current_bytes);

            if (new_capacity < _count)
            {
                resize(_capacity, false);
            }
            else
            {
                _is_nullable = true;
                _capacity = new_capacity;
                _memory_layout = memory_layout(_type, true, new_capacity);

                _null_index = Bitset(MemorySpan(
                    &_memory[_memory_layout.values_bytes],
                    _memory_layout.bitset_bytes
                ));
            }
        }

        void make_non_nullable()
        {
            if (!_is_nullable) return;

            if (_null_index.any())
            {
                throw std::runtime_error("Column contains nulls, unable to remove nullability."
                    "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
            }

            _is_nullable = false;
        }

        void set_column_name(Span<const char> name)
        {
            if (name.length > 0)
            {
                memcpy(_column_name, name.get_pointer(), std::min(name.length, sizeof(_column_name)));
                _column_name[std::min(name.length, sizeof(_column_name) - 1)] = 0;
            }
            else
            {
                _column_name[0] = 0;
            }
        }

        void reuse_as(DataFrame::Type type, bool is_nullable, uint64_t count) noexcept
        {
            auto capacity = capacity_as(type, is_nullable, _memory_layout.total_bytes());

            if (capacity >= count)
            {
                auto new_layout = memory_layout(type, is_nullable, capacity);

                _type = type, _is_nullable = is_nullable;
                _count = count, _memory_layout = new_layout;

                if (is_nullable)
                {
                    auto new_bitset_memory = MemorySpan(&_memory[new_layout.values_bytes], new_layout.bitset_bytes);

                    _null_index = Bitset(new_bitset_memory, true);
                }
            }
            else
            {
                auto new_layout = memory_layout(type, is_nullable, count);

                std::free(_memory);

                _type = type, _is_nullable = is_nullable;
                _count = count, _memory_layout = new_layout;
                _memory = static_cast<char*>(std::malloc(new_layout.total_bytes()));

                if (is_nullable)
                {
                    auto new_bitset_memory = MemorySpan(&_memory[new_layout.values_bytes], new_layout.bitset_bytes);

                    _null_index = Bitset(new_bitset_memory, true);
                }
            }

            if (_type == DataFrame::Type::STRING)
            {
                if (type == DataFrame::Type::STRING)
                {
                    _string_manager->reset();
                    initialize_string_overflow_references(0);
                }
                else
                {
                    delete _string_manager;
                }
            }
            else if (type == DataFrame::Type::STRING)
            {
                _string_manager = new StringManager();
                initialize_string_overflow_references(0);
            }
        }

        void reinterpret_type(DataFrame::Type type, bool is_nullable)
        {
            if (_type == type)
            {
                return is_nullable ? make_nullable() : make_non_nullable();
            }

            if (type_size(_type) == type_size(type))
            {
                is_nullable ? make_nullable() : make_non_nullable();

                table_type_cast_array[static_cast<int>(_type)][static_cast<int>(type)](_memory, _count);

                _type = type;

                return;
            }

            if (type == DataFrame::Type::STRING)
            {
                ColumnMemory new_layout = memory_layout(type, is_nullable, _capacity);
                uint64_t new_total_bytes = new_layout.total_bytes();
                auto new_memory = static_cast<char*>(std::malloc(new_total_bytes));

                if (is_nullable)
                {
                    auto new_bitset_memory = MemorySpan(
                        &new_memory[new_layout.values_bytes], new_layout.bitset_bytes);

                    _null_index = Bitset(new_bitset_memory);

                    if (_is_nullable)
                    {
                        memcpy(
                           &new_memory[new_layout.values_bytes],
                           &_memory[_memory_layout.values_bytes],
                           _memory_layout.bitset_bytes
                       );
                    }
                }

                _string_manager = new StringManager(_capacity);
                auto strings = reinterpret_cast<String*>(new_memory);

                auto stride = type_size(_type);
                StringCastFunction cast = table_string_cast[static_cast<int>(type)];

                for (uint32_t i = 0; i < _capacity; ++i)
                {
                    strings[i].initialize();

                    if (i >= _count) continue;

                    if (_is_nullable && _null_index.check(i)) continue;

                    strings[i].length = cast(strings[i].string, &_memory[stride * i]);
                }

                _type = type;
                std::free(_memory);
                _memory = new_memory, _memory_layout = new_layout;

                return;
            }

            if (_type == DataFrame::Type::STRING)
            {
                ColumnMemory new_layout = memory_layout(type, is_nullable, _capacity);
                uint64_t new_total_bytes = new_layout.total_bytes();
                auto new_memory = static_cast<char*>(std::malloc(new_total_bytes));

                if (is_nullable)
                {
                    auto new_bitset_memory = MemorySpan(
                        &new_memory[new_layout.values_bytes], new_layout.bitset_bytes);

                    _null_index = Bitset(new_bitset_memory);

                    if (_is_nullable)
                    {
                        memcpy(
                            &new_memory[new_layout.values_bytes],
                            &_memory[_memory_layout.values_bytes],
                            _memory_layout.bitset_bytes
                        );
                    }
                }

                auto stride = type_size(type);

                StringIteratorParseFunction string_parser = table_string_parse[static_cast<int>(type)];

                for (uint64_t i = 0; i < _count; ++i)
                {
                    if (_is_nullable && _null_index.check(i)) continue;

                    auto& string = get_string(i);

                    auto iterator = _string_manager->get_iterator(string);

                    string_parser(iterator, &new_memory[stride * i]);
                }

                _type = type;
                std::free(_memory), delete _string_manager;
                _memory = new_memory, _memory_layout = new_layout;

                return;
            }

            throw std::runtime_error("Unable to complete type conversion."
                "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
        }

        explicit Column(DataFrame::Type type, bool is_nullable, uint64_t count = 0)
            : _type(type), _is_nullable(is_nullable), _count(count)//, _column_name({})
            , _capacity(std::max(count, STARTING_COLUMN_LENGTH)), _string_manager(nullptr)
            , _memory_layout(memory_layout(type, is_nullable, std::max(count, STARTING_COLUMN_LENGTH)))
        {
            auto total_bytes = _memory_layout.total_bytes();

            _memory = static_cast<char*>(std::malloc(total_bytes));

            if (_memory_layout.bitset_bytes > 0)
            {
                auto memory_span = MemorySpan(
                    _memory + _memory_layout.values_bytes,
                    _memory_layout.bitset_bytes
                );

                _null_index = Bitset(memory_span);
            }

            if (_type == DataFrame::Type::STRING)
            {
                _string_manager = new StringManager();
                initialize_string_overflow_references(0);
            }
        }

        ~Column()
        {
            std::free(_memory);

            if (_type == DataFrame::Type::STRING)
            {
                delete _string_manager;
            }
        }

        Column(Column&& other) noexcept
            : _memory(other._memory), _count(other._count), _capacity(other._capacity)
            , _null_index(other._null_index), _memory_layout(other._memory_layout), _type(other._type)
            , _is_nullable(other._is_nullable), _string_manager(other._string_manager)//, _column_name({})
        {
            other._count = 0;
            other._capacity = 0;
            other._memory = nullptr;
            other._type = DataFrame::Type::INT;

            memcpy(_column_name, other._column_name, sizeof(_column_name));
        }

        Column& operator=(Column&& other) noexcept
        {
            if (this != &other)
            {
                this->~Column();
                new(this) Column(std::move(other));
            }

            return *this;
        }

        Column(Column& other) = delete;
        Column& operator=(const Column& copy) = delete;
    };

    inline uint32_t integer_character_length(int64_t integer)
    {
        if (integer == 0) return 1;

        return static_cast<uint32_t>(std::log10(std::abs(integer))) + (integer < 0 ? 2 : 1);
    }

    static inline DateTime add(DateTime& date, int64_t value)
    {
        return date + std::chrono::milliseconds(value);
    }

    static inline DateTime add(DateTime& date, double value)
    {
        auto duration = std::chrono::milliseconds(static_cast<int>(value));

        return date + duration;
    }

    static inline DateTime add(DateTime& date, DateTime value)
    {
        return date + value.time_since_epoch();
    }
}

/*******************************************************************************************************************
 INTERNAL VALUE
 ******************************************************************************************************************/
namespace nml
{
    class DataFrame::ReferenceValue
    {
        uint64_t _row;
        uint32_t _column;
        DataFrame& _parent;

    public:

        explicit ReferenceValue(DataFrame& parent, uint32_t column, uint64_t row) noexcept
            : _parent(parent), _column(column), _row(row)
        { }

        [[nodiscard]] bool is_null() const noexcept
        {
            auto& column = _parent._columns[_column];

            return column._is_nullable && column._null_index.check(_row);
        }

        [[nodiscard]] DataFrame::Type get_type() const noexcept
        {
            auto& column = _parent._columns[_column];

            return column._type;
        }

        [[nodiscard]] int64_t& as_int() const noexcept
        {
            auto& column = _parent._columns[_column];

            return reinterpret_cast<int64_t*>(column._memory)[_row];
        }

        [[nodiscard]] double& as_float() const noexcept
        {
            auto& column = _parent._columns[_column];

            return reinterpret_cast<double*>(column._memory)[_row];
        }

        [[nodiscard]] DateTime& as_datetime() const noexcept
        {
            auto& column = _parent._columns[_column];

            return reinterpret_cast<DateTime*>(column._memory)[_row];
        }

        [[nodiscard]] String& as_string() const noexcept
        {
            auto& column = _parent._columns[_column];

            return reinterpret_cast<String*>(column._memory)[_row];
        }

        //        Value& operator+(int other)
        //        {
        //            // TODO
        //        }

        ReferenceValue& operator=(Value& value)
        {
            if (value.is_null)
            {
                auto& column = _parent._columns[_column];

                if (!column._is_nullable)
                {
                    // TODO throw
                }

                column._null_index.set(_row);

                return *this;
            }

            switch (value.type)
            {
                case Type::INT: *this = value.value_int; break;
                case Type::FLOAT: *this = value.value_float; break;
                case Type::STRING: *this = value.get_string(); break;
                case Type::DATETIME: *this = value.value_date_time; break;
            }

            return *this;
        }

        ReferenceValue& operator=(DateTime other)
        {
            auto& column = _parent._columns[_column];

            column._null_index.reset(_row);

            switch (column._type)
            {
                case DataFrame::Type::INT:
                case DataFrame::Type::DATETIME:
                {
                    as_datetime() = other;
                    break;
                }
                case DataFrame::Type::FLOAT:
                {
                    as_float() = static_cast<double>(other.time_since_epoch().count());
                    break;
                }
                case DataFrame::Type::STRING:
                {
                    char buffer[25];
                    String& string = as_string();
                    imprint_date_time(buffer, other);
                    column._string_manager->set_string(string, buffer);
                    break;
                }
            }

            return *this;
        }

        ReferenceValue& operator=(int64_t other)
        {
            auto& column = _parent._columns[_column];

            column._null_index.reset(_row);

            switch (column._type)
            {
                case DataFrame::Type::INT:
                {
                    as_int() = other;
                    break;
                }
                case DataFrame::Type::FLOAT:
                {
                    as_float() = static_cast<double>(other);
                    break;
                }
                case DataFrame::Type::DATETIME:
                {
                    DateTime& date_time = as_datetime();
                    // TODO
                    break;
                }
                case DataFrame::Type::STRING:
                {
                    char buffer[22];
                    String& string = as_string();
                    imprint_int64_t(buffer, other);
                    column._string_manager->set_string(string, buffer);
                    break;
                }
            }

            return *this;
        }

        ReferenceValue& operator=(double other)
        {
            auto& column = _parent._columns[_column];

            column._null_index.reset(_row);

            switch (column._type)
            {
                case DataFrame::Type::INT:
                case DataFrame::Type::DATETIME:
                {
                    as_int() = static_cast<int64_t>(other);
                    break;
                }
                case DataFrame::Type::FLOAT:
                {
                    as_float() = other;
                    break;
                }
                case DataFrame::Type::STRING:
                {
                    char buffer[20];
                    String& string = as_string();
                    imprint_double(buffer, other);
                    column._string_manager->set_string(string, buffer);
                    break;
                }
            }

            return *this;
        }

        ReferenceValue& operator=(Span<const char> right)
        {
            auto& column = _parent._columns[_column];

            column._null_index.reset(_row);

            if (column._type == DataFrame::Type::STRING)
            {
                String& string = as_string();
                column._string_manager->set_string(string, right);
                return *this;
            }

            auto parsed_value = Value::parse_value(right);

            if (parsed_value.type == Type::INT)
            {
                *this = parsed_value.value_int;
            }
            else if (parsed_value.type == Type::DATETIME)
            {
//                *this = parsed_value.value_date_time;

                // TODO
            }
            else if (parsed_value.type == Type::FLOAT)
            {
                *this = parsed_value.value_float;
            }
            else
            {
                throw std::invalid_argument("Unable to parse: " + std::string(right.get_pointer(), right.length)
                    + " as type: " + type_to_string(column._type)
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
            }

            return *this;
        }

        ReferenceValue& operator+=(int64_t other)
        {
            auto& column = _parent._columns[_column];

            switch (column._type)
            {
                case DataFrame::Type::INT:
                {
                    as_int() += other;
                    break;
                }
                case DataFrame::Type::FLOAT:
                {
                    as_float() += static_cast<double>(other);
                    break;
                }
                case DataFrame::Type::DATETIME:
                {
                    DateTime& date_time = as_datetime();
                    date_time = add(date_time, other);
                    break;
                }
                case DataFrame::Type::STRING:
                {
                    char buffer[21];
                    String& string = as_string();
                    imprint_int64_t(buffer, other);
                    column._string_manager->string_append(string, buffer);
                    break;
                }
            }

            return *this;
        }

        void print(uint32_t print_width = 0) const
        {
            DataFrame::Type type = get_type();

            uint32_t buffer_size;

            switch (type)
            {
                case DataFrame::Type::INT: buffer_size = std::max(21u, print_width + 1); break;
                case DataFrame::Type::FLOAT: buffer_size = std::max(12u, print_width + 1); break;
                case DataFrame::Type::DATETIME: buffer_size = std::max(21u, print_width + 1); break;
                case DataFrame::Type::STRING:
                {
                    auto str = as_string();
                    buffer_size = print_width == 0 ? str.length + 1 : print_width + 1;
                    break;
                }
            }

            if (is_null())
            {
                std::cout << "NULL";

                for (uint32_t i = 4; i < print_width; ++i)
                {
                    std::cout << " ";
                }

                return;
            }

            char* print_buffer = reinterpret_cast<char*>(alloca(buffer_size));

            memset(print_buffer, ' ', buffer_size);

            switch (type)
            {
                case DataFrame::Type::INT:
                {
                    imprint_int64_t(print_buffer, as_int(), false);

                    print_buffer[print_width] = 0;

                    break;
                }
                case DataFrame::Type::FLOAT:
                {
                    imprint_float(print_buffer, as_float(), false); break;
                }
                case DataFrame::Type::DATETIME:
                {
                    imprint_date_time(print_buffer, as_datetime());
                    break;
                }
                case DataFrame::Type::STRING:
                {
                    auto str = as_string();

                    auto copy_length = std::min(str.length, buffer_size - 1);

                    memcpy(print_buffer, str.string, copy_length);

                    if (copy_length < str.length)
                    {
                        print_buffer[buffer_size - 2] = '.';
                        print_buffer[buffer_size - 3] = '.';
                        print_buffer[buffer_size - 4] = '.';
                    }

                    break;
                }
                default: throw std::invalid_argument("Unknown Type");
            }

            print_buffer[buffer_size - 1] = 0;
            std::cout << print_buffer;
        }
    };
}

/*******************************************************************************************************************
 ROW VIEW
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::dataframe_internal;

    class DataFrame::RowView
    {
        friend class DataFrame;

        uint64_t _row;
        DataFrame& _parent;

        inline ReferenceValue get_value(uint32_t column) noexcept
        {
            return ReferenceValue(_parent, _parent._column_shifts[column], _row);
        }

        explicit RowView(DataFrame& parent, uint64_t row)
            : _row(row), _parent(parent)
        { }

    public:

        ReferenceValue operator[](uint32_t column)
        {
            if (column >= _parent._column_shifts.count)
            {
                throw std::out_of_range("Column regex_offset out of bounds: " + std::to_string(column)
                    + "\n(File: " + std::string(__FILE__) + ", Line: " + std::to_string(__LINE__) + ")");
            }

            return get_value(column);
        }

        void print(Span<uint32_t> column_widths)
        {
            for (uint32_t column = 0; column < _parent._column_shifts.count; ++column)
            {
                std::cout << "| ";
                get_value(column).print(column_widths[column] - 2);
                if (column == _parent._column_shifts.count - 1) std::cout << " |";
                else std::cout << " ";
            }

            std::cout << "\n";
        }
    };
}

/*******************************************************************************************************************
 COLUMN VIEW
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::dataframe_internal;

    class DataFrame::ColumnView
    {
        friend class DataFrame;

        uint32_t _column;
        DataFrame& _parent;

        Column& get_column() noexcept { return _parent._columns[_column]; }

        explicit ColumnView(DataFrame& parent, uint32_t column) noexcept
            : _parent(parent), _column(column)
        { }

    public:



        [[nodiscard]] inline char* get_column_name() noexcept;
        inline void set_column_name(const char* name) noexcept;
        inline void set_column_name(Span<const char> name) noexcept;

        [[nodiscard]] inline Type get_type() noexcept;
        [[nodiscard]] inline bool is_nullable() noexcept;
        [[nodiscard]] inline uint64_t get_length() noexcept;

        [[nodiscard]] inline Span<double> get_floats() noexcept;
        [[nodiscard]] inline Span<int64_t> get_integers() noexcept;
        [[nodiscard]] inline Span<DateTime> get_date_times() noexcept;

        [[nodiscard]] inline ReferenceValue operator[](uint32_t row);
        [[nodiscard]] const inline Bitset& get_null_index() noexcept;

        ColumnView& operator=(double value);
        ColumnView& operator=(int64_t value);
        ColumnView& operator=(int32_t value);
        ColumnView& operator=(DateTime value);
        ColumnView& operator=(const char* value);
        ColumnView& operator=(const Value& value);
        ColumnView& operator=(Span<const char> value);

        // TODO finish operators
        ColumnView& operator+=(double other);
    };

    void DataFrame::ColumnView::set_column_name(Span<const char> name) noexcept
    {
        _parent.throw_if_column_name_taken(name);

        get_column().set_column_name(name);
    }

    void DataFrame::ColumnView::set_column_name(const char* name) noexcept
    {
        auto span = Span<const char>(name, strlen(name));

        get_column().set_column_name(span);
    }

    char* DataFrame::ColumnView::get_column_name() noexcept
    {
        return get_column()._column_name;
    }

    uint64_t DataFrame::ColumnView::get_length() noexcept
    {
        return get_column()._count;
    }

    DataFrame::Type DataFrame::ColumnView::get_type() noexcept
    {
        return get_column()._type;
    }

    bool DataFrame::ColumnView::is_nullable() noexcept
    {
        return get_column()._is_nullable;
    }

    const inline Bitset& DataFrame::ColumnView::get_null_index() noexcept
    {
        return get_column()._null_index;
    }

    DataFrame::ReferenceValue DataFrame::ColumnView::operator[](uint32_t row)
    {
        if (row >= get_column()._count)
        {
            throw std::out_of_range("Row out of bounds: " + std::to_string(row)
                + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
        }

        return ReferenceValue(_parent, _column, row);
    }

    Span<int64_t> DataFrame::ColumnView::get_integers() noexcept
    {
        auto& column = get_column();

        return Span<int64_t>(column.get_values_int(), column._count);
    }

    Span<double> DataFrame::ColumnView::get_floats() noexcept
    {
        auto& column = get_column();

        return Span<double >(column.get_values_float(), column._count);
    }

    Span<DateTime> DataFrame::ColumnView::get_date_times() noexcept
    {
        auto& column = get_column();

        return Span<DateTime>(column.get_values_datetime(), column._count);
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(DateTime value)
    {
        *this = DataFrame::Value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(double value)
    {
        *this = DataFrame::Value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(int32_t value)
    {
        *this = DataFrame::Value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(int64_t value)
    {
        *this = DataFrame::Value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(const DataFrame::Value& value)
    {
        auto& column = get_column();

        if (column._type != value.type)
        {
            column.reuse_as(value.type, false, _parent._row_count);
        }

        if (column._is_nullable)
        {
            column._null_index.reset_all();
        }

        column.fill_value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(Span<const char> value)
    {
        *this = DataFrame::Value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator=(const char* value)
    {
        *this = DataFrame::Value(value);

        return *this;
    }

    DataFrame::ColumnView& DataFrame::ColumnView::operator+=(double other)
    {
        auto& column = get_column();

        switch (column._type)
        {
            case DataFrame::Type::INT:
            {
                auto ints = column.get_values_int();

                for (uint32_t i = 0; i < column._count; ++i)
                {
                    ints[i] += static_cast<int>(other);
                }

                break;
            }
            case DataFrame::Type::FLOAT:
            {
                auto floats = column.get_values_float();

                for (uint32_t i = 0; i < column._count; ++i)
                {
                    floats[i] += other;
                }

                break;
            }
            case DataFrame::Type::DATETIME:
            {
                auto date_times = column.get_values_datetime();

                for (uint32_t i = 0; i < column._count; ++i)
                {
                    date_times[i] = add(date_times[i], other);
                }

                break;
            }
            case DataFrame::Type::STRING:
            {
                char buffer[22];
                imprint_double(buffer, other);

                auto strings = column.get_values_string();

                for (uint32_t i = 0; i < column._count; ++i)
                {
                    String& string = strings[i];
                    column._string_manager->string_append(string, buffer);
                }

                break;
            }
        }

        return *this;
    }
}

/*******************************************************************************************************************
 ADD ROW
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::dataframe_internal;

    uint64_t DataFrame::add_row(std::initializer_list<Value> row)
    {
        auto start = const_cast<Value*>(row.begin());

        Span<Value> span = Span<Value>(start, row.size());

        return add_row(span);
    }

    uint64_t DataFrame::add_row(Span<Value> row)
    {
        if (row.length > _column_shifts.count)
        {
            throw std::out_of_range("Unable to add row of " + std::to_string(row.length)
                + " elements to DataFrame with " + std::to_string(_column_shifts.count) + " column_ct."
                + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
        }

        for (uint32_t column_offset = 0; column_offset < _column_shifts.count; ++column_offset)
        {
            auto& row_value = row[column_offset];
            auto& column = _columns[_column_shifts[column_offset]];

            column.add(row_value);
        }

        return _row_count++;
    }
}

/*******************************************************************************************************************
 REMOVE COLUMN
 ******************************************************************************************************************/
namespace nml
{
    void DataFrame::remove_column(uint32_t offset) noexcept
    {
        uint32_t temp = MAX_U32;

        for (uint32_t i = _column_shifts.count; i > offset; --i)
        {
            std::swap(temp, _column_shifts[i - 1]);
        }

        _column_shifts.pop();
    }

    void DataFrame::remove_column(const char* column_name) noexcept
    {
        int32_t offset = find_column_offset(column_name);

        if (offset > 0)
        {
            remove_column(static_cast<uint32_t>(offset));
        }
    }

    void DataFrame::remove_column(Span<const char> column_name) noexcept
    {
        int32_t offset = find_column_offset(column_name);

        if (offset > 0)
        {
            remove_column(static_cast<uint32_t>(offset));
        }
    }
}

/*******************************************************************************************************************
 ADD COLUMN
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::dataframe_internal;

    uint32_t DataFrame::add_column(DataFrame::Type type, bool nullable, uint64_t length, Span<const char> name, uint32_t offset)
    {// TODO possibly reuse memory from soft deletes
        if (_row_count == 0 && column_count() == 0)
        {
            _row_count = length;
        }

        if (length != _row_count)
        {
            throw std::out_of_range("Unable to add a column of " + std::to_string(length)
                + " elements to a DataFrame of " + std::to_string(_row_count) + " elements.");
        }

        throw_if_column_name_taken(name);

        offset = std::min(offset, (uint32_t)_column_shifts.count);

        uint32_t next_column_offset = find_next_column_offset();

        _column_shifts.add(next_column_offset);

        uint32_t temp = next_column_offset;

        for (uint32_t i = offset; i < _column_shifts.count; ++i)
        {
            std::swap(temp, _column_shifts[i]);
        }

        if (next_column_offset < _columns.count)
        {
            auto& column = _columns[next_column_offset];

            column.set_column_name(name);

            column.reuse_as(type, nullable, _row_count);
        }
        else
        {
            auto column = Column(type, nullable, _row_count);

            column.set_column_name(name);

            if (_columns.capacity == _columns.count)
            {
                _columns.resize();
                initialize_allocated_columns();
            }

            _columns[_columns.count++] = std::move(column);
        }

        return offset;
    }

    DataFrame::ColumnView DataFrame::add_column(DataFrame::Type type, uint32_t offset)
    {
        return add_column(type, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(const Value& value, uint32_t offset)
    {
        return add_column(value, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<int64_t> values, uint32_t offset)
    {
        return add_column(values, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<double> values, uint32_t offset)
    {
        return add_column(values, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<DateTime> dates, uint32_t offset)
    {
        return add_column(dates, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<char*> c_strings, uint32_t offset)
    {
        return add_column(c_strings, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<std::string> strings, uint32_t offset)
    {
        return add_column(strings, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(ColumnView column_to_copy, uint32_t offset)
    {
        return add_column(column_to_copy, "", offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Type type, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(type, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(const Value& value, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(value, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<int64_t> values, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(values, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<double> values, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(values, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<DateTime> dates, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(dates, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<char*> c_strings, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(c_strings, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<std::string> strings, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(strings, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(ColumnView column_to_copy, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));
        return add_column(column_to_copy, span, offset);
    }

    DataFrame::ColumnView DataFrame::add_column(Type type, Span<const char> column_name, uint32_t offset)
    {
        auto column_index = add_column(type, true, _row_count, column_name, offset);

        auto& column = get_raw_column_unsafe(column_index);

        column._null_index.set_all();

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<int64_t> values, Span<const char> column_name, uint32_t offset)
    {
        uint32_t column_index = add_column(Type::INT, false, values.length, column_name, offset);

        auto& column = get_raw_column_unsafe(column_index);

        column.fill(values);

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<double> values, Span<const char> column_name, uint32_t offset)
    {
        auto column_index = add_column(Type::FLOAT, false, values.length, column_name, offset);

        auto& column = get_raw_column_unsafe(column_index);

        column.fill(values);

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<DateTime> dates, Span<const char> column_name, uint32_t offset)
    {
        uint32_t column_index = add_column(Type::DATETIME, false, dates.length, column_name, offset);

        auto& column = get_raw_column_unsafe(column_index);

        column.fill(dates);

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<char*> c_strings, Span<const char> column_name, uint32_t offset)
    {
        uint32_t column_index = add_column(Type::STRING, false, c_strings.length, column_name, offset);

        auto& column = get_raw_column_unsafe(column_index);

        column.fill(c_strings);

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(Span<std::string> strings, Span<const char> column_name, uint32_t offset)
    {
        uint32_t column_index = add_column(Type::STRING, false, strings.length, column_name, offset);

        auto& column = get_raw_column_unsafe(column_index);

        column.fill(strings);

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(ColumnView column_to_copy, Span<const char> column_name, uint32_t offset)
    {
        auto column_index = add_column(
            column_to_copy.get_type(),
            column_to_copy.is_nullable(),
            column_to_copy.get_length(), column_name, offset
        );

        auto& old_column = column_to_copy.get_column();
        auto& new_column = get_raw_column_unsafe(column_index);

        if (column_to_copy.is_nullable())
        {
            MemorySpan old_null_index_memory = old_column.get_null_index_memory();
            MemorySpan new_null_index_memory = new_column.get_null_index_memory();

            memcpy(
                new_null_index_memory.get_pointer(),
                old_null_index_memory.get_pointer(),
                new_null_index_memory.bytes
            );
        }

        switch (column_to_copy.get_type())
        {
            case Type::INT:
            {
                new_column.fill(column_to_copy.get_integers()); break;
            }
            case Type::FLOAT:
            {
                new_column.fill(column_to_copy.get_floats()); break;
            }
            case Type::DATETIME:
            {
                new_column.fill(column_to_copy.get_date_times()); break;
            }
            case Type::STRING:
            {
                auto strings = Span<String>(old_column.get_values_string(), new_column._count);

                uint32_t buffer_length = COLUMN_STRING_LENGTH * 5;

                char* buffer = reinterpret_cast<char*>(std::malloc(buffer_length));

                for (uint32_t i = 0; i < new_column._count; ++i)
                {
                    auto string_length = strings[i].length + 2;

                    if (string_length > buffer_length)
                    {
                        std::free(buffer);
                        buffer_length = string_length;
                        buffer = reinterpret_cast<char*>(std::malloc(buffer_length));
                    }

                    old_column._string_manager->string_copy(strings[i], buffer);

                    new_column.add(Span<const char>(buffer, string_length));
                }

                std::free(buffer);

                break;
            }
            default: throw std::invalid_argument("Unknown Type");
        }

        return get_column_unsafe(column_index);
    }

    DataFrame::ColumnView DataFrame::add_column(const Value& value, Span<const char> column_name, uint32_t offset)
    {
        uint32_t column_index = add_column(value.type, false, _row_count, column_name, offset);

        auto& new_column = get_raw_column_unsafe(column_index);

        switch (value.type)
        {
            case Type::INT:
            {
                auto integer = value.get_integer();
                auto integers = new_column.get_values_int();

                for (uint32_t i = 0; i < _row_count; ++i)
                {
                    integers[i] = integer;
                }

                break;
            }
            case Type::FLOAT:
            {
                auto float_value = value.get_float();
                auto floats = new_column.get_values_float();

                for (uint32_t i = 0; i < _row_count; ++i)
                {
                    floats[i] = float_value;
                }

                break;
            }
            case Type::STRING:
            {
                auto string = value.get_string();
                auto strings = new_column.get_values_string();

                for (uint32_t i = 0; i < _row_count; ++i)
                {
                    new_column._string_manager->set_string(strings[i], string);
                }

                break;
            }
            case Type::DATETIME:
            {
                auto date_time = value.get_datetime();
                auto date_times = new_column.get_values_datetime();

                for (uint32_t i = 0; i < _row_count; ++i)
                {
                    date_times[i] = date_time;
                }

                break;
            }
        }

        return get_column_unsafe(column_index);
    }
}

/*******************************************************************************************************************
 FACTORIZE
 ******************************************************************************************************************/
namespace nml
{
    DataFrame::ColumnView DataFrame::factorize_column(ColumnView column, uint32_t offset)
    {
        return factorize_column(column, "", offset);
    }

    DataFrame::ColumnView DataFrame::factorize_column_with_distinct(ColumnView column, ResizableList<int64_t>* distinct, uint32_t offset)
    {
        return factorize_column_with_distinct(column, "", distinct, offset);
    }

    DataFrame::ColumnView DataFrame::factorize_column(ColumnView column, const char* column_name, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));

        return factorize_column_with_distinct(column, span, nullptr, offset);
    }

    DataFrame::ColumnView DataFrame::factorize_column_with_distinct(ColumnView column, const char* column_name, ResizableList<int64_t>* distinct, uint32_t offset)
    {
        auto span = Span<const char>(column_name, strlen(column_name));

        return factorize_column_with_distinct(column, span, distinct, offset);
    }

    DataFrame::ColumnView DataFrame::factorize_column(ColumnView column, Span<const char> column_name, uint32_t offset)
    {
        return factorize_column_with_distinct(column, column_name, nullptr, offset);
    }

    DataFrame::ColumnView DataFrame::factorize_column_with_distinct(ColumnView column, Span<const char> column_name, ResizableList<int64_t>* distinct, uint32_t offset)
    {
        auto& column_root = column.get_column();

        if (column_root._type != Type::STRING)
        {
            throw std::invalid_argument("Column factorization requires type STRING."
                "\n(File: " + std::string(__FILE__) + ", Line: " + std::to_string(__LINE__) + ")");
        }

        uint32_t factorized_column_offset = add_column(
            Type::INT,
            column_root._is_nullable,
            column_root._count,
            column_name,
            offset
        );

        auto& factorized_column = get_raw_column_unsafe(factorized_column_offset);

        auto _distinct = HashMap<StringManager::Container, uint64_t>();

        String* strings = column_root.get_values_string();

        for (uint64_t i = 0; i < column_root._count; ++i)
        {
            if (column_root._is_nullable && column_root._null_index.check(i)) continue;

            auto container = column_root._string_manager->container(strings[i]);

            uint64_t starting_ct = _distinct.count();

            uint64_t value = *_distinct.insert(container, starting_ct);

            factorized_column.set(i, static_cast<int64_t>(value));

            if (distinct != nullptr && value == starting_ct)
            {
                distinct->add(static_cast<int64_t>(value));
            }
        }

        return ColumnView(*this, factorized_column_offset);
    }
}

/*******************************************************************************************************************
 PRINT
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::dataframe_internal;

    void DataFrame::print(uint64_t rows)
    {// TODO this is nasty

        std::cout << "row ct: " << row_count() << ", ";
        std::cout << "column ct: " << column_count() << "\n";

        const char* default_column_name = "Column: ";

        rows = rows == 0 ? _row_count : std::min(rows, _row_count);
        uint32_t row_col_width = integer_character_length(static_cast<int>(rows)) + 2;

        uint32_t schema_memory_offset = 0;
        uint32_t schema_memory_bytes = _column_shifts.count * (2 * sizeof(uint32_t) + sizeof(bool));
        auto schema_memory = MemorySpan(reinterpret_cast<char*>(alloca(schema_memory_bytes)), schema_memory_bytes);

        auto column_widths = Span<uint32_t>(schema_memory, _column_shifts.count);
        schema_memory_offset += column_widths.bytes();
        auto default_columns = Span<bool>(schema_memory.offset(schema_memory_offset), _column_shifts.count);
        schema_memory_offset += default_columns.bytes();
        auto column_name_widths = Span<uint32_t>(schema_memory.offset(schema_memory_offset), _column_shifts.count);

        for (uint32_t col = 0; col < column_widths.length; ++col)
        {
            auto column = get_column_unsafe(col);

            char* column_name = column.get_column_name();

            column_name_widths[col] = strlen(column_name);

            default_columns[col] = column_name_widths[col] == 0;

            if (default_columns[col])
            {
                column_name_widths[col] = strlen(default_column_name)
                    + integer_character_length(static_cast<int>(col));
            }

            column_widths[col] = column_name_widths[col] + 2;

            Type type = column.get_type();

            switch (type)
            {
                case Type::FLOAT: column_widths[col] = std::max(13u, column_widths[col]); break;
                case Type::DATETIME: column_widths[col] = std::max(22u, column_widths[col]); break;
                case Type::INT:
                {
                    for (uint32_t row = 0; row < rows; ++row)
                    {
                        uint32_t str_len = integer_character_length(column[row].as_int()) + 2;

                        column_widths[col] = std::max(column_widths[col], str_len);
                    }

                    break;
                }
                case Type::STRING:
                {
                    for (uint32_t row = 0; row < rows; ++row)
                    {
                        auto string = column[row].as_string();

                        column_widths[col] = std::max(column_widths[col], string.length + 2);
                    }

                    column_widths[col] = std::min(COLUMN_STRING_LENGTH + 2, std::max(column_widths[col], 10u));

                    break;
                }
            }
        }

        std::cout << "+";

        for (uint32_t j = 0; j < row_col_width; ++j) std::cout << "-";

        for (uint32_t col = 0; col < column_widths.length; ++col)
        {
            if (col == 0) std::cout << "+";

            for (uint32_t i = 0; i < column_widths[col]; ++i) std::cout << '-';

            std::cout << "+";
        }

        std::cout << "\n";

        std::cout << "|";

        for (uint32_t j = 0; j < row_col_width; ++j) std::cout << " ";

        for (uint32_t col = 0; col < column_widths.length; ++col)
        {
            std::cout << "| ";

            if (default_columns[col])
            {
                std::cout << default_column_name << col;
            }
            else
            {
                std::cout << get_column_unsafe(col).get_column_name();
            }

            uint32_t required_padding = column_widths[col] - column_name_widths[col];

            for (uint32_t i = 1; i < required_padding; ++i) std::cout << ' ';

            if (col == _column_shifts.count - 1) std::cout << "|";
        }

        std::cout << "\n";

        std::cout << "+";

        for (uint32_t j = 0; j < row_col_width; ++j) std::cout << "-";

        for (uint32_t col = 0; col < column_widths.length; ++col)
        {
            std::cout << "+";

            for (uint32_t i = 0; i < column_widths[col]; ++i) std::cout << '-';

            if (col == _column_shifts.count - 1) std::cout << "+";
        }

        std::cout << "\n";

        for (int i = 0; i < rows; ++i)
        {
            std::cout << "| " << i;

            uint32_t row_width = integer_character_length(i);

            for (uint32_t j = 0; j < row_col_width - row_width - 1; ++j)
            {
                std::cout << " ";
            }

            get_row_unsafe(i).print(column_widths);
        }

        std::cout << "+";

        for (uint32_t j = 0; j < row_col_width; ++j) std::cout << "-";

        for (uint32_t col = 0; col < column_widths.length; ++col)
        {
            std::cout << "+";

            for (uint32_t i = 0; i < column_widths[col]; ++i) std::cout << '-';

            if (col == _column_shifts.count - 1) std::cout << "+";
        }

        std::cout << "\n";
    }
}

/*******************************************************************************************************************
 DATA READER
 ******************************************************************************************************************/
namespace nml
{
    using namespace nml::dataframe_internal;

    DataFrame DataFrame::from_delimited(const std::string& path, bool has_headers, char delimiter)
    {
        return from_delimited(path.c_str(), has_headers, delimiter);
    }

    DataFrame DataFrame::from_delimited(const char* path, bool has_headers, char delimiter)
    {
        auto df = DataFrame();

        uint64_t file_line = 0;

        auto file_iterator = DataReader::delimited_iterator(path, delimiter);

        if (has_headers)
        {
            while (file_iterator.has_next())
            {
                auto next = file_iterator.next();

                if (next.type != DataReader::DelimitedTokenType::VALUE)
                {
                    file_line += 1; break;
                }

                auto header_span = Span<const char>(next.start, next.length);

                df.add_column(Type::INT, trim_string_span(header_span));
            }
        }
        else
        {
            df._row_count = 1;

            while (file_iterator.has_next())
            {
                auto next = file_iterator.next();

                if (next.type != DataReader::DelimitedTokenType::VALUE)
                {
                    file_line += 1; break;
                }

                auto string = Span<const char>(next.start, next.length);

                Value parsed_value = Value::parse_value(trim_string_span(string));
               
                ColumnView column = df.add_column(parsed_value);
            }
        }

        auto row_memory = static_cast<DataFrame::Value*>(alloca(sizeof(DataFrame::Value) * df._column_shifts.count - 1));

        auto row = Span<DataFrame::Value>(row_memory, df._column_shifts.count);

        do
        {
            for (uint32_t i = 0; i < row.length; ++i)
            {
                row[i].is_null = true;
            }

            for (uint32_t column_offset = 0; file_iterator.has_next(); ++column_offset)
            {
                auto next = file_iterator.next();

                if (next.type != DataReader::DelimitedTokenType::VALUE)
                {
                    file_line += 1; break;
                }

                if (column_offset >= row.length)
                {
                    throw std::out_of_range("Encountered extra column on line: " + std::to_string(file_line)
                        + " while trying to parse " + std::string(path)
                        + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
                }

                Span<const char> string = trim_string_span(Span<const char>(next.start, next.length));

                Column& column = df.get_raw_column_unsafe(column_offset);

                if (column._type == Type::STRING)
                {
                    row[column_offset] = Value(string);

                    continue;
                }

                Value parsed_value = Value::parse_value(string);

                if (parsed_value.type == column._type || parsed_value.is_null)
                {
                    row[column_offset] = parsed_value;

                    continue;
                }

                if (parsed_value.type == Type::STRING)
                {
                    column.reinterpret_type(Type::STRING, true);
                }
                else if (parsed_value.type == Type::DATETIME)
                {
                    column.reinterpret_type(Type::DATETIME, true);
                }
                else if (parsed_value.type == Type::FLOAT && column._type == Type::INT)
                {
                    column.reinterpret_type(Type::FLOAT, true);
                }

                row[column_offset] = parsed_value;
            }

            df.add_row(row);
        } 
        while (!file_iterator.is_end());

        return df;
    }
}

/*******************************************************************************************************************
 DATA WRITER
 ******************************************************************************************************************/
namespace nml
{
    MatrixOwner DataFrame::to_matrix()
    {
        Span<uint32_t> span = _column_shifts.to_span();

        return to_matrix(span);
    }

    MatrixOwner DataFrame::to_matrix(std::initializer_list<uint32_t> columns)
    {
        auto start = const_cast<uint32_t*>(columns.begin());

        auto span = Span<uint32_t>(start, columns.size());

        return to_matrix(span);
    }

    MatrixOwner DataFrame::to_matrix(Span<uint32_t> columns)
    {
        auto matrix = MatrixOwner(_row_count, columns.length);

        MatrixSpan matrix_span = matrix.to_span();

        for (uint32_t column_offset = 0; column_offset < columns.length; ++column_offset)
        {
            Column& column = _columns[columns[column_offset]];

            if (column._type == Type::STRING)
            {
                throw std::invalid_argument("Null value not allowed in non-nullable column "
                    + std::to_string(column_offset) + "."
                    + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
            }

            if (column._type == Type::FLOAT)
            {
                auto floats = column.get_values_float();

                if (column._is_nullable)
                {
                    for (uint64_t row_offset = 0; row_offset < _row_count; ++row_offset)
                    {
                        if (column._null_index.check(row_offset))
                        {
                            throw std::invalid_argument("Matrices may not contain null values. "
                                "(Row: " + std::to_string(row_offset)
                                + ", Column: " +  std::to_string(column_offset) + ")"
                                + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
                        }

                        matrix_span.set_unsafe(row_offset, column_offset, static_cast<float>(floats[row_offset]));
                    }
                }
                else
                {
                    for (uint64_t row_offset = 0; row_offset < _row_count; ++row_offset)
                    {
                        matrix_span.set_unsafe(row_offset, column_offset, static_cast<float>(floats[row_offset]));
                    }
                }
            }
            else
            {
                auto ints = column.get_values_int();

                if (column._is_nullable)
                {
                    for (uint64_t row_offset = 0; row_offset < _row_count; ++row_offset)
                    {
                        if (column._null_index.check(row_offset))
                        {
                            throw std::invalid_argument("Matrices may not contain null values. "
                                "(Row: " + std::to_string(row_offset)
                                + ", Column: " +  std::to_string(column_offset) + ")"
                                + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
                        }

                        matrix_span.set_unsafe(row_offset, column_offset, static_cast<float>(ints[row_offset]));
                    }
                }
                else
                {
                    for (uint64_t row_offset = 0; row_offset < _row_count; ++row_offset)
                    {
                        matrix_span.set_unsafe(row_offset, column_offset, static_cast<float>(ints[row_offset]));
                    }
                }
            }
        }

        return matrix;
    }
}

namespace nml
{
    using namespace nml::dataframe_internal;

    DataFrame::DataFrame() noexcept
        : _row_count(0)
        , _columns(ResizableList<Column>())
        , _column_shifts(ResizableList<uint32_t>())
    {
        initialize_allocated_columns();
    }

    DataFrame::~DataFrame() noexcept
    {
        for (uint32_t i = 0; i < _columns.capacity; ++i)
        {
            _columns[i].~Column();
        }
    }

    DataFrame::DataFrame(const DataFrame& copy) noexcept
        : _row_count(copy._row_count)
        , _columns(ResizableList<Column>())
        , _column_shifts(ResizableList<uint32_t>())
    {
        initialize_allocated_columns();

        for (uint32_t i = 0; i < copy._column_shifts.count; ++i)
        {
            auto offset = copy._column_shifts[i];
            add_column( get_column_unsafe(offset), i);
        }
    }

    DataFrame::DataFrame(DataFrame&& move) noexcept
        : _row_count(move._row_count)
        , _columns(std::move(move._columns))
        , _column_shifts(std::move(move._column_shifts))
    {
        move._row_count = 0;
    }

    DataFrame& DataFrame::operator=(const DataFrame& copy) noexcept
    {
        if (this != &copy)
        {
            this->~DataFrame();
            new (this) DataFrame(copy);
        }

        return *this;
    }

    DataFrame& DataFrame::operator=(DataFrame&& move) noexcept
    {
        if (this != &move)
        {
            this->~DataFrame();
            new(this) DataFrame(std::move(move));
        }

        return *this;
    }

    void DataFrame::initialize_allocated_columns() noexcept
    {
        for (uint32_t i = _columns.count; i < _columns.capacity; ++i)
        {
            _columns[i]._memory = nullptr;
        }
    }

    DataFrame::RowView DataFrame::operator[](uint64_t row)
    {
        if (row >= _row_count)
        {
            throw std::out_of_range("Row out of bounds: " + std::to_string(row)
                + "\n(File: " + std::string(__FILE__) + ", Line: " + std::to_string(__LINE__) + ")");
        }

        return get_row_unsafe(row);
    }    
    
    DataFrame::ReferenceValue DataFrame::operator()(uint64_t row, uint32_t column)
    {
        if (row >= _row_count)
        {
            throw std::out_of_range("Row out of bounds: " + std::to_string(row)
                + "\n(File: " + std::string(__FILE__) + ", Line: " + std::to_string(__LINE__) + ")");
        }

        if (column >= _column_shifts.count)
        {
            throw std::out_of_range("Column out of bounds: " + std::to_string(column)
                + "\n(File: " + std::string(__FILE__) + ", Line: " + std::to_string(__LINE__) + ")");
        }

        return ReferenceValue(*this, column, row);
    }

    int32_t DataFrame::find_column_offset(const char* name) noexcept
    {
        if (!*name) return -1;

        for (int32_t c = 0; c < _column_shifts.count; ++c)
        {
            auto& column = _columns[_column_shifts[c]];

            if (!column._column_name[0]) continue;

            const char* match_name = name;
            const char* column_name = column._column_name;

            while (*column_name && *match_name)
            {
                char cn = *column_name, mn = *match_name;

                if (cn >= 'A' && cn <= 'Z') cn += 32;
                if (mn >= 'A' && mn <= 'Z') mn += 32;

                if (cn != mn) break;

                column_name++, match_name++;
            }

            if (*column_name == *match_name)
            {
                return c;
            }
        }

        return -1;
    }

    int32_t DataFrame::find_column_offset(Span<const char> name) noexcept
    {
        if (name.length == 0) return -1;

        for (int32_t c = 0; c < _column_shifts.count; ++c)
        {
            auto& column = _columns[_column_shifts[c]];

            if (!column._column_name[0]) continue;

            const char* column_name = column._column_name;

            uint64_t ch = 0;

            for (; ch < name.length; ++ch, ++column_name)
            {
                char cn = *column_name, mn = name[ch];

                if (cn >= 'A' && cn <= 'Z') cn += 32;
                if (mn >= 'A' && mn <= 'Z') mn += 32;

                if (cn != mn) break;
            }

            if (ch == name.length && *column_name == 0)
            {
                return c;
            }
        }

        return -1;
    }

    DataFrame::ColumnView DataFrame::get_column_unsafe(uint32_t offset) noexcept
    {
        return ColumnView(*this, _column_shifts[offset]);
    }

    NMLResult<DataFrame::ColumnView> DataFrame::get_column(uint32_t offset) noexcept
    {
        if (offset >= _column_shifts.count)
        {
            return NMLResult<DataFrame::ColumnView>::err(NMLErrorCode::OUT_OF_BOUNDS);
        }

        return NMLResult<DataFrame::ColumnView>::ok(get_column_unsafe(offset));
    }

    DataFrame::ColumnView DataFrame::get_column_by_name_unsafe(const std::string& name) noexcept
    {
        return get_column_by_name_unsafe(name.c_str());
    }

    DataFrame::ColumnView DataFrame::get_column_by_name_unsafe(const char* name) noexcept
    {
        auto offset = find_column_offset(name);

        if (offset < 0)
        {
            throw std::out_of_range("Column not found: " + std::string(name)
                + "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")");
        }
        else
        {
            return get_column_unsafe(offset);
        }
    }

    NMLResult<DataFrame::ColumnView> DataFrame::get_column_by_name(const std::string& name) noexcept
    {
        return get_column_by_name(name.c_str());
    }

    NMLResult<DataFrame::ColumnView> DataFrame::get_column_by_name(const char* name) noexcept
    {
        auto offset = find_column_offset(name);

        if (offset < 0)
        {
            return NMLResult<DataFrame::ColumnView>::err(NMLErrorCode::NOT_FOUND);
        }
        else
        {
            return NMLResult<DataFrame::ColumnView>::ok(get_column_unsafe(offset));
        }
    }

    DataFrame::RowView DataFrame::get_row_unsafe(uint64_t row) noexcept
    {
        return RowView(*this, row);
    }

    NMLResult<DataFrame::RowView> DataFrame::get_row(uint64_t row) noexcept
    {
        if (row >= _row_count)
        {
            return NMLResult<DataFrame::RowView>::err(NMLErrorCode::OUT_OF_BOUNDS);
        }

        return NMLResult<DataFrame::RowView>::ok(get_row_unsafe(row));
    }

    inline void DataFrame::throw_if_column_name_taken(const char* name)
    {
        if (find_column_offset(name) > 0)
        {
            throw std::invalid_argument("DataFrame already contains Column with name "
                + std::string(name) + "."
                "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
        }
    }

    inline void DataFrame::throw_if_column_name_taken(Span<const char> name)
    {
        if (find_column_offset(name) > 0)
        {
            throw std::invalid_argument("DataFrame already contains Column with name "
                + std::string(name.get_pointer(), name.length) + "."
                "\n(File: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ")\n");
        }
    }

    uint32_t DataFrame::find_next_column_offset() noexcept
    {
        auto bytes = Bitset::required_bytes(_column_shifts.count);

        void* memory_raw = alloca(bytes);

        auto memory = MemorySpan(memory_raw, bytes);

        auto bitset = Bitset(memory, true);

        for (uint32_t i = 0; i < _column_shifts.count; ++i)
        {
            bitset.set(_column_shifts[i]);
        }

        for (uint32_t i = 0; i < _column_shifts.count; ++i)
        {
            if (!bitset.check_unsafe(i))
            {
                return i;
            }
        }

        return _column_shifts.count;
    }
}

#endif //NML_DATA_FRAME_H