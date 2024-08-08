//
// Created by nik on 5/8/2024.
//

#ifndef NML_REGRESSION_SUMMARY_H
#define NML_REGRESSION_SUMMARY_H

#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "span.h"

namespace nml
{
    struct SummaryTable
    {
        struct Item
        {
            std::string name;
            bool is_float;
            union
            {
                float value_float;
                unsigned value_unsigned;
            };
        };

        struct Grouping
        {
            std::string title;
            std::vector<std::string> column_headers;
            std::vector<std::string> variable_names;
            std::vector<std::vector<float>> column_values;
        };

        std::string title;
        std::vector<Item> row_items;
        std::vector<Grouping> groups;
        std::vector<std::string> warnings;

        explicit SummaryTable(std::string&& title) noexcept
            : title(std::move(title))
        { }

        void print();
    };
}

namespace nml::summary_print_internal
{
    static inline std::string format_number(unsigned number)
    {
        int count = 0;
        std::string formatted;
        std::string number_str = std::to_string(number);

        for (int i = number_str.length() - 1; i >= 0; i--)
        {
            if (count == 3)
            {
                formatted.push_back(',');
                count = 0;  // Reset the counter after inserting a comma
            }

            formatted.push_back(number_str[i]);  // Add the current digit to the formatted string
            count++;
        }

        std::reverse(formatted.begin(), formatted.end());

        return formatted;
    }

    static inline std::string format_float(float number)
    {
        float abs = std::abs(number);

        if ((abs < 1000 && abs > 0.001) || abs == 0)
        {
            return std::to_string(number);
        }
        else
        {
            std::stringstream ss;
            ss << std::scientific << std::setprecision(4) << number;
            return ss.str();
        }
    }

    static inline std::string pad_right(const std::string& title, unsigned length)
    {
        auto result = std::string(length, ' ');

        for (unsigned i = 0; i < title.length() && i < length; ++i)
        {
            result[i] = title[i];
        }

        if (title.length() > length)
        {
            result[length - 1] = '.';
            result[length - 2] = '.';
            result[length - 3] = '.';
        }

        return result;
    }

    static inline std::string pad_left(const std::string& title, unsigned length)
    {
        auto result = std::string(length, ' ');

        unsigned start = length - title.length();

        for (unsigned i = 0; i < title.length() && i + start < length; ++i)
        {
            result[i + start] = title[i];
        }

        return result;
    }

    static inline std::string item_string(SummaryTable::Item& item, unsigned length)
    {
        auto column = std::string(length, ' ');

        std::string number_string = item.is_float ?
                format_float(item.value_float) : format_number(item.value_unsigned);

        for (unsigned i = 0; i < item.name.length(); ++i)
        {
            column[i] = item.name[i];
        }

        column[item.name.length()] = ':';

        unsigned number_start = length - number_string.length();

        for (unsigned i = 0; i < number_string.length(); ++i)
        {
            column[number_start + i] = number_string[i];
        }

        return column;
    }

    static inline bool is_whitespace(const char ch) noexcept
    {
        return ch ==  ' ' || ch == '\t' ||
               ch == '\n' || ch == '\r' ||
               ch == '\f' || ch == '\v' ;
    }

    static inline void break_lines(std::string& string, uint32_t max_length)
    {
        uint32_t copied = 0, line = 0, last_word_break = 0;

        std::string result;

        while (copied < string.size())
        {
            if (is_whitespace(string[copied]))
            {
                last_word_break = copied;
            }

            if (line + 1 >= max_length)
            {
                line = 0, string[last_word_break] ='\n';
            }

            copied++, line++;
        }
    }
}

namespace nml
{
    void SummaryTable::print()
    {
        using namespace summary_print_internal;

        const int width = 43;
        const int padding = 4;
        const int number_width = 10;
        const int table_padding = 2;
        const int table_width = (width) * 2 + padding;
        const int coefficient_table_width = (width) * 2 + padding - 1;
        const int coefficient_table_item_width = number_width + table_padding + 1;

        auto full_title = pad_left(title, title.length() + (table_width - title.length()) / 2);

        std::cout << "\n";
        std::cout << pad_left(title, title.length() + (table_width - title.length()) / 2) << "\n";
        std::cout << std::string(table_width, '=') << "\n";

        for (int i = 0; i < row_items.size(); ++i)
        {
            bool left = i % 2 == 0;

            std::cout << item_string(row_items[i], width);

            if (left)
            {
                std::cout << std::string(padding, ' ');
            }
            else
            {
                std::cout << "\n";
            }
        }

        std::cout << std::string(table_width, '=') << "\n";

        for (int g = 0; g < groups.size(); ++g)
        {
            if (g > 0)
            {
                std::cout << std::string(table_width, '-') << "\n";
            }

            auto& group = groups[g];

            std::cout << pad_right(group.title, 12);

            for (int i = 0; i < group.column_headers.size(); ++i)
            {
                std::cout << pad_left(group.column_headers[i], coefficient_table_item_width);
            }

            std::cout << "\n" << std::string(table_width, '-') << "\n";

            for (int variable = 0; variable < group.variable_names.size(); ++variable)
            {
                std::cout << pad_right(group.variable_names[variable], 12);

                auto column = group.column_values[variable];

                for (int value = 0; value < column.size(); ++value)
                {
                    std::cout << pad_left(format_float(column[value]), coefficient_table_item_width);
                }

                std::cout << '\n';
            }
        }

        std::cout << std::string(table_width, '=') << "\n";

        for (uint32_t i = 0; i < warnings.size(); ++i)
        {
            break_lines(warnings[i], table_width);

            std::cout << warnings[i] << "\n";
        }
    }
}

#endif //NML_REGRESSION_SUMMARY_H