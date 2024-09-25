//
// Created by nik on 6/16/2024.
//


#include <gtest/gtest.h>

#include <string_view>
#include <unordered_set>

#include "../primitives/hash.h"

using namespace nml;

TEST(hash_tests, basic_hash)
{
    const char* test_string_1 = "1 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ligula mi, molestie at laoreet sit amet, tempor id urna. Etiam sagittis bibendum neque aliquet fringilla. Vivamus luctus arcu eget dui consequat pellentesque. Nullam a sagittis ante, at varius tellus. Morbi efficitur in felis id congue. Mauris mattis ex sit amet aliquam fermentum. Integer sollicitudin massa eget malesuada interdum. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Donec id facilisis lorem, sit amet imperdiet erat. Sed rutrum dapibus arcu interdum blandit. Integer vel risus nec dolor fermentum tincidunt ac in est. Ut eu mauris sit amet nunc dictum cursus. Sed tincidunt elementum ex vel lobortis. Praesent at turpis eget leo rhoncus finibus sit amet rhoncus nulla.";
    const char* test_string_2 = "2 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ligula mi, molestie at laoreet sit amet, tempor id urna. Etiam sagittis bibendum neque aliquet fringilla. Vivamus luctus arcu eget dui consequat pellentesque. Nullam a sagittis ante, at varius tellus. Morbi efficitur in felis id congue. Mauris mattis ex sit amet aliquam fermentum. Integer sollicitudin massa eget malesuada interdum. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Donec id facilisis lorem, sit amet imperdiet erat. Sed rutrum dapibus arcu interdum blandit. Integer vel risus nec dolor fermentum tincidunt ac in est. Ut eu mauris sit amet nunc dictum cursus. Sed tincidunt elementum ex vel lobortis. Praesent at turpis eget leo rhoncus finibus sit amet rhoncus nulla.";

    uint64_t hash_1 = hash_internal::hash_value(test_string_1);
    uint64_t hash_2 = hash_internal::hash_value(test_string_2);

    ASSERT_NE(hash_1, hash_2);
}

TEST(hash_tests, hash_integers)
{
    int positive = 100;
    int negative = -100;

    uint64_t hash_1 = hash_internal::hash_value(positive);
    uint64_t hash_2 = hash_internal::hash_value(negative);

    ASSERT_NE(hash_1, hash_2);
}

TEST(hash_tests, hash_span_char)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    uint64_t hash_1 = hash_internal::hash_value(Span<char>(string + 0, 8));
    uint64_t hash_2 = hash_internal::hash_value(Span<char>(string + 0, 8));
    uint64_t hash_3 = hash_internal::hash_value(Span<char>(string + 11, 8));

    ASSERT_EQ(hash_1, hash_2);
    ASSERT_NE(hash_1, hash_3);
}

TEST(hash_tests, compare_span_char)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    bool comparison1 = hash_internal::compare_values(Span<char>(string + 0, 8), Span<char>(string + 0, 8));
    bool comparison2 = hash_internal::compare_values(Span<char>(string + 0, 8), Span<char>(string + 11, 8));

    ASSERT_TRUE(comparison1);
    ASSERT_FALSE(comparison2);
}

TEST(hash_tests, hash_string_view)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    uint64_t hash_1 = hash_internal::hash_value(std::string_view(string + 0, 8));
    uint64_t hash_2 = hash_internal::hash_value(std::string_view(string + 0, 8));
    uint64_t hash_3 = hash_internal::hash_value(std::string_view(string + 11, 8));

    ASSERT_EQ(hash_1, hash_2);
    ASSERT_NE(hash_1, hash_3);
}

TEST(hash_tests, compare_string_view)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    bool comparison1 = hash_internal::compare_values(std::string_view(string + 0, 8), std::string_view(string + 0, 8));
    bool comparison2 = hash_internal::compare_values(std::string_view(string + 0, 8), std::string_view(string + 11, 8));

    ASSERT_TRUE(comparison1);
    ASSERT_FALSE(comparison2);
}

TEST(hash_tests, hash_span_span_int)
{
    int list_1[] = { 1, 2, 3, 4 };
    int list_2[] = { 1, 2, 3, 4 };
    int list_3[] = { 4, 3, 2, 1 };

    Span<int> span_span_1[] = { Span<int>(list_1, 4), Span<int>(list_1, 4) };
    Span<int> span_span_2[] = { Span<int>(list_2, 4), Span<int>(list_1, 4) };
    Span<int> span_span_3[] = { Span<int>(list_1, 4), Span<int>(list_3, 4) };

    uint64_t hash_1 = hash_internal::hash_value(Span<Span<int>>(span_span_1, 2));
    uint64_t hash_2 = hash_internal::hash_value(Span<Span<int>>(span_span_2, 2));
    uint64_t hash_3 = hash_internal::hash_value(Span<Span<int>>(span_span_3, 2));

    ASSERT_EQ(hash_1, hash_2);
    ASSERT_NE(hash_1, hash_3);
}

TEST(hash_tests, compare_span_span_int)
{
    int list_1[] = { 1, 2, 3, 4 };
    int list_2[] = { 1, 2, 3, 4 };
    int list_3[] = { 4, 3, 2, 1 };

    Span<int> span_span_1[] = { Span<int>(list_1, 4), Span<int>(list_1, 4) };
    Span<int> span_span_2[] = { Span<int>(list_2, 4), Span<int>(list_1, 4) };
    Span<int> span_span_3[] = { Span<int>(list_1, 4), Span<int>(list_3, 4) };

    bool comparison1 = hash_internal::compare_values(Span<Span<int>>(span_span_1, 2), Span<Span<int>>(span_span_2, 2));
    bool comparison2 = hash_internal::compare_values(Span<Span<int>>(span_span_1, 2), Span<Span<int>>(span_span_3, 2));

    ASSERT_TRUE(comparison1);
    ASSERT_FALSE(comparison2);
}

struct CustomType
{
    uint8_t padded_value;
    uint64_t remaining_value;

    [[nodiscard]] uint64_t hash() const
    { 
        auto padded_value_hash = std::hash<uint8_t>{}(padded_value);
        auto remaining_value_hash = std::hash<uint64_t>{}(remaining_value);

        return padded_value_hash ^ (remaining_value_hash << 1) ^ (remaining_value_hash >> 1);
    }

    bool operator==(const CustomType& rhs) const { return padded_value == rhs.padded_value && remaining_value == rhs.remaining_value; }
};

namespace std { template<> struct hash<CustomType> { size_t operator()(const CustomType& ct) const noexcept { return ct.hash(); } }; }

TEST(hash_tests, hash_custom_type)
{
    uint64_t hash_1 = hash_internal::hash_value(CustomType { 0, 1 });
    uint64_t hash_2 = hash_internal::hash_value(CustomType { 0, 1 });
    uint64_t hash_3 = hash_internal::hash_value(CustomType { 0, 2 });

    ASSERT_EQ(hash_1, hash_2);
    ASSERT_NE(hash_1, hash_3);
}

TEST(hash_tests, compare_custom_type)
{
    bool comparison1 = hash_internal::compare_values(CustomType{ 0, 1 }, CustomType{ 0, 1 });
    bool comparison2 = hash_internal::compare_values(CustomType{ 0, 1 }, CustomType{ 0, 2 });

    ASSERT_TRUE(comparison1);
    ASSERT_FALSE(comparison2);
}

TEST(hash_tests, insert_contains)
{
    int range = 20;
    auto set = HashSet<int>();

    for (int i = -range; i <= range; i += 2)
    {
        bool inserted = set.insert(i);
    }

    int ct = 0, sum = 0;

    for (int i = -range; i <= range; i += 1)
    {
        bool contains = set.contains(i);

        if (contains)
        {
            sum += i;
//            if (ct++ > 0) std::cout << "," << i;
//            else std::cout << i;
        }
    }

    ASSERT_EQ(0, sum);
    ASSERT_GT(set.count(), 0);

//    std::cout << '\n' << "Sum: " << sum << '\n';
}

TEST(hash_tests, insert_remove_iterate)
{
    int range = 200'000, sum1 = 0, ct1 = 0;
    auto set = HashSet<int>();

    for (int i = -range; i <= range; i += 2)
    {
        sum1 += i; ct1 += 1;
        bool inserted = set.insert(i);
        ASSERT_TRUE(inserted);
    }

    {
        int ct = 0, sum = 0;

        for (auto& value : set)
        {
            ct += 1, sum += value;
        }

        ASSERT_EQ(sum, sum1);
        ASSERT_EQ(ct, set.count());
        ASSERT_EQ(ct1, set.count());
    }

    for (int i = -range; i <= range; i += 4)
    {
        ct1 -= 1, sum1 -= i;
        bool removed = set.remove(i);
        ASSERT_TRUE(removed);
    }

    {
        int ct = 0, sum = 0;

        for (auto& value : set)
        {
            sum += value, ct += 1;
        }

        ASSERT_EQ(sum, sum1);
        ASSERT_EQ(ct, set.count());
        ASSERT_EQ(ct1, set.count());
    }
}

TEST(hash_tests, insert_count_strings)
{
    char string_1[] = "string 1";
    char string_2[] = "string 2";
    char string_3[] = "string 3";
    char string_4[] = "string 4";
    char string_5[] = "string 5";

    auto set = HashSet<char*>();

    for (unsigned i = 0; i < 10; ++i)
    {
        set.insert(string_1);
        set.insert(string_2);
        set.insert(string_3);
        set.insert(string_4);
        set.insert(string_5);
    }

    int ct = 0;

    for (auto& value : set)
    {
        ct += 1;
    }

    ASSERT_EQ(5, ct);
}

TEST(hash_tests, insert_count_string_spans)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    auto set = HashSet<std::string_view>();

    for (unsigned i = 0; i < 10; ++i)
    {
        set.insert(std::string_view(string + 0, 8));
        set.insert(std::string_view(string + 11, 8));
        set.insert(std::string_view(string + 22, 8));
        set.insert(std::string_view(string + 33, 8));
        set.insert(std::string_view(string + 44, 8));
    }

    ASSERT_EQ(5, set.count());
}

TEST(hash_tests, insert_count_string_iterator)
{
    auto string_1 = "string 11";
    auto string_2 = "string 12";
    auto string_3 = "string 13";
    auto string_4 = "string 14";
    auto string_5 = "string 15";

    auto set = HashSet<std::string_view>();

    for (unsigned i = 0; i < 100; ++i)
    {
        set.insert(std::string_view(string_1));
        set.insert(std::string_view(string_2));
        set.insert(std::string_view(string_3));
        set.insert(std::string_view(string_4));
        set.insert(std::string_view(string_5));
    }

    ASSERT_EQ(5, set.count());
}

TEST(hash_tests, insert_benchmark)
{
    int range = 2'000, iterations = 1'000;

    auto hash = HashSet<int>();
    auto set = std::unordered_set<int>();

    auto start_hash = std::chrono::high_resolution_clock::now();

    for (int i = 0; i <= iterations; ++i)
    {
        for (int j = -range; j <= range; ++j)
        {
            hash.insert(j);
        }
    }

    auto end_hash = std::chrono::high_resolution_clock::now();

    auto start_set = std::chrono::high_resolution_clock::now();

    for (int i = 0; i <= iterations; ++i)
    {
        for (int j = -range; j <= range; ++j)
        {
            set.insert(j);
        }
    }

    auto end_set = std::chrono::high_resolution_clock::now();

    auto duration_hash = std::chrono::duration_cast<std::chrono::milliseconds>(end_hash - start_hash).count();
    auto duration_set = std::chrono::duration_cast<std::chrono::milliseconds>(end_set - start_set).count();

    std::cout << "HashSet Insert Duration: " << duration_hash << " ms" << std::endl;
    std::cout << "std::unordered_set Insert Duration: " << duration_set << " ms" << std::endl;
}


TEST(hash_tests, hash_map_insert_remove)
{
    int range = 100;
    auto map = HashMap<int32_t, uint32_t>();

    for (int i = -range; i <= range; ++i)
    {
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
        map.insert(i, map.count() * 2);
    }

    for (int i = -100; i < 100; ++i)
    {
        auto value = *map.get_value(i);
        ASSERT_EQ(value, (i + 100) * 2);
    }

    for (int i = -100; i <= range; ++i)
    {
        ASSERT_TRUE(map.remove(i));
        ASSERT_FALSE(map.contains_key(i));
    }

    ASSERT_EQ(0, map.count());
}

TEST(hash_tests, hash_map_string_view_insert)
{
    auto string_1 = "string 11";
    auto string_2 = "string 12";
    auto string_3 = "string 13";
    auto string_4 = "string 14";
    auto string_5 = "string 15";

    auto map = HashMap<std::string_view, uint32_t>();

    for (int i = 1; i < 100; ++i)
    {
        map.insert(std::string_view(string_1), i);
        map.insert(std::string_view(string_2), i);
        map.insert(std::string_view(string_3), i);
        map.insert(std::string_view(string_4), i);
        map.insert(std::string_view(string_5), i);
    }

    int ct = 0;

    for (auto& value : map)
    {
        ct += 1;
    }

    ASSERT_EQ(5, ct);
    ASSERT_EQ(5, map.count());
    ASSERT_EQ(*map[std::string_view(string_1)], 1);
}

TEST(hash_tests, hash_set_move)
{
    auto initialize = []()
    {
        auto starting = HashSet<int>();

        for (int i = 0; i < 100; ++i)
        {
            starting.insert(i);
        }

        return starting;
    };

    auto starting = initialize();

    ASSERT_EQ(100, starting.count());
    ASSERT_TRUE(starting.contains(10));

    HashSet<int> moved = std::move(starting);

    ASSERT_EQ(100, moved.count());
    ASSERT_EQ(0, starting.count());
    ASSERT_TRUE(moved.contains(10));
}

TEST(hash_tests, hash_map_move)
{
    auto initialize = []()
    {
        auto starting = HashMap<int, int>();

        for (int i = 0; i < 100; ++i)
        {
            starting.insert(i, i);
        }

        return starting;
    };

    HashMap<int, int> starting = initialize();

    ASSERT_EQ(100, starting.count());
    ASSERT_TRUE(starting.contains_key(10));

    HashMap<int, int> moved = std::move(starting);

    ASSERT_EQ(100, moved.count());
    ASSERT_EQ(0, starting.count());
    ASSERT_TRUE(moved.contains_key(10));
}