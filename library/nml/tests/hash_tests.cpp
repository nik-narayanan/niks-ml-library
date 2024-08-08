//
// Created by nik on 6/16/2024.
//


#include <gtest/gtest.h>
#include <unordered_set>
#include "../primitives/hash.h"

using namespace nml;

TEST(hash_tests, basic_hash)
{
    const char* test_string_1 = "1 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ligula mi, molestie at laoreet sit amet, tempor id urna. Etiam sagittis bibendum neque aliquet fringilla. Vivamus luctus arcu eget dui consequat pellentesque. Nullam a sagittis ante, at varius tellus. Morbi efficitur in felis id congue. Mauris mattis ex sit amet aliquam fermentum. Integer sollicitudin massa eget malesuada interdum. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Donec id facilisis lorem, sit amet imperdiet erat. Sed rutrum dapibus arcu interdum blandit. Integer vel risus nec dolor fermentum tincidunt ac in est. Ut eu mauris sit amet nunc dictum cursus. Sed tincidunt elementum ex vel lobortis. Praesent at turpis eget leo rhoncus finibus sit amet rhoncus nulla.";
    const char* test_string_2 = "2 Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ligula mi, molestie at laoreet sit amet, tempor id urna. Etiam sagittis bibendum neque aliquet fringilla. Vivamus luctus arcu eget dui consequat pellentesque. Nullam a sagittis ante, at varius tellus. Morbi efficitur in felis id congue. Mauris mattis ex sit amet aliquam fermentum. Integer sollicitudin massa eget malesuada interdum. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Donec id facilisis lorem, sit amet imperdiet erat. Sed rutrum dapibus arcu interdum blandit. Integer vel risus nec dolor fermentum tincidunt ac in est. Ut eu mauris sit amet nunc dictum cursus. Sed tincidunt elementum ex vel lobortis. Praesent at turpis eget leo rhoncus finibus sit amet rhoncus nulla.";

    uint64_t hash_1 = hash_internal::hash_value(MemorySpan((void *) test_string_1, strlen(test_string_1)));
    uint64_t hash_2 = hash_internal::hash_value(MemorySpan((void *) test_string_2, strlen(test_string_2)));

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

    ASSERT_EQ(hash_1, hash_2);
}

TEST(hash_tests, compare_span_char)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    bool comparison = hash_internal::compare_values(Span<char>(string + 0, 8), Span<char>(string + 0, 8));

    ASSERT_TRUE(comparison);
}

TEST(hash_tests, insert_contains)
{
    auto set = HashSet<int>();

    for (int i = 0; i < 100; i += 2)
    {
        bool inserted = set.insert(i);
    }

    for (int i = 0; i < 100; i += 2)
    {
        bool inserted = set.insert(i);
    }

    int ct = 0, sum = 0;

    for (int i = 0; i < 100; i += 1)
    {
        bool contains = set.contains(i);

        if (contains)
        {
            sum += i;
//            if (ct++ > 0) std::cout << "," << i;
//            else std::cout << i;
        }
    }

    ASSERT_EQ(2450, sum);

//    std::cout << '\n' << "Sum: " << sum << '\n';
}

TEST(hash_tests, insert_iterate)
{
    auto set = HashSet<int>();

    for (int i = -100; i <= 100; i += 2)
    {
        bool inserted = set.insert(i);
    }

    for (int i = 0; i < 100; i += 2)
    {
        bool inserted = set.insert(i);
    }

    int ct = 0, sum = 0;
    auto iterator = set.to_iterator();

    while (iterator.has_next())
    {
        auto next = iterator.next();
        sum += next;
//        if (ct++ > 0) std::cout << "," << next;
//        else std::cout << next;
    }

    ASSERT_EQ(101, set.count());

//    std::cout << '\n' << "Sum: " << sum << '\n';
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
    auto iterator = set.to_iterator();

    while (iterator.has_next())
    {
        auto next = iterator.next();
//        if (ct++ > 0) std::cout << ", " << next;
//        else std::cout << next;
    }

    ASSERT_EQ(5, set.count());

//    std::cout << '\n' << "Distinct Strings: " << ct << '\n';
}

TEST(hash_tests, insert_count_string_spans)
{
    char string[] = "string 1 | string 2 | string 3 | string 4 | string 5";

    auto set = HashSet<Span<char>>();

    for (unsigned i = 0; i < 10; ++i)
    {
        set.insert(Span<char>(string + 0, 8));
        set.insert(Span<char>(string + 11, 8));
        set.insert(Span<char>(string + 22, 8));
        set.insert(Span<char>(string + 33, 8));
        set.insert(Span<char>(string + 44, 8));
    }

    int ct = 0;
    auto iterator = set.to_iterator();

    while (iterator.has_next())
    {
        auto next = iterator.next();
//        if (ct++ > 0) std::cout << ", ", next.print("", false);
//        else next.print("", false);
    }

    ASSERT_EQ(5, set.count());

//    std::cout << '\n' << "Distinct Strings: " << ct << '\n';
}

struct StringIterator
{
    const char* _string;
    unsigned _length;

    uint64_t _index;

    explicit StringIterator(const char* string)
        : _string(string), _length(strlen(string)), _index(0)
    { }

    [[nodiscard]] inline int64_t length() const noexcept
    {
        return _length;
    }

    [[nodiscard]] inline bool is_end() const noexcept
    {
        return _index > _length;
    }

    [[nodiscard]] inline char next() const noexcept
    {
        return _string[_index - 1];
    }

    [[nodiscard]] inline bool has_next() noexcept
    {
        _index += 1;

        return !is_end();
    }

    inline void reset() noexcept
    {
        _index = 0;
    }
};

TEST(hash_tests, insert_count_string_iterator)
{
    auto string_1 = StringIterator("string 11");
    auto string_2 = StringIterator("string 12");
    auto string_3 = StringIterator("string 13");
    auto string_4 = StringIterator("string 14");
    auto string_5 = StringIterator("string 15");

    auto set = HashSet<Iterator<char, StringIterator>>();

    for (unsigned i = 0; i < 100; ++i)
    {
        set.insert(Iterator<char, StringIterator>(string_1));
        set.insert(Iterator<char, StringIterator>(string_2));
        set.insert(Iterator<char, StringIterator>(string_3));
        set.insert(Iterator<char, StringIterator>(string_4));
        set.insert(Iterator<char, StringIterator>(string_5));
    }

    int ct = 0;
    auto iterator = set.to_iterator();

    while (iterator.has_next())
    {
        auto next = iterator.next();
//        if (ct++ > 0) std::cout << ", " << next.container._string;
//        else std::cout << next.container._string;
    }

    ASSERT_EQ(5, set.count());

//    std::cout << '\n' << "Distinct Strings: " << ct << '\n';
}

TEST(tree_tests, benchmark)
{
    int range = 10'000, iterations = 1'000;

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


TEST(hash_tests, hash_map)
{
    auto map = HashMap<int32_t, uint32_t>();

    for (int i = -100; i < 100; ++i)
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
}

TEST(hash_tests, hash_map_iterators)
{
    auto string_1 = StringIterator("string 11");
    auto string_2 = StringIterator("string 12");
    auto string_3 = StringIterator("string 13");
    auto string_4 = StringIterator("string 14");
    auto string_5 = StringIterator("string 15");

    auto map = HashMap<Iterator<char, StringIterator>, uint32_t>();

    for (unsigned i = 0; i < 100; ++i)
    {
        map.insert(Iterator<char, StringIterator>(string_1), i);
        map.insert(Iterator<char, StringIterator>(string_2), i);
        map.insert(Iterator<char, StringIterator>(string_3), i);
        map.insert(Iterator<char, StringIterator>(string_4), i);
        map.insert(Iterator<char, StringIterator>(string_5), i);
    }
}