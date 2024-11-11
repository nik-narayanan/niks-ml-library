//
// Created by nik on 6/16/2024.
//


#include <gtest/gtest.h>
#include "../library/nml/primitives/tree_red_black.h"

using namespace nml;

TEST(tree_tests, inserts)
{
    int range = 10'000;

    auto tree = RedBlackTree<int>();

    for (int i = -range; i <= range; ++i)
    {
        tree.insert(i);
    }
}

TEST(tree_tests, deletes_in_order)
{
    int range = 10'000;

    auto tree = RedBlackTree<int>();

    for (int i = -range; i <= range; ++i)
    {
        tree.insert(i);
    }

    for (int i = -range; i <= range; ++i)
    {
        uint64_t min_index = tree.min();
        auto& min_node = tree.get_node(min_index);
        ASSERT_EQ(min_node.data, i);
        tree.remove(i);
    }

    ASSERT_TRUE(tree.is_empty());

    for (int i = -range; i <= range; ++i)
    {
        tree.insert(i);
    }

    for (int i = range; i >= -range; --i)
    {
        uint64_t max_index = tree.max();
        auto& max_node = tree.get_node(max_index);
        ASSERT_EQ(max_node.data, i);
        tree.remove(i);
    }

    ASSERT_TRUE(tree.is_empty());
}


TEST(tree_tests, benchmark)
{
    int range = 10'000;

    auto tree = RedBlackTree<int>();
    auto set = std::set<int>();

    auto start_set = std::chrono::high_resolution_clock::now();

    for (int i = -range; i <= range; ++i)
    {
        set.insert(-i);
    }

    for (int i = -range; i <= range; ++i)
    {
        set.insert(-i);
    }

    auto end_set = std::chrono::high_resolution_clock::now();

    auto start_tree = std::chrono::high_resolution_clock::now();

    for (int i = -range; i <= range; ++i)
    {
        tree.insert(i, true);
    }

    for (int i = -range; i <= range; ++i)
    {
        tree.insert(i, true);
    }

    auto end_tree = std::chrono::high_resolution_clock::now();

    auto duration_tree = std::chrono::duration_cast<std::chrono::milliseconds>(end_tree - start_tree).count();
    auto duration_set = std::chrono::duration_cast<std::chrono::milliseconds>(end_set - start_set).count();

    std::cout << "RedBlackTree Insert Duration: " << duration_tree << " ms" << std::endl;
    std::cout << "std::set Insert Duration: " << duration_set << " ms" << std::endl;
}