//
// Created by nik on 4/1/2024.
//

#include <random>
#include <gtest/gtest.h>

#include "../primitives/heap_min_max.h"
#include "../primitives/memory_owner.h"
#include "../primitives/heap.h"

using namespace nml;

const bool print_results = true;

TEST(data_structures_tests, heap_memory_boundaries)
{
    unsigned queue_size = 20;

    auto memory = MemoryOwner(MinMaxHeap<ScoredValue<int>>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span(0);

    auto queue = MinMaxHeap<ScoredValue<int>>(memory_span);

    int i = 0;
    for (; i < 1000; ++i)
    {
        queue.insert_min({static_cast<float>(i), i});
    }

    i = 0;
    unsigned count = 0;
    while (!queue.is_empty())
    {
        auto next = queue.remove_min_unsafe();
        ASSERT_EQ(next.value, i++);
        ++count;
    }

    ASSERT_EQ(count, queue_size);

    i = 0;
    for (; i < 1000; ++i)
    {
        queue.insert_max({static_cast<float>(i), i});
    }

    count = 0;
    while (!queue.is_empty())
    {
        auto next = queue.remove_max_unsafe();
        ASSERT_EQ(next.value, --i);
        ++count;
    }

    ASSERT_EQ(count, queue_size);
}

int random_int(int min, int max)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<> dis(min, max);

    return dis(gen);
}

TEST(data_structures_tests, heap_switch)
{
    unsigned queue_size = 1000;

    auto memory = MemoryOwner(MinMaxHeap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span(0);

    auto queue = MinMaxHeap<int>(memory_span);

    int test_count = 10'000;

    auto sorted = std::vector<int>(test_count);
    auto unsorted = std::vector<int>(test_count);

    for (int i = 0; i < test_count; ++i)
    {
        unsorted[i] = random_int(-test_count, test_count);
        sorted[i] = unsorted[i];
    }

    std::sort(sorted.begin(), sorted.end());

    int i = 0;
    for (; i < test_count; ++i)
    {
        queue.insert_min(unsorted[i]);
    }

    i = 0;
    while (!queue.is_empty())
    {
        auto next = queue.remove_min_unsafe();
        ASSERT_EQ(next, sorted[i++]);
    }

    for (; i < test_count; ++i)
    {
        queue.insert_min(unsorted[i]);
    }

    i = 0;
    for (; i < test_count; ++i)
    {
        queue.insert_max(unsorted[i]);
    }

    while (!queue.is_empty())
    {
        auto next = queue.remove_max_unsafe();
        ASSERT_EQ(next, sorted[--i]);
    }
}

TEST(data_structures_tests, heap_benchmark)
{
    unsigned queue_size = 1000;

    auto memory = MemoryOwner(MinMaxHeap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span(0);

    auto queue = MinMaxHeap<int>(memory_span);

    int test_count = 100'000;

    auto unsorted = std::vector<int>(test_count);

    for (int i = 0; i < test_count; ++i)
    {
        unsorted[i] = random_int(-test_count, test_count);
    }

    auto start = std::chrono::high_resolution_clock::now();

    int i = 0;
    for (; i < test_count; ++i)
    {
        queue.insert_min(unsorted[i]);
    }

    while (!queue.is_empty())
    {
        auto next = queue.remove_min_unsafe();
    }

    i = 0;
    for (; i < test_count; ++i)
    {
        queue.insert_max(unsorted[i]);
    }

    while (!queue.is_empty())
    {
        auto next = queue.remove_max_unsafe();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "1000 Element Average Time (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / test_count << "\n";
}

//TEST(data_structures_tests, bst_)
//{
//    std::vector<int> vec = {5, 1, 7, 3, 9, 2, 6, 2};
//
//    auto bst = BinarySearchTree(vec);
//
//    bst.print_inorder();
//
//    bst.insert(4);
//    bst.print_inorder();
////    bst.dump_memory();
//
////    bst.remove(1);
//    bst.remove(5);
//    bst.print_inorder();
////    bst.dump_memory();
//
//    std::cout << std::endl;
//
//    std::cout << "Contains 6? " << bst.contains(6) << std::endl; // Should print 1 (true)
//    std::cout << "Contains 10? " << bst.contains(10) << std::endl; // Should print 0 (false)
//}
//
//TEST(data_structures_tests, benchmark_ann_split)
//{
//    int trials = 20;
//    std::vector<int> test_sizes = {10, 100, 1'000, 10'000, 100'000, 1'000'000};
//
//    for (const auto test_size:test_sizes)
//    {
//        size_t test_1_time = 0, test_2_time = 0;
//
//        for (int trial = 0; trial < trials; ++trial)
//        {
//            auto test_root = std::vector<int>(test_size);
//
//            for (int i = 0; i < test_size; ++i)
//            {
//                test_root[i] = random_int(-test_size, test_size);
//            }
//
//            auto start_1 = std::chrono::high_resolution_clock::now();
//
//            std::vector<int> odd;
//            std::vector<int> even;
//
//            for (int i = 0; i < test_size; ++i)
//            {
//                bool is_even = test_root[i] % 2 == 0;
//
//                if (is_even) even.push_back(test_root[i]);
//                else odd.push_back(test_root[i]);
//            }
//
//            auto end_1 = std::chrono::high_resolution_clock::now();
//
//            test_1_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_1 - start_1).count();
//
//            auto test_2 = std::vector(test_root);
//
//            auto start_2 = std::chrono::high_resolution_clock::now();
//
//            unsigned last_left = 0, last_left_checked = 0;
//            unsigned left_index = 0, right_index = test_2.size() - 1;
//
//            while (left_index < right_index)
//            {
//                bool continue_left = true, continue_right = true;
//
//                while (left_index < right_index && continue_left)
//                {
//                    last_left_checked = left_index;
//                    bool is_even = test_2[left_index] % 2 == 0;
//
//                    if (!is_even) last_left = left_index++;
//                    else continue_left = false;
//                }
//
//                while (left_index < right_index && continue_right)
//                {
//                    bool is_even = test_2[right_index] % 2 == 0;
//
//                    if (is_even) --right_index;
//                    else continue_right = false;
//                }
//
//                if (left_index < right_index)
//                {
//                    last_left_checked = last_left = left_index;
//                    std::swap(test_2[left_index++], test_2[right_index--]);
//                }
//            }
//
//            if (left_index < test_2.size() && last_left_checked != left_index)
//            {
//                bool is_even = test_2[left_index] % 2 == 0;
//                if (!is_even) last_left = left_index;
//            }
//
//            auto end_2 = std::chrono::high_resolution_clock::now();
//
//            test_2_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_2 - start_2).count();
//
//            if (std::max((size_t)1, odd.size()) - 1 != last_left)
//            {
//                std::cout << (unsigned)std::max((size_t)0, odd.size() - 1) << "\n";
//                std::cout << "last_left: " << last_left << "\n";
//                std::cout << "{";
//                for (int i = 0; i < test_2.size(); ++i) { if (i > 0) std::cout << ","; std::cout << test_2[i]; } std::cout << "}" << std::endl;
//            }
//        }
//
//        std::cout << "Test 1 Size " << test_size << " Average Time (ns): " << test_1_time / trials << "\n";
//        std::cout << "Test 2 Size " << test_size << " Average Time (ns): " << test_2_time / trials << "\n";
//    }
//}

TEST(data_structures_tests, heap_max)
{
    const int queue_size = 20;

    auto memory = MemoryOwner(Heap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span();

    auto heap_max = Heap<int, false>(memory_span);

    for (int i = 0; i < queue_size; ++i)
    {
        heap_max.push(i);
    }

    int value = queue_size;
    while (!heap_max.is_empty())
    {
        ASSERT_EQ(--value, heap_max.pop());
    }

    for (int i = queue_size; i > 0; --i)
    {
        heap_max.push(i);
    }

    value = queue_size;
    while (!heap_max.is_empty())
    {
        ASSERT_EQ(value--, heap_max.pop());
    }
}

TEST(data_structures_tests, heap_min)
{
    const int queue_size = 20;

    auto memory = MemoryOwner(Heap<int>::required_bytes(queue_size));

    auto memory_span = memory.to_memory_span();

    auto heap_min = Heap<int, true>(memory_span);

    for (int i = 0; i < queue_size; ++i)
    {
        heap_min.push(i);
    }

    int value = 0;
    while (!heap_min.is_empty())
    {
        ASSERT_EQ(value++, heap_min.pop());
    }

    for (int i = queue_size; i > 0; --i)
    {
        heap_min.push(i);
    }

    value = 0;
    while (!heap_min.is_empty())
    {
        ASSERT_EQ(++value, heap_min.pop());
    }
}