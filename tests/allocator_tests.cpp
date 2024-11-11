//
// Created by nik on 9/26/2024.
//

#include <gtest/gtest.h>

#include "../library/nml/primitives/allocator.h"
#include "../library/nml/primitives/list.h"

using namespace nml;

TEST(allocator_tests, thread_safe)
{
    static uint32_t thread_ct = std::max(1u, std::thread::hardware_concurrency()), per_thread_ct = 2'000, default_ct = 100;

    auto allocator = Allocator<int, true>();
    auto threads = StaticOwnerList<std::thread>(thread_ct);

    allocator.lock();

    for (int thread_id = 0; thread_id < thread_ct; ++thread_id)
    {
        threads.add(std::thread([thread_id, &allocator]()
        {
            for (uint32_t element = 0; element < per_thread_ct; ++element)
            {
                uint64_t element_id = allocator.claim_next_index();
                allocator.get_element(element_id) = thread_id + 1;
            }
        }));
    }

    for (uint32_t i = 0; i < default_ct; ++i)
    {
        uint64_t element_id = allocator.claim_next_index_unsafe();
        allocator.get_element(element_id) = -1;
    }

    allocator.unlock();

    for (auto& t : threads) t.join();

    threads.count = 0;

    for (int thread_id = 0; thread_id < thread_ct; ++thread_id)
    {
        threads.add(std::thread([&allocator]()
        {
            for (uint32_t element = 0; element < per_thread_ct; ++element)
            {
                allocator.return_index(allocator.claim_next_index());
            }
        }));
    }

    for (auto& t : threads) t.join();

    ASSERT_EQ(default_ct + thread_ct * per_thread_ct, allocator.claimed_ct());

    for (uint32_t element_id = 1; element_id <= allocator.claimed_ct(); ++element_id)
    {
        if (element_id <= default_ct)
        {
            ASSERT_EQ(-1, allocator.get_element(element_id));
        }
        else
        {
            int element = allocator.get_element(element_id);
            ASSERT_TRUE(element > 0 && element <= thread_ct);
        }
    }
}