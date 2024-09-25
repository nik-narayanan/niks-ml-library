//
// Created by nik on 6/16/2024.
//

#ifndef NML_TREE_RED_BLACK_H
#define NML_TREE_RED_BLACK_H

#include "list.h"
#include "heap.h"
#include "allocator.h"

namespace nml
{
    template<typename T>
    struct RedBlackTree
    {
        struct Node
        {
            uint64_t left = 0;
            uint64_t right = 0;
            uint64_t parent = 0;

            T data;
            bool is_red = true;

            inline void initialize() noexcept
            {
                right = 0; parent = 0;
                left = 0; is_red = true;
            }

            Node(Node&& other) = delete;
            Node(const Node& other) = delete;
            Node& operator=(Node&& other) = delete;
            Node& operator=(const Node& other) = delete;
        };

        explicit RedBlackTree(uint64_t initial_capacity = 0) noexcept;

        RedBlackTree(const RedBlackTree& other) = delete;
        RedBlackTree& operator=(const RedBlackTree& other) = delete;

        [[nodiscard]] bool is_empty() noexcept { return _root_index == 0; }
        [[nodiscard]] uint64_t get_root_index() noexcept { return _root_index; }
        [[nodiscard]] inline Node& get_node(uint64_t index) noexcept { return _node_allocator.get_element(index); }
        [[nodiscard]] inline Node& get_parent(uint64_t index) noexcept { return get_node(get_node(index).parent); }

        bool remove(T value) noexcept;
        [[nodiscard]] uint64_t find(T value) noexcept;
        uint64_t insert(T value, bool insert_unique = false) noexcept;

        [[nodiscard]] uint64_t min(uint64_t index = 0) noexcept;
        [[nodiscard]] uint64_t max(uint64_t index = 0) noexcept;

        void print(uint64_t index, int space = 0) noexcept;

    private:

        uint64_t _root_index;
        Allocator<Node> _node_allocator;

        void _insert_adjust(uint64_t index) noexcept;
        void _delete_adjust(uint64_t index) noexcept;

        void _rotate_left(uint64_t index) noexcept;
        void _rotate_right(uint64_t index) noexcept;
        void _transplant(uint64_t root_index, uint64_t transplant_index) noexcept;
    };

    template<typename T>
    RedBlackTree<T>::RedBlackTree(uint64_t initial_capacity) noexcept
        : _root_index(0)
        , _node_allocator(Allocator<Node>(initial_capacity))
    { }

    template<typename T>
    uint64_t RedBlackTree<T>::min(uint64_t index) noexcept
    {
        if (is_empty()) return 0;

        if (index == 0) index = _root_index;

        uint64_t previous = index;

        while (index != 0)
        {
            previous = index;
            index = get_node(index).left;
        }

        return previous;
    }

    template<typename T>
    uint64_t RedBlackTree<T>::max(uint64_t index) noexcept
    {
        if (is_empty()) return 0;

        if (index == 0) index = _root_index;

        uint64_t previous = index;

        while (index != 0)
        {
            previous = index;
            index = get_node(index).right;
        }

        return previous;
    }

    template<typename T>
    uint64_t RedBlackTree<T>::find(const T value) noexcept
    {
        uint64_t current = _root_index;

        while (current != 0)
        {
            Node& current_node = get_node(current);

            if (value < current_node.data)
            {
                current = current_node.left;
            }
            else if (value > current_node.data)
            {
                current = current_node.right;
            }
            else
            {
                return current;
            }
        }

        return 0;
    }

    template<typename T>
    uint64_t RedBlackTree<T>::insert(const T value, const bool insert_unique) noexcept
    {
        uint64_t parent_index = 0;
        uint64_t current_index = _root_index;

        while (current_index != 0)
        {
            parent_index = current_index;

            Node& parent_node = get_node(parent_index);

            if (insert_unique && value == parent_node.data)
            {
                return parent_index;
            }
            else if (value < parent_node.data)
            {
                current_index = parent_node.left;
            }
            else
            {
                current_index = parent_node.right;
            }
        }

        uint64_t new_index = _node_allocator.claim_next_index();
        Node& new_node = get_node(new_index);

        new_node.data = value;

        if (is_empty())
        {
            _root_index = new_index;
            new_node.is_red = false;
            return 0;
        }

        new_node.parent = parent_index;
        Node& parent_node = get_node(parent_index);

        if (value < parent_node.data)
        {
            parent_node.left = new_index;
        }
        else
        {
            parent_node.right = new_index;
        }

        _insert_adjust(new_index);

        return 0;
    }

    template<typename T>
    bool RedBlackTree<T>::remove(const T value) noexcept
    {
        uint64_t index = find(value);

        if (index == 0) return false;

        Node& node = get_node(index);

        uint64_t current_index = 0;
        uint64_t parent_index = index;
        bool started_red = node.is_red;

        if (node.left == 0)
        {
            current_index = node.right;
            _transplant(index, node.right);
        }
        else if (node.right == 0)
        {
            current_index = node.left;
            _transplant(index, node.left);
        }
        else
        {
            parent_index = min(node.right);
            Node& parent_node = get_node(parent_index);

            started_red = parent_node.is_red;
            current_index = parent_node.right;

            if (parent_node.parent == index)
            {
                if (current_index != 0)
                {
                    get_node(current_index).parent = parent_index;
                }
            }
            else
            {
                if (current_index != 0)
                {
                    get_node(current_index).parent = parent_node.parent;
                }

                _transplant(parent_index, parent_node.right);

                if (parent_node.right != 0)
                {
                    get_node(parent_node.right).parent = parent_index;
                }

                parent_node.right = node.right;

                if (parent_node.right != 0)
                {
                    get_node(parent_node.right).parent = parent_index;
                }
            }

            _transplant(index, parent_index);

            parent_node.left = node.left;

            if (parent_node.left != 0)
            {
                get_node(parent_node.left).parent = parent_index;
            }

            parent_node.is_red = node.is_red;
        }

        if (!started_red && current_index != 0)
        {
            _delete_adjust(current_index);
        }

        _node_allocator.return_index(index);

        return true;
    }

    template<typename T>
    void RedBlackTree<T>::_insert_adjust(uint64_t index) noexcept
    {
        while (index != _root_index && get_parent(index).is_red)
        {
            uint64_t parent_index = get_node(index).parent;
            uint64_t grandparent_index = get_node(parent_index).parent;

            if (get_node(index).parent == get_node(grandparent_index).left)
            {
                uint64_t uncle_right_index = get_node(grandparent_index).right;

                if (uncle_right_index != 0 && get_node(uncle_right_index).is_red)
                {
                    get_node(parent_index).is_red = false;
                    get_node(grandparent_index).is_red = true;
                    get_node(uncle_right_index).is_red = false;

                    index = grandparent_index;
                }
                else
                {
                    if (index == get_node(parent_index).right)
                    {
                        index = parent_index;
                        _rotate_left(index);
                    }

                    parent_index = get_node(index).parent;
                    grandparent_index = get_node(parent_index).parent;

                    get_node(parent_index).is_red = false;
                    get_node(grandparent_index).is_red = true;

                    _rotate_right(grandparent_index);
                }
            }
            else
            {
                int64_t uncle_left_index = get_node(grandparent_index).left;

                if (uncle_left_index != 0 && get_node(uncle_left_index).is_red)
                {
                    get_node(uncle_left_index).is_red = false;
                    get_node(parent_index).is_red = false;
                    get_node(grandparent_index).is_red = true;

                    index = grandparent_index;
                }
                else
                {
                    if (index == get_node(parent_index).left)
                    {
                        index = parent_index;
                        _rotate_right(index);
                    }

                    parent_index = get_node(index).parent;
                    grandparent_index = get_node(parent_index).parent;

                    get_node(parent_index).is_red = false;
                    get_node(grandparent_index).is_red = true;

                    _rotate_left(grandparent_index);
                }
            }
        }

        if (!is_empty())
        {
            get_node(_root_index).is_red = false;
        }
    }

    template<typename T>
    void RedBlackTree<T>::_delete_adjust(uint64_t index) noexcept
    {
        while (index != _root_index && index != 0 && !get_node(index).is_red)
        {
            uint64_t parent_index = get_node(index).parent;

            if (index == get_node(parent_index).left)
            {
                uint64_t brother_right_index = get_node(parent_index).right;

                if (get_node(brother_right_index).is_red)
                {
                    get_node(brother_right_index).is_red = false;
                    get_node(parent_index).is_red = true;

                    _rotate_left(parent_index);

                    brother_right_index = get_node(parent_index).right;
                }

                if ((get_node(brother_right_index).left == 0 || !get_node(get_node(brother_right_index).left).is_red) &&
                    (get_node(brother_right_index).right == 0 || !get_node(get_node(brother_right_index).right).is_red))
                {
                    get_node(brother_right_index).is_red = true;
                    index = get_node(index).parent;
                }
                else
                {
                    if (get_node(brother_right_index).right == 0 ||
                       !get_node(get_node(brother_right_index).right).is_red)
                    {
                        if (get_node(brother_right_index).left != 0)
                        {
                            get_node(get_node(brother_right_index).left).is_red = false;
                        }

                        get_node(brother_right_index).is_red = true;

                        _rotate_right(brother_right_index);

                        index = get_parent(index).right;
                    }

                    get_node(brother_right_index).is_red = get_parent(index).is_red;

                    get_parent(index).is_red = false;

                    if (get_node(brother_right_index).right != 0)
                    {
                        get_node(get_node(brother_right_index).right).is_red = false;
                    }

                    _rotate_left(get_node(index).parent);
                    index = _root_index;
                }
            }
            else
            {
                uint64_t brother_left_index = get_node(parent_index).left;

                if (get_node(brother_left_index).is_red)
                {
                    get_node(brother_left_index).is_red = false;
                    get_node(parent_index).is_red = true;

                    _rotate_right(parent_index);

                    brother_left_index = get_parent(index).left;
                }

                if ((get_node(brother_left_index).right == 0 ||
                    !get_node(get_node(brother_left_index).right).is_red)
                    &&
                    (get_node(brother_left_index).left == 0 ||
                    !get_node(get_node(brother_left_index).left).is_red))
                {
                    get_node(brother_left_index).is_red = true;
                    index = get_node(index).parent;
                }
                else
                {
                    if (get_node(brother_left_index).left == 0 ||
                       !get_node(get_node(brother_left_index).left).is_red)
                    {
                        if (get_node(brother_left_index).right != 0)
                        {
                            get_node(get_node(brother_left_index).right).is_red = false;
                        }

                        get_node(brother_left_index).is_red = true;
                        _rotate_left(brother_left_index);

                        brother_left_index = get_parent(index).left;
                    }

                    get_node(brother_left_index).is_red = get_parent(index).is_red;
                    get_parent(index).is_red = false;

                    if (get_node(brother_left_index).left != 0)
                    {
                        get_node(get_node(brother_left_index).left).is_red = false;
                    }

                    _rotate_right(get_node(index).parent);
                    index = _root_index;
                }
            }
        }

        if (index != 0)
        {
            get_node(index).is_red = false;
        }
    }

    template<typename T>
    void RedBlackTree<T>::_rotate_left(const uint64_t index) noexcept
    {
        if (index == 0 || get_node(index).right == 0)
        {
            return;
        }

        Node& node = get_node(index);

        uint64_t right_index = node.right;
        Node& right_node = get_node(right_index);

        node.right = right_node.left;

        if (right_node.left != 0)
        {
            Node& right_left_node = get_node(right_node.left);
            right_left_node.parent = index;
        }

        right_node.parent = node.parent;

        if (node.parent == 0)
        {
            _root_index = right_index;
        }
        else if (index == get_parent(index).left)
        {
            get_parent(index).left = right_index;
        }
        else
        {
            get_parent(index).right = right_index;
        }

        get_node(right_index).left = index;
        node.parent = right_index;
    }

    template<typename T>
    void RedBlackTree<T>::_rotate_right(uint64_t index) noexcept
    {
        if (index == 0 || get_node(index).left == 0)
        {
            return;
        }

        Node& node = get_node(index);

        uint64_t left_index = node.left;
        Node& left_node = get_node(left_index);

        node.left = left_node.right;

        if (left_node.right != 0)
        {
            Node& left_right_node = get_node(left_node.right);
            left_right_node.parent = index;
        }

        left_node.parent = node.parent;

        if (node.parent == 0)
        {
            _root_index = left_index;
        }
        else if (index == get_parent(index).left)
        {
            get_parent(index).left = left_index;
        }
        else
        {
            get_parent(index).right = left_index;
        }

        get_node(left_index).right = index;
        node.parent = left_index;
    }

    template<typename T>
    void RedBlackTree<T>::_transplant(uint64_t root_index, uint64_t transplant_index) noexcept
    {
        Node& root_node = get_node(root_index);
        Node& transplant_node = get_node(transplant_index);

        if (root_node.parent == 0)
        {
            _root_index = transplant_index;
        }
        else if (root_index == get_node(root_node.parent).left)
        {
            get_node(root_node.parent).left = transplant_index;
        }
        else
        {
            get_node(root_node.parent).right = transplant_index;
        }

        if (transplant_index != 0)
        {
            transplant_node.parent = root_node.parent;
        }
    }

    template<typename T>
    void RedBlackTree<T>::print(uint64_t index, int space) noexcept
    {
        constexpr int COUNT = 5;
        if (index == 0) return;

        Node& node = get_node(index);

        space += COUNT;
        print(node.right, space);

        std::cout << std::endl;

        for (int i = COUNT; i < space; i++)
            std::cout << " ";

        std::cout << node.data << "(" << ((node.is_red) ? "RED" : "BLACK") << ")" << std::endl;

        print(node.left, space);
    }
}

#endif //NML_TREE_RED_BLACK_H