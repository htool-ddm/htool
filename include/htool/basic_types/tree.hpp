#ifndef HTOOL_BASIC_TYPES_TREE_HPP
#define HTOOL_BASIC_TYPES_TREE_HPP
#include <iterator> // for make_move_iterator
#include <memory>   // for unique_ptr, make_shared, shared_ptr
#include <stack>    // for stack
#include <utility>  // for move, forward
#include <vector>   // for vector

namespace htool {

// CRTP base class (https://www.fluentcpp.com/2017/05/19/crtp-helper/, folded
// into TreeNode itself rather than kept as a separate helper since it's the
// only user -- could be removed with the deducing-this feature from C++23).
// Constructors are private + friended to Derived only, so TreeNode<Derived,
// TreeData> can't be constructed (directly, or via some unrelated subclass)
// with a Derived that doesn't match the object's actual most-derived type,
// which underlying()'s static_cast relies on.
template <typename Derived, typename TreeData>
class TreeNode {
  protected:
    std::vector<std::unique_ptr<Derived>> m_children{};
    unsigned int m_depth{0};
    bool m_is_root{true};
    std::shared_ptr<TreeData> m_tree_data{std::make_shared<TreeData>()};

  private:
    TreeNode() = default;
    TreeNode(const TreeNode &rhs) : m_tree_data(rhs.m_tree_data) {}
    TreeNode(TreeNode &&) noexcept = default;
    friend Derived;

  public:
    TreeNode &operator=(const TreeNode &)     = delete;
    TreeNode &operator=(TreeNode &&) noexcept = default;
    virtual ~TreeNode()                       = default;

    Derived &underlying() { return static_cast<Derived &>(*this); }
    Derived const &underlying() const { return static_cast<Derived const &>(*this); }

    template <typename... Args>
    Derived *add_child(Args &&...args) {
        m_children.emplace_back(new Derived(this->underlying(), std::forward<Args>(args)...));
        m_children.back()->m_depth   = m_depth + 1;
        m_children.back()->m_is_root = false;
        return m_children.back().get();
    }

    void steal_children_from(Derived &node) {
        m_children.insert(m_children.end(), std::make_move_iterator(node.m_children.begin()), std::make_move_iterator(node.m_children.end()));
    }

    void delete_children() { m_children.clear(); }
    void assign_children(std::vector<std::unique_ptr<Derived>> &new_children) {
        for (auto &new_child : new_children) {
            m_children.push_back(std::move(new_child));
        }
    }
    unsigned int get_depth() const { return m_depth; }
    // TODO: C++ 23, use std::range https://stackoverflow.com/a/70942702/5913047
    const std::vector<std::unique_ptr<Derived>> &get_children() const { return m_children; }
    std::vector<std::unique_ptr<Derived>> &get_children_with_ownership() { return m_children; }

    bool is_leaf() const { return m_children.empty(); }
    bool is_root() const { return m_is_root; }
};

template <typename NodeType, typename PreOrderFunction>
void preorder_tree_traversal(NodeType &node, PreOrderFunction preorder_visitor) {
    std::stack<NodeType *> node_stack;
    node_stack.push(&node);

    while (!node_stack.empty()) {
        NodeType *current_node = node_stack.top();
        node_stack.pop();
        preorder_visitor(*current_node);

        const auto &children = current_node->get_children();
        for (auto child = children.rbegin(); child != children.rend(); child++) {
            node_stack.push(child->get());
        }
    }
}

template <typename NodeType, typename PostOrderFunction>
void postorder_tree_traversal(NodeType &node, PostOrderFunction postorder_visitor) {
    const auto &children = node.get_children();
    for (auto &child : children) {
        postorder_tree_traversal(*child, postorder_visitor);
    }
    postorder_visitor(node);
}

} // namespace htool

#endif
