#ifndef HTOOL_BASIC_TYPES_TREE_HPP
#define HTOOL_BASIC_TYPES_TREE_HPP
#include <memory>
#include <stack>
#include <vector>

namespace htool {

// https://www.fluentcpp.com/2017/05/19/crtp-helper/
// It could be removed with using deducing this feature from C++23
template <typename T>
struct CRTPHelper {
    T &underlying() { return static_cast<T &>(*this); }
    T const &underlying() const { return static_cast<T const &>(*this); }
};

// CRTP base class
template <typename Derived, typename TreeData>
class TreeNode : public CRTPHelper<Derived> {
  protected:
    std::vector<std::unique_ptr<Derived>> m_children{};
    unsigned int m_depth{0};
    bool m_is_root{true};
    std::shared_ptr<TreeData> m_tree_data{std::make_shared<TreeData>()};

  public:
    TreeNode() = default;
    // TreeNode(const TreeNode &)                = delete;
    TreeNode &operator=(const TreeNode &)     = delete;
    TreeNode(TreeNode &&) noexcept            = default;
    TreeNode &operator=(TreeNode &&) noexcept = default;
    virtual ~TreeNode()                       = default;

    TreeNode(const TreeNode &rhs) : m_tree_data(rhs.m_tree_data) {}

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

} // namespace htool

#endif
