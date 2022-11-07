#ifndef HTOOL_BLOCKS_SUM_EXPRESSIONS_HPP
#define HTOOL_BLOCKS_SUM_EXPRESSIONS_HPP

namespace htool {

template <typename T>
class SumExpression {
  private:
    std::vector < std::pair<Block<T> *, Block<T> *> SumExpressionR;
    std::vector < std::pair<Block<T> *, Block<T> *> SumExpressionH;

  public:
    SumExpression() {}

    // append_R(Block<T> *, Block<T> *);
    // append_H(Block<T> *, Block<T> *);
};
} // namespace htool
#endif
