#ifndef HTOOL_DENSE_GENERATOR_HPP
#define HTOOL_DENSE_GENERATOR_HPP

#include "vector.hpp"
#include <cassert>
#include <iterator>

namespace htool {

template <typename T>
class VirtualDenseBlocksGenerator {
  public:
    // C style
    virtual void copy_dense_blocks(const std::vector<int> &M, const std::vector<int> &N, const std::vector<const int *> &rows, const std::vector<const int *> &cols, std::vector<T *> &ptr) const = 0;

    virtual ~VirtualDenseBlocksGenerator(){};
};

} // namespace htool

#endif
