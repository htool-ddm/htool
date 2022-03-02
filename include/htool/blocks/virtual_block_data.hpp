#ifndef HTOOL_BLOCKS_VIRTUAL_BLOCK_DATA_HPP
#define HTOOL_BLOCKS_VIRTUAL_BLOCK_DATA_HPP

#include <cassert>
#include <iostream>
#include <iterator>

namespace htool {

template <typename T>
class VirtualBlockData {
  public:
    virtual void add_mvprod_row_major(const T *const in, T *const out, const int &mu, char transb = 'T', char op = 'N') const = 0;

    virtual ~VirtualBlockData(){};
};

} // namespace htool
#endif
