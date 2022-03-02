#ifndef HTOOL_BLOCKS_DENSE_BLOCK_DATA_HPP
#define HTOOL_BLOCKS_DENSE_BLOCK_DATA_HPP

#include "../types/matrix.hpp"
#include "blocks.hpp"
#include "virtual_block_data.hpp"
#include <cassert>
#include <iterator>

namespace htool {

template <typename T>
class Block;

template <typename T>
class DenseBlockData : public virtual VirtualBlockData<T>, public Matrix<T> {

  public:
    DenseBlockData(const Block<T> &block, VirtualGenerator<T> &generator, bool use_permutation) : Matrix<T>(block.get_target_cluster().get_size(), block.get_source_cluster().get_size()) {
        if (use_permutation)
            generator.copy_submatrix(this->nr, this->nc, block.get_target_cluster().get_perm_data(), block.get_source_cluster().get_perm_data(), this->mat);
        else {
            std::vector<int> no_perm_target(block.get_target_cluster().get_size()), no_perm_source(block.get_source_cluster().get_size());
            std::iota(no_perm_target.begin(), no_perm_target.end(), block.get_target_cluster().get_offset());
            std::iota(no_perm_source.begin(), no_perm_source.end(), block.get_source_cluster().get_offset());
            generator.copy_submatrix(this->nr, this->nc, no_perm_target.data(), no_perm_source.data(), this->mat);
        }
    }

    void add_mvprod_row_major(const T *const in, T *const out, const int &mu, char transb = 'T', char op = 'N') const override { Matrix<T>::add_mvprod_row_major(in, out, mu, transb, op); };
};

} // namespace htool
#endif
