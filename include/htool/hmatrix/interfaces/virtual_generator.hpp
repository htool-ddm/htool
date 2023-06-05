#ifndef HTOOL_GENERATOR_HPP
#define HTOOL_GENERATOR_HPP

#include <cassert>
#include <iterator>

namespace htool {

template <typename CoefficientPrecision>
class VirtualGenerator {

  public:
    virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const = 0;
    // virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const = 0;

    VirtualGenerator() {}
    VirtualGenerator(const VirtualGenerator &)            = default;
    VirtualGenerator &operator=(const VirtualGenerator &) = default;
    VirtualGenerator(VirtualGenerator &&)                 = default;
    VirtualGenerator &operator=(VirtualGenerator &&)      = default;
    virtual ~VirtualGenerator() {}
};

template <typename CoefficientPrecision>
class VirtualGeneratorWithPermutation : public VirtualGenerator<CoefficientPrecision> {

  protected:
    const std::vector<int> &m_target_permutation;
    const std::vector<int> &m_source_permutation;

  public:
    VirtualGeneratorWithPermutation(const std::vector<int> &target_permutation, const std::vector<int> &source_permutation) : m_target_permutation(target_permutation), m_source_permutation(source_permutation) {}

    virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        copy_submatrix(M, N, m_target_permutation.data() + row_offset, m_source_permutation.data() + col_offset, ptr);
    }

    virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const = 0;
};

} // namespace htool

#endif
