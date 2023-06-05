#ifndef HTOOL_GENERATOR_HPP
#define HTOOL_GENERATOR_HPP

namespace htool {

template <typename CoefficientPrecision>
class VirtualInternalGenerator {

  public:
    virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const = 0;

    VirtualInternalGenerator() {}
    VirtualInternalGenerator(const VirtualInternalGenerator &)            = default;
    VirtualInternalGenerator &operator=(const VirtualInternalGenerator &) = default;
    VirtualInternalGenerator(VirtualInternalGenerator &&)                 = default;
    VirtualInternalGenerator &operator=(VirtualInternalGenerator &&)      = default;
    virtual ~VirtualInternalGenerator() {}
};

template <typename CoefficientPrecision>
class VirtualGenerator {

  public:
    virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const = 0;

    VirtualGenerator() {}
    VirtualGenerator(const VirtualGenerator &)            = default;
    VirtualGenerator &operator=(const VirtualGenerator &) = default;
    VirtualGenerator(VirtualGenerator &&)                 = default;
    VirtualGenerator &operator=(VirtualGenerator &&)      = default;
    virtual ~VirtualGenerator() {}
};

template <typename CoefficientPrecision>
class InternalGeneratorWithPermutation : public VirtualInternalGenerator<CoefficientPrecision> {

  protected:
    const VirtualGenerator<CoefficientPrecision> &m_generator_in_user_numbering;
    const int *m_target_permutation;
    const int *m_source_permutation;

  public:
    InternalGeneratorWithPermutation(const VirtualGenerator<CoefficientPrecision> &generator_in_user_numbering, const int *target_permutation, const int *source_permutation) : m_generator_in_user_numbering(generator_in_user_numbering), m_target_permutation(target_permutation), m_source_permutation(source_permutation) {
    }

    virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        m_generator_in_user_numbering.copy_submatrix(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, ptr);
    }
};

} // namespace htool

#endif
