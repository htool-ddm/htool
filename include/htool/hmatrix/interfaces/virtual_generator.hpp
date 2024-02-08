#ifndef HTOOL_GENERATOR_HPP
#define HTOOL_GENERATOR_HPP

#include <cassert>
#include <iterator>

namespace htool {

/**
 * @brief Define the interface for the user to give Htool a function generating dense sub-blocks of the global matrix the user wants to compress. This is done by the user implementing VirtualGenerator::copy_submatrix.
 *
 * @tparam T Precision of the coefficients (float, double,...).
 */
template <typename CoefficientPrecision>
class VirtualGenerator {

  public:
    /**
     * @brief Generate a dense sub-block of the global matrix the user wants to compress. Note that sub-blocks queried by Htool are potentially non-contiguous in the user's numbering.
     *
     * @param[in] M specifies the number of columns of the queried block
     * @param[in] N specifies the number of rows of the queried block
     * @param[in] row_offset specifies the offset of the queried block.
     * @param[in] col_offset specifies the offset of the queried block.
     * @param[out] ptr is a \p T precision array of size \f$ M\times N\f$. Htool already allocates and desallocates it internally, so it should **not** be allocated by the user.
     */
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
    const int *m_target_permutation;
    const int *m_source_permutation;

  public:
    VirtualGeneratorWithPermutation(const int *target_permutation, const int *source_permutation) : m_target_permutation(target_permutation), m_source_permutation(source_permutation) {
    }

    virtual void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        copy_submatrix_from_user_numbering(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, ptr);
    }

    /**
     * @brief Generate a dense sub-block of the global matrix the user wants to compress. Note that sub-blocks queried by Htool are potentially non-contiguous in the user's numbering.
     *
     * @param[in] M specifies the number of columns of the queried block
     * @param[in] N specifies the number of rows of the queried block
     * @param[in] rows is an integer array of size \f$M\f$. It specifies the queried columns in the user's numbering
     * @param[in] cols is an integer array of size \f$N\f$. It specifies the queried rows in the user's numbering
     * @param[out] ptr is a \p T precision array of size \f$ M\times N\f$. Htool already allocates and desallocates it internally, so it should **not** be allocated by the user.
     */
    virtual void copy_submatrix_from_user_numbering(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const = 0;
};

} // namespace htool

#endif
