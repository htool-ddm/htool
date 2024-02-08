#ifndef HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP

#include "../../basic_types/matrix.hpp"
#include "../../misc/misc.hpp"
#include "virtual_generator.hpp"
#include <cassert>
#include <iterator>

namespace htool {

/**
 * @brief
 *
 * @tparam CoefficientPrecision
 * @tparam CoordinatesPrecision
 */
template <typename CoefficientPrecision, typename CoordinatesPrecision>
class VirtualLowRankGenerator {
  public:
    VirtualLowRankGenerator() {}

    // C style
    /**
     * @brief Build a low rank approximation of the block defined by \p t , \p s and \p A . The block has a size \f$ (M \times N)\f$ where \f$ M \f$ is the size of \p t , and \f$ N \f$ is the size of \p s .
     *
     * The low rank approximation is defined as \f$ \mathbf{U} \mathbf{V}\f$, where \f$\mathbf{U}\f$ has a size \f$ (M \times r)\f$ and \f$\mathbf{V}\f$ has a size \f$ (r \times N)\f$, \f$ r\f$ being the resulting \p rank . The resulting \f$\mathbf{U}\f$ and \f$\mathbf{V}\f$ are stored on output in \p U and \p V.
     *
     * @param[in] A VirtualGenerator given to generator coefficient of the approximated kernel.
     * @param[in] t target Cluster that defines the rows of the approximated block
     * @param[in] s source Cluster that defines the columns of the approximated block
     * @param[in] epsilon input tolerance
     * @param[inout] rank On input, given tolerance or if \p rank <0, the approximation is expected to continue up to an epsilon tolerance. On output, it is the obtained rank at the end of the low rank approximation.
     * @param[out] U \f$ \mathbf{U} \f$-part of the low-rank approximation on output
     * @param[out] V \f$ \mathbf{V} \f$-part of the low-rank approximation on output
     */
    virtual void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const = 0;

    /**
     * @brief Check if htool needs to desallocate or not the data stored in \p U and \p V .
     *
     * @return true
     * @return false
     */
    virtual bool is_htool_owning_data() const { return true; }
    virtual ~VirtualLowRankGenerator() {}
};

} // namespace htool

#endif
