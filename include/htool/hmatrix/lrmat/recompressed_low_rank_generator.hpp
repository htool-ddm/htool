#ifndef HTOOL_HMATRIX_LRMAT_RECOMPRESSED_LOW_RANK_GENERATOR_HPP
#define HTOOL_HMATRIX_LRMAT_RECOMPRESSED_LOW_RANK_GENERATOR_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../misc/misc.hpp"                                  // for unde...
#include "../lrmat/utils/SVD_recompression.hpp"
#include <functional>

namespace htool {

template <typename CoefficientPrecision>
class RecompressedLowRankGenerator final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {
    const VirtualInternalLowRankGenerator<CoefficientPrecision> &m_low_rank_generator;
    std::function<void(LowRankMatrix<CoefficientPrecision> &)> m_recompression;

  public:
    RecompressedLowRankGenerator(const VirtualInternalLowRankGenerator<CoefficientPrecision> &low_rank_generator, std::function<void(LowRankMatrix<CoefficientPrecision> &)> recompression) : m_low_rank_generator(low_rank_generator), m_recompression(recompression) {}

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        bool info = m_low_rank_generator.copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat);
        m_recompression(lrmat);
        return info;
    }

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        bool info = m_low_rank_generator.copy_low_rank_approximation(M, N, row_offset, col_offset, reqrank, lrmat);
        return info;
    }
};

} // namespace htool

#endif
