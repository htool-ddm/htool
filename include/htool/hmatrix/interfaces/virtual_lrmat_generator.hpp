#ifndef HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP

#include "../../clustering/cluster_node.hpp" // for Cluster
#include "../../hmatrix/lrmat/lrmat.hpp"     // for LowRankMatrix
#include "../../matrix/matrix.hpp"           // for Matrix
#include "../../misc/misc.hpp"               // for underlying_type

namespace htool {

template <typename CoefficientPrecision>
class VirtualInternalLowRankGenerator {
  public:
    VirtualInternalLowRankGenerator() {}

    // C style
    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const = 0;

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const = 0;

    virtual ~VirtualInternalLowRankGenerator() {}
};

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class VirtualLowRankGenerator {
  public:
    VirtualLowRankGenerator() {}

    // C style
    virtual bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, LowRankMatrix<CoefficientPrecision> &lrmat) const = 0;

    virtual bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const = 0;

    virtual ~VirtualLowRankGenerator() {}
};

template <typename CoefficientPrecision>
class InternalLowRankGenerator : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

  protected:
    const VirtualLowRankGenerator<CoefficientPrecision> &m_low_rank_generator;
    const int *m_target_permutation;
    const int *m_source_permutation;

  public:
    InternalLowRankGenerator(const VirtualLowRankGenerator<CoefficientPrecision> &low_rank_generator, const int *target_permutation, const int *source_permutation) : m_low_rank_generator(low_rank_generator), m_target_permutation(target_permutation), m_source_permutation(source_permutation) {
    }

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return m_low_rank_generator.copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat);
    }

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return m_low_rank_generator.copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, reqrank, lrmat);
    }

};

} // namespace htool

#endif
