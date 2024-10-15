#ifndef HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP

#include "../../clustering/cluster_node.hpp" // for Cluster
#include "../../matrix/matrix.hpp"           // for Matrix
#include "../../misc/misc.hpp"               // for underlying_type
#include "virtual_generator.hpp"             // for VirtualGenerator

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class VirtualInternalLowRankGenerator {
  public:
    VirtualInternalLowRankGenerator() {}

    // C style
    virtual void copy_low_rank_approximation(const VirtualInternalGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const = 0;

    virtual bool is_htool_owning_data() const { return true; }
    virtual ~VirtualInternalLowRankGenerator() {}
};

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class VirtualLowRankGenerator {
  public:
    VirtualLowRankGenerator() {}

    // C style
    virtual void copy_low_rank_approximation(const VirtualInternalGenerator<CoefficientPrecision> &A, int M, int N, const int *rows, const int *cols, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const = 0;

    virtual bool is_htool_owning_data() const { return true; }
    virtual ~VirtualLowRankGenerator() {}
};

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class InternalLowRankGenerator : public VirtualInternalLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> {

  protected:
    const VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &m_low_rank_generator;

  public:
    InternalLowRankGenerator(const VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &low_rank_generator) : m_low_rank_generator(low_rank_generator) {
    }

    virtual void copy_low_rank_approximation(const VirtualInternalGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {
        m_low_rank_generator.copy_low_rank_approximation(A, t.get_size(), s.get_size(), t.get_permutation().data() + t.get_offset(), s.get_permutation().data() + s.get_offset(), epsilon, rank, U, V);
    }

    virtual bool is_htool_owning_data() const override { return m_low_rank_generator.is_htool_owning_data(); }
};

} // namespace htool

#endif
