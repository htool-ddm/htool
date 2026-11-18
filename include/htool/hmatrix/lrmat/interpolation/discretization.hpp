#ifndef HTOOL_HMATRIX_LRMAT_INTERPOLATION_DISCRETIZATION_HPP
#define HTOOL_HMATRIX_LRMAT_INTERPOLATION_DISCRETIZATION_HPP

#include "../../../clustering/cluster_node.hpp"
#include "../../../matrix/matrix.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wold-style-cast"
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#    pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdouble-promotion"
#    pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <theia_interpolation_only.hpp>

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

namespace htool {

template <typename CoordinatePrecision, int dimension>
using interpolation_box = std::pair<std::array<CoordinatePrecision, dimension>, std::array<CoordinatePrecision, dimension>>; // {min, max}

// Number of Chebyshev nodes per axis needed for epsilon.
inline int interpolation_order(double epsilon) {
    return int(std::ceil(std::log(1. / epsilon) / std::log(10))) + 1;
}

// Number of tensor-product interpolation nodes, L^dimension.
template <int dimension>
int number_of_interpolation_nodes(int L) {
    int n_interp = 1;
    for (int d = 0; d < dimension; d++)
        n_interp *= L;
    return n_interp;
}

// Entry of W in the tensor form of a discretization's interpolation operator: particle -> local dof.
template <typename CoefficientPrecision>
struct interpolation_weight {
    int particle;
    int dof;
    CoefficientPrecision value;
};

// Pads a box by a fixed margin, and further if it is degenerate along some axis.
template <typename CoordinatePrecision, int dimension>
void inflate_box(interpolation_box<CoordinatePrecision, dimension> &box) {
    for (int dim = 0; dim < dimension; dim++) {
        box.second[dim] += 0.001;
        box.first[dim] -= 0.001;
        if (std::abs(box.second[dim] - box.first[dim]) < std::numeric_limits<CoordinatePrecision>::epsilon() * 100) {
            box.second[dim] += 0.01;
            box.first[dim] -= 0.01;
        }
    }
}

// One side (target or source) of an interpolation-based compressor, attached to the cluster tree of that side.
// Dofs are given as a range [offset, offset + size) of the tree's permutation, like the row/col offsets of
// copy_low_rank_approximation. Methods are const and must be safe to call concurrently.
template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class VirtualDiscretization {
  public:
    // Cluster tree whose permutation offsets refer to; the compressor's precomputation walks it.
    virtual const Cluster<CoordinatePrecision> &cluster() const = 0;

    // Inflated box that the interpolation of these dofs must cover.
    virtual interpolation_box<CoordinatePrecision, dimension> box(int offset, int size) const = 0;

    // L^dimension x size matrix mapping these dofs to the interpolation nodes of box (L nodes per dimension).
    virtual Matrix<CoefficientPrecision> interpolation_matrix(int L, interpolation_box<CoordinatePrecision, dimension> box, int offset, int size) const = 0;

    // Same operator in tensor form: P2M at the discretization's particles, then W (particle -> local dof),
    // so that interpolation_matrix == to_dense(p2m) * W.
    virtual void tensor_interpolation(int L, interpolation_box<CoordinatePrecision, dimension> box, int offset, int size, theia::TensorP2M<dimension, CoefficientPrecision> &p2m, std::vector<interpolation_weight<CoefficientPrecision>> &w) const = 0;

    virtual ~VirtualDiscretization() = default;
};

} // namespace htool

#endif
