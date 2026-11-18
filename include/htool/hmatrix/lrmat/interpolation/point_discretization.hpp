#ifndef HTOOL_HMATRIX_LRMAT_INTERPOLATION_POINT_DISCRETIZATION_HPP
#define HTOOL_HMATRIX_LRMAT_INTERPOLATION_POINT_DISCRETIZATION_HPP

#include "discretization.hpp"
#include <algorithm>

namespace htool {

// Point cloud: each dof is a point, the interpolation operator is the P2M at the points (W is the identity).
// points holds dimension coordinates per point in user numbering; points and cluster must outlive this object.
template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class PointDiscretization final : public VirtualDiscretization<CoefficientPrecision, CoordinatePrecision, dimension> {
    const CoordinatePrecision *m_points;
    const Cluster<CoordinatePrecision> *m_cluster;

    std::vector<std::array<CoordinatePrecision, dimension>> points(int offset, int size) const {
        std::vector<std::array<CoordinatePrecision, dimension>> result(size);
        for (int i = 0; i < size; i++) {
            std::copy_n(m_points + std::size_t(dimension) * m_cluster->get_permutation()[offset + i], dimension, result[i].begin());
        }
        return result;
    }

  public:
    PointDiscretization(const CoordinatePrecision *points, const Cluster<CoordinatePrecision> &cluster) : m_points(points), m_cluster(&cluster) {}

    const Cluster<CoordinatePrecision> &cluster() const override { return *m_cluster; }

    interpolation_box<CoordinatePrecision, dimension> box(int offset, int size) const override {
        interpolation_box<CoordinatePrecision, dimension> result;
        result.first.fill(std::numeric_limits<CoordinatePrecision>::max());
        result.second.fill(std::numeric_limits<CoordinatePrecision>::lowest());
        for (const auto &point : points(offset, size)) {
            for (int dim = 0; dim < dimension; dim++) {
                result.first[dim]  = std::min(result.first[dim], point[dim]);
                result.second[dim] = std::max(result.second[dim], point[dim]);
            }
        }
        inflate_box<CoordinatePrecision, dimension>(result);
        return result;
    }

    Matrix<CoefficientPrecision> interpolation_matrix(int L, interpolation_box<CoordinatePrecision, dimension> box, int offset, int size) const override {
        Matrix<CoefficientPrecision> result(number_of_interpolation_nodes<dimension>(L), size);
        auto local_points          = points(offset, size);
        CoefficientPrecision *data = result.data();
        theia::get_polynomials<dimension, CoordinatePrecision, CoefficientPrecision, 0>(L, data, box.first.data(), box.second.data(), local_points.data(), size);
        return result;
    }

    void tensor_interpolation(int L, interpolation_box<CoordinatePrecision, dimension> box, int offset, int size, theia::TensorP2M<dimension, CoefficientPrecision> &p2m, std::vector<interpolation_weight<CoefficientPrecision>> &w) const override {
        auto local_points = points(offset, size);
        std::array<int, dimension> Ls;
        Ls.fill(L);
        theia::P2M<dimension, CoordinatePrecision, CoefficientPrecision, 0>(box.first.data(), box.second.data(), size, local_points.data(), Ls.data(), p2m, theia::op_tensor_interpolation<dimension, CoordinatePrecision, CoefficientPrecision, 0>{});
        w.clear();
        for (int i = 0; i < size; i++) {
            w.push_back({i, i, CoefficientPrecision(1)});
        }
    }
};

} // namespace htool

#endif
