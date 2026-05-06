#ifndef HTOOL_QUADRATURES_GEOMETRY_HPP
#define HTOOL_QUADRATURES_GEOMETRY_HPP
#include "triangle.hpp"
#include <array>
#include <vector>
namespace htool {

// 2D
template <typename CoordinatePrecision>
double triangle_jacobian(
    const std::array<CoordinatePrecision, 2> &A,
    const std::array<CoordinatePrecision, 2> &B,
    const std::array<CoordinatePrecision, 2> &C) {
    std::array<CoordinatePrecision, 3> e1, e2;
    for (int dim = 0; dim < 2; dim++) {
        e1[dim] = B[dim] - A[dim];
        e2[dim] = C[dim] - A[dim];
    }

    return std::abs(
        e1[0] * e2[1] - e1[1] * e2[0]);
}

// 3D
template <typename CoordinatePrecision>
double triangle_jacobian(
    const std::array<CoordinatePrecision, 3> &A,
    const std::array<CoordinatePrecision, 3> &B,
    const std::array<CoordinatePrecision, 3> &C) {
    std::array<CoordinatePrecision, 3> e1, e2;
    for (int dim = 0; dim < 3; dim++) {
        e1[dim] = B[dim] - A[dim];
        e2[dim] = C[dim] - A[dim];
    }

    std::array<CoordinatePrecision, 3> cross =
        {
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0]};

    return std::sqrt(
        cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
}

template <typename CoordinatePrecision, std::size_t dimension>
std::vector<std::array<CoordinatePrecision, dimension>> map_dunavant_to_triangle(
    const std::array<CoordinatePrecision, dimension> &A,
    const std::array<CoordinatePrecision, dimension> &B,
    const std::array<CoordinatePrecision, dimension> &C,
    const TriangleRule<CoordinatePrecision> &rule) {

    std::vector<std::array<CoordinatePrecision, dimension>> result;
    std::array<CoordinatePrecision, dimension> e1, e2;
    for (int dim = 0; dim < dimension; dim++) {
        e1[dim] = B[dim] - A[dim];
        e2[dim] = C[dim] - A[dim];
    }

    result.reserve(rule.nb_points);
    std::array<CoordinatePrecision, dimension> tmp;
    for (int qp = 0; qp < rule.nb_points; qp++) {
        for (int dim = 0; dim < dimension; dim++) {
            tmp[dim] = A[dim] + rule.quad_points[qp].point[0] * e1[dim] + rule.quad_points[qp].point[1] * e2[dim];
        }
        result.push_back(tmp);
    }

    return result;
}

} // namespace htool

#endif
