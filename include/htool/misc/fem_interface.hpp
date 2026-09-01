#ifndef HTOOL_FEM_INTERFACE_HPP
#define HTOOL_FEM_INTERFACE_HPP

#include <cstddef>
#include <functional>
#include <map>
#include <vector>

namespace htool {

/**
 * @brief Non-owning description of a finite element space over one side (target or source) of a
 * boundary element problem: mesh connectivity, DOF layout, basis functions and quadrature order.
 * @tparam CoefficientPrecision Scalar type of the basis functions.
 * @tparam CoordinatePrecision Scalar type of the mesh point coordinates.
 * @tparam dimension Ambient dimension of the mesh geometry (2 or 3).
 */
template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
struct FEMSpace {
    /// @brief basis_function(element_index, local_dof_index, reference_coordinates) evaluates a local shape function.
    typedef std::function<CoefficientPrecision(int, int, const std::vector<CoordinatePrecision> &)> basis_function_type;

    basis_function_type basis_function;                      ///< Local shape functions, see basis_function_type.
    const std::map<int, std::vector<int>> *dofs_to_elements; ///< Global DOF index -> flattened (element_index, local_dof_index) pairs.
    int number_of_dofs_per_element;                           ///< Number of local DOFs per element (e.g. 1 for P0, independent of number_of_points_per_element).
    const int *elements_to_points;                            ///< Element -> geometric point indices, number_of_points_per_element entries per element.
    int number_of_points_per_element;                         ///< 2 for a segment, 3 for a triangle; no other value is supported.
    const CoordinatePrecision *points;                        ///< Flat point coordinates, dimension entries per point.
    std::size_t size;                                         ///< Length of points.
    const int *permutation;                                   ///< Local DOF index -> global DOF index, i.e. the key used into dofs_to_elements.
    int quadrature_order;                                     ///< Requested quadrature order.
};

} // namespace htool

#endif
