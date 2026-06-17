#include "htool/misc/misc.hpp"
#include "htool/quadratures/gauss_legendre.hpp"
#include "htool/quadratures/geometry.hpp"
#include <functional>
#include <htool/hmatrix/lrmat/interpolation.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp> // for LowRank...
#include <htool/matrix/utils.hpp>
#include <map>
#include <numeric>
#include <stdexcept>

using namespace htool;
using namespace std;

template <typename CoordinatePrecision, int dimension>
using Point = std::array<CoordinatePrecision, dimension>;

enum class KernelType { Constant,
                        Laplace,
                        Helmholtz };

enum class ElementType { P0,
                         P1 };

template <typename CoefficientPrecision, int dimension, KernelType kernel_type>
auto make_kernel() {
    using KernelFunction =
        std::function<void(std::array<double, dimension> *, int, std::array<double, dimension> *, int, CoefficientPrecision *)>;

    if constexpr (is_complex<CoefficientPrecision>() && kernel_type == KernelType::Helmholtz) {

        return KernelFunction(
            [](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, CoefficientPrecision *mat) {
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        double r = std::sqrt(
                            std::inner_product(
                                target_points[i].begin(), target_points[i].end(), source_points[j].begin(), 0.0, std::plus<double>(), [](double u, double v) {
                                    return (u - v) * (u - v);
                                }));

                        mat[i + j * Nx] = std::exp(std::complex<double>(0., 1.) * r) / (4. * M_PI * r);
                    }
                }
            });
    } else if constexpr (!is_complex<CoefficientPrecision>() && kernel_type == KernelType::Laplace) {

        return KernelFunction(
            [](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, CoefficientPrecision *mat) {
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {

                        double r = std::sqrt(
                            std::inner_product(
                                target_points[i].begin(), target_points[i].end(), source_points[j].begin(), 0.0, std::plus<double>(), [](double u, double v) {
                                    return (u - v) * (u - v);
                                }));

                        mat[i + j * Nx] = 1. / (4. * M_PI * r);
                    }
                }
            });
    } else if constexpr (kernel_type == KernelType::Constant) {
        return KernelFunction(
            [](std::array<double, dimension> *, int Nx, std::array<double, dimension> *, int Ny, CoefficientPrecision *mat) {
                for (int i = 0; i < Nx; ++i) {
                    for (int j = 0; j < Ny; ++j) {
                        mat[i + j * Nx] = 1;
                    }
                }
            });
    } else {
        static_assert(
            ((is_complex<CoefficientPrecision>() && kernel_type == KernelType::Helmholtz) || (!is_complex<CoefficientPrecision>() && kernel_type == KernelType::Laplace) || (kernel_type == KernelType::Constant)), "Kernel not implemented for this combination of CoefficientPrecision and KernelType");
    }
}

template <typename CoefficientPrecision,
          typename CoordinatePrecision,
          std::size_t dimension>
using basis_function_type = typename BEMHCA<CoefficientPrecision, CoordinatePrecision, dimension>::basis_function_type;

template <typename CoefficientPrecision, typename CoordinatePrecision, std::size_t dimension>
basis_function_type<CoefficientPrecision, CoordinatePrecision, dimension> make_p0_basis_on_triangle() {
    return [](
               int,
               int i,
               const std::vector<CoordinatePrecision> &)
               -> CoefficientPrecision {
        if (i != 0) {
            throw std::out_of_range(
                "P0 basis index must be 0");
        }

        return CoefficientPrecision(1);
    };
}

template <typename CoefficientPrecision, typename CoordinatePrecision, std::size_t dimension>
basis_function_type<CoefficientPrecision, CoordinatePrecision, dimension> make_p0_basis_on_segment() {
    return [](
               int,
               int i,
               const std::vector<CoordinatePrecision> &)
               -> CoefficientPrecision {
        if (i != 0) {
            throw std::out_of_range(
                "P0 basis index must be 0");
        }

        return CoefficientPrecision(1);
    };
}

template <typename CoefficientPrecision, typename CoordinatePrecision, std::size_t dimension>
basis_function_type<CoefficientPrecision, CoordinatePrecision, dimension> make_p1_basis_on_segment() {
    return [](
               int,
               int dof_local_index,
               const std::vector<CoordinatePrecision> &x)
               -> CoefficientPrecision {
        const CoordinatePrecision xi = x[0];

        switch (dof_local_index) {
        case 0:
            return CoefficientPrecision(0.5) * (CoefficientPrecision(1) - xi);

        case 1:
            return CoefficientPrecision(0.5) * (CoefficientPrecision(1) + xi);

        default:
            throw std::out_of_range(
                "P1 segment basis index must be 0 or 1");
        }
    };
}

template <typename CoefficientPrecision, typename CoordinatePrecision, std::size_t dimension>
basis_function_type<CoefficientPrecision, CoordinatePrecision, dimension> make_p1_basis_on_triangle() {
    return [](
               int,
               int dof_local_index,
               const std::vector<CoordinatePrecision> &x)
               -> CoefficientPrecision {
        const CoordinatePrecision xi  = x[0];
        const CoordinatePrecision eta = x[1];

        switch (dof_local_index) {
        case 0:
            return CoefficientPrecision(1) - xi - eta;

        case 1:
            return xi;

        case 2:
            return eta;

        default:
            throw std::out_of_range(
                "P1 segment basis index must be 0, 1 or 2");
        }
    };
}

template <typename CoefficientPrecision, int dimension>
Matrix<CoefficientPrecision> compute_reference_matrix_on_triangle(typename BEMHCA<CoefficientPrecision, double, dimension>::kernel_type kernel, const std::vector<double> &target_points, int target_element_index, const std::vector<int> &target_elements_to_points, int target_number_of_dofs_by_elements, basis_function_type<CoefficientPrecision, double, dimension> target_basis_function, const std::vector<double> &source_points, int source_element_index, const std::vector<int> &source_elements_to_points, int source_number_of_dofs_by_elements, basis_function_type<CoefficientPrecision, double, dimension> source_basis_function, int quadrature_order) {
    Matrix<CoefficientPrecision> mat_ref(target_number_of_dofs_by_elements, source_number_of_dofs_by_elements);
    std::vector<std::array<double, dimension>> local_target_quadrature_points;
    std::vector<std::array<double, dimension>> tmp_target_cell_points(3);
    std::vector<std::array<double, dimension>> local_source_quadrature_points;
    std::vector<std::array<double, dimension>> tmp_source_cell_points(3);
    for (int p = 0; p < 3; p++) {
        int point_index = target_elements_to_points[target_element_index * 3 + p];
        for (int dim = 0; dim < dimension; dim++) {
            tmp_target_cell_points[p][dim] = target_points[point_index * dimension + dim];
        }
    }
    for (int p = 0; p < 3; p++) {
        int point_index = source_elements_to_points[source_element_index * 3 + p];
        for (int dim = 0; dim < dimension; dim++) {
            tmp_source_cell_points[p][dim] = source_points[point_index * dimension + dim];
        }
    }
    double target_jacobian = 0;
    std::vector<std::vector<double>> target_quad_points(triangle_rules<double>[quadrature_order].nb_points);
    std::vector<double> target_weights(triangle_rules<double>[quadrature_order].nb_points);
    for (int i = 0; i < triangle_rules<double>[quadrature_order].nb_points; i++) {
        target_quad_points[i].resize(2);
        for (int d = 0; d < 2; d++) {
            target_quad_points[i][d] = triangle_rules<double>[quadrature_order].quad_points[i].point[d];
        }
        target_weights[i] = triangle_rules<double>[quadrature_order].quad_points[i].w;
    }
    double source_jacobian = 0;
    std::vector<std::vector<double>> source_quad_points(triangle_rules<double>[quadrature_order].nb_points);
    std::vector<double> source_weights(triangle_rules<CoefficientPrecision>[quadrature_order].nb_points);
    for (int i = 0; i < triangle_rules<CoefficientPrecision>[quadrature_order].nb_points; i++) {
        source_quad_points[i].resize(2);
        for (int d = 0; d < 2; d++) {
            source_quad_points[i][d] = triangle_rules<double>[quadrature_order].quad_points[i].point[d];
        }
        source_weights[i] = triangle_rules<double>[quadrature_order].quad_points[i].w;
    }
    target_jacobian                = triangle_jacobian(tmp_target_cell_points[0], tmp_target_cell_points[1], tmp_target_cell_points[2]);
    source_jacobian                = triangle_jacobian(tmp_source_cell_points[0], tmp_source_cell_points[1], tmp_source_cell_points[2]);
    local_target_quadrature_points = map_reference_to_triangle(tmp_target_cell_points[0], tmp_target_cell_points[1], tmp_target_cell_points[2], triangle_rules<double>[quadrature_order]);
    local_source_quadrature_points = map_reference_to_triangle(tmp_source_cell_points[0], tmp_source_cell_points[1], tmp_source_cell_points[2], triangle_rules<double>[quadrature_order]);

    for (int i = 0; i < target_number_of_dofs_by_elements; i++) {
        for (int j = 0; j < source_number_of_dofs_by_elements; j++) {
            for (int p = 0; p < local_target_quadrature_points.size(); p++) {
                for (int q = 0; q < local_source_quadrature_points.size(); q++) {
                    CoefficientPrecision kernel_eval;
                    kernel(&local_target_quadrature_points[p], 1, &local_source_quadrature_points[q], 1, &kernel_eval);
                    mat_ref(i, j) += target_jacobian * source_jacobian * target_weights[p] * source_weights[q] * kernel_eval * target_basis_function(target_element_index, i, target_quad_points[p]) * source_basis_function(source_element_index, j, source_quad_points[q]);
                }
            }
        }
    }
    return mat_ref;
}

template <typename CoefficientPrecision, int dimension>
Matrix<CoefficientPrecision> compute_reference_matrix_on_segment(typename BEMHCA<CoefficientPrecision, double, dimension>::kernel_type kernel, const std::vector<double> &target_points, int target_element_index, const std::vector<int> &target_elements_to_points, int target_number_of_dofs_by_elements, basis_function_type<CoefficientPrecision, double, dimension> target_basis_function, const std::vector<double> &source_points, int source_element_index, const std::vector<int> &source_elements_to_points, int source_number_of_dofs_by_elements, basis_function_type<CoefficientPrecision, double, dimension> source_basis_function, int quadrature_order) {
    Matrix<CoefficientPrecision> mat_ref(target_number_of_dofs_by_elements, source_number_of_dofs_by_elements);
    std::vector<std::array<double, dimension>> local_target_quadrature_points;
    std::vector<std::array<double, dimension>> tmp_target_cell_points(2);
    std::vector<std::array<double, dimension>> local_source_quadrature_points;
    std::vector<std::array<double, dimension>> tmp_source_cell_points(2);
    for (int p = 0; p < 2; p++) {
        int point_index = target_elements_to_points[target_element_index * 2 + p];
        for (int dim = 0; dim < dimension; dim++) {
            tmp_target_cell_points[p][dim] = target_points[point_index * dimension + dim];
        }
    }
    for (int p = 0; p < 2; p++) {
        int point_index = source_elements_to_points[source_element_index * 2 + p];
        for (int dim = 0; dim < dimension; dim++) {
            tmp_source_cell_points[p][dim] = source_points[point_index * dimension + dim];
        }
    }
    double target_jacobian = 0;
    std::vector<std::vector<double>> target_quad_points(gauss_legendre_rules<double>[quadrature_order].nb_points);
    std::vector<double> target_weights(gauss_legendre_rules<double>[quadrature_order].nb_points);
    for (int i = 0; i < gauss_legendre_rules<double>[quadrature_order].nb_points; i++) {
        target_quad_points[i].resize(1);
        for (int d = 0; d < 1; d++) {
            target_quad_points[i][d] = gauss_legendre_rules<double>[quadrature_order].quad_points[i].point[d];
        }
        target_weights[i] = gauss_legendre_rules<double>[quadrature_order].quad_points[i].w;
    }
    double source_jacobian = 0;
    std::vector<std::vector<double>> source_quad_points(gauss_legendre_rules<double>[quadrature_order].nb_points);
    std::vector<double> source_weights(gauss_legendre_rules<CoefficientPrecision>[quadrature_order].nb_points);
    for (int i = 0; i < gauss_legendre_rules<CoefficientPrecision>[quadrature_order].nb_points; i++) {
        source_quad_points[i].resize(1);
        for (int d = 0; d < 1; d++) {
            source_quad_points[i][d] = gauss_legendre_rules<double>[quadrature_order].quad_points[i].point[d];
        }
        source_weights[i] = gauss_legendre_rules<double>[quadrature_order].quad_points[i].w;
    }
    target_jacobian                = segment_jacobian(tmp_target_cell_points[0], tmp_target_cell_points[1]);
    source_jacobian                = segment_jacobian(tmp_source_cell_points[0], tmp_source_cell_points[1]);
    local_target_quadrature_points = map_reference_to_segment(tmp_target_cell_points[0], tmp_target_cell_points[1], gauss_legendre_rules<double>[quadrature_order]);
    local_source_quadrature_points = map_reference_to_segment(tmp_source_cell_points[0], tmp_source_cell_points[1], gauss_legendre_rules<double>[quadrature_order]);

    for (int i = 0; i < target_number_of_dofs_by_elements; i++) {
        for (int j = 0; j < source_number_of_dofs_by_elements; j++) {
            for (int p = 0; p < local_target_quadrature_points.size(); p++) {
                for (int q = 0; q < local_source_quadrature_points.size(); q++) {
                    CoefficientPrecision kernel_eval;
                    kernel(&local_target_quadrature_points[p], 1, &local_source_quadrature_points[q], 1, &kernel_eval);
                    mat_ref(i, j) += target_jacobian * source_jacobian * target_weights[p] * source_weights[q] * kernel_eval * target_basis_function(target_element_index, i, target_quad_points[p]) * source_basis_function(source_element_index, j, source_quad_points[q]);
                }
            }
        }
    }
    return mat_ref;
}

template <typename CoefficientPrecision, int dimension, KernelType kernel_type, ElementType element_type>
bool test_lrmat_build_BEMHCA_triangle(double epsilon, int quadrature_order, double tolerance) {

    auto kernel = make_kernel<CoefficientPrecision, dimension, kernel_type>();

    using basis_function_type = typename BEMHCA<CoefficientPrecision, htool::underlying_type<CoefficientPrecision>, dimension>::basis_function_type;

    basis_function_type target_basis_function;
    basis_function_type source_basis_function;

    int target_number_of_dofs;
    int source_number_of_dofs;
    if (element_type == ElementType::P0) {
        target_basis_function = make_p0_basis_on_triangle<CoefficientPrecision, double, dimension>();
        source_basis_function = make_p0_basis_on_triangle<CoefficientPrecision, double, dimension>();
        target_number_of_dofs = 2;
        source_number_of_dofs = 2;
    } else if (element_type == ElementType::P1) {
        target_basis_function = make_p1_basis_on_triangle<CoefficientPrecision, double, dimension>();
        source_basis_function = make_p1_basis_on_triangle<CoefficientPrecision, double, dimension>();
        target_number_of_dofs = 4;
        source_number_of_dofs = 4;
    }

    // -----------------------------------------------------------------------------
    // Unit square mesh with two triangles
    //
    // 2 ----- 3
    // | \     |
    // |   \   |
    // |     \ |
    // 0 ----- 1
    // -----------------------------------------------------------------------------
    std::map<int, std::vector<int>> target_dofs_to_elements;
    std::map<std::pair<int, int>, int> target_element_indices_to_dofs;
    int target_number_of_dofs_by_elements;
    if (element_type == ElementType::P0) {
        target_dofs_to_elements[0]             = {0, 0};
        target_dofs_to_elements[1]             = {1, 0};
        target_element_indices_to_dofs[{0, 0}] = 0;
        target_element_indices_to_dofs[{1, 0}] = 1;
        target_number_of_dofs_by_elements      = 1;
    } else if (element_type == ElementType::P1) {
        target_dofs_to_elements[0]             = {0, 0};
        target_dofs_to_elements[1]             = {0, 1, 1, 0};
        target_dofs_to_elements[2]             = {0, 2, 1, 2};
        target_dofs_to_elements[3]             = {1, 1};
        target_element_indices_to_dofs[{0, 0}] = 0;
        target_element_indices_to_dofs[{0, 1}] = 1;
        target_element_indices_to_dofs[{0, 2}] = 2;
        target_element_indices_to_dofs[{1, 0}] = 1;
        target_element_indices_to_dofs[{1, 1}] = 3;
        target_element_indices_to_dofs[{1, 2}] = 2;
        target_number_of_dofs_by_elements      = 3;
    }
    std::vector<int> target_elements_to_points = {0, 1, 2, 1, 3, 2};
    int target_number_of_elements              = 2;
    int target_number_of_points_per_element    = 3;
    std::vector<double> target_points;
    if constexpr (dimension == 3) {
        target_points = {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0};
    }
    if constexpr (dimension == 2) {
        target_points = {0, 0, 0, 1, 1, 0, 1, 1};
    }
    std::vector<int> target_permutation(4);
    std::iota(target_permutation.begin(), target_permutation.end(), 0);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 5;
    distance[1] = 6;
    distance[2] = 7;
    distance[3] = 8;

    bool is_error = 0;
    for (int idist = 0; idist < ndistance; idist++) {
        std::map<int, std::vector<int>> source_dofs_to_elements;
        std::map<std::pair<int, int>, int> source_element_indices_to_dofs;
        int source_number_of_dofs_by_elements;
        if (element_type == ElementType::P0) {
            source_dofs_to_elements[0]             = {0, 0};
            source_dofs_to_elements[1]             = {1, 0};
            source_element_indices_to_dofs[{0, 0}] = 0;
            source_element_indices_to_dofs[{1, 0}] = 1;
            source_number_of_dofs_by_elements      = 1;
        } else if (element_type == ElementType::P1) {
            source_dofs_to_elements[0]             = {0, 0};
            source_dofs_to_elements[1]             = {0, 1, 1, 0};
            source_dofs_to_elements[2]             = {0, 2, 1, 2};
            source_dofs_to_elements[3]             = {1, 1};
            source_element_indices_to_dofs[{0, 0}] = 0;
            source_element_indices_to_dofs[{0, 1}] = 1;
            source_element_indices_to_dofs[{0, 2}] = 2;
            source_element_indices_to_dofs[{1, 0}] = 1;
            source_element_indices_to_dofs[{1, 1}] = 3;
            source_element_indices_to_dofs[{1, 2}] = 2;
            source_number_of_dofs_by_elements      = 3;
        }
        std::vector<int> source_elements_to_points = {0, 1, 2, 1, 3, 2};
        int source_number_of_elements              = 2;
        int source_number_of_points_per_element    = 3;
        std::vector<double> source_points;
        source_points = target_points;
        if constexpr (dimension == 3) {
            source_points[2] += distance[idist];
            source_points[5] += distance[idist];
            source_points[8] += distance[idist];
            source_points[11] += distance[idist];
        } else if constexpr (dimension == 2) {
            source_points[1] += distance[idist];
            source_points[3] += distance[idist];
            source_points[5] += distance[idist];
            source_points[7] += distance[idist];
        }
        std::vector<int> source_permutation(4);
        std::iota(source_permutation.begin(), source_permutation.end(), 0);

        BEMHCA<CoefficientPrecision, double, dimension> compressor(kernel, target_basis_function, target_dofs_to_elements, target_number_of_dofs_by_elements, target_elements_to_points.data(), target_number_of_points_per_element, target_points.data(), target_points.size(), target_permutation.data(), quadrature_order, source_basis_function, source_dofs_to_elements, source_number_of_dofs_by_elements, source_elements_to_points.data(), source_number_of_points_per_element, source_points.data(), source_points.size(), source_permutation.data(), quadrature_order);
        compressor.check_size = false;
        LowRankMatrix<CoefficientPrecision> A(target_number_of_dofs, source_number_of_dofs, epsilon);
        compressor.copy_low_rank_approximation(target_number_of_dofs, source_number_of_dofs, 0, 0, A);

        // Compute reference
        Matrix<CoefficientPrecision> mat_ref(target_number_of_dofs, source_number_of_dofs);
        for (int target_element_index = 0; target_element_index < target_number_of_elements; target_element_index++) {
            for (int source_element_index = 0; source_element_index < source_number_of_elements; source_element_index++) {
                auto local_mat_ref = compute_reference_matrix_on_triangle<CoefficientPrecision, dimension>(kernel, target_points, target_element_index, target_elements_to_points, target_number_of_dofs_by_elements, target_basis_function, source_points, source_element_index, source_elements_to_points, source_number_of_dofs_by_elements, source_basis_function, quadrature_order);
                for (int target_local_index = 0; target_local_index < target_number_of_dofs_by_elements; target_local_index++) {
                    for (int source_local_index = 0; source_local_index < source_number_of_dofs_by_elements; source_local_index++) {
                        mat_ref(target_element_indices_to_dofs[{target_element_index, target_local_index}], source_element_indices_to_dofs[{source_element_index, source_local_index}]) += local_mat_ref(target_local_index, source_local_index);
                    }
                }
            }
        }

        // Compute error
        double approximation_error;
        Matrix<CoefficientPrecision> dense_lrmat(target_number_of_dofs, source_number_of_dofs);
        A.copy_to_dense(dense_lrmat.data());
        // print(dense_lrmat, std::cout, ",");
        // print(mat_ref, std::cout, ",");
        approximation_error = normFrob(mat_ref - dense_lrmat) / normFrob(mat_ref);
        cout << "approximation error : " << approximation_error << endl;
        is_error = is_error || !(approximation_error < epsilon * (1 + tolerance));
    }
    return is_error;
}

template <typename CoefficientPrecision, int dimension, KernelType kernel_type, ElementType element_type>
bool test_lrmat_build_BEMHCA_segment(double epsilon, int quadrature_order, double tolerance) {

    auto kernel = make_kernel<CoefficientPrecision, dimension, kernel_type>();

    using basis_function_type = typename BEMHCA<CoefficientPrecision, htool::underlying_type<CoefficientPrecision>, dimension>::basis_function_type;

    basis_function_type target_basis_function;
    basis_function_type source_basis_function;

    int target_number_of_dofs;
    int source_number_of_dofs;

    if (element_type == ElementType::P0) {
        target_basis_function = make_p0_basis_on_segment<CoefficientPrecision, double, dimension>();
        source_basis_function = make_p0_basis_on_segment<CoefficientPrecision, double, dimension>();
        target_number_of_dofs = 2;
        source_number_of_dofs = 2;
    } else if (element_type == ElementType::P1) {
        target_basis_function = make_p1_basis_on_segment<CoefficientPrecision, double, dimension>();
        source_basis_function = make_p1_basis_on_segment<CoefficientPrecision, double, dimension>();
        target_number_of_dofs = 3;
        source_number_of_dofs = 3;
    }

    std::map<int, std::vector<int>> target_dofs_to_elements;
    std::map<std::pair<int, int>, int> target_element_indices_to_dofs;
    int target_number_of_dofs_by_elements;
    if (element_type == ElementType::P0) {
        target_dofs_to_elements[0]             = {0, 0};
        target_dofs_to_elements[1]             = {1, 0};
        target_element_indices_to_dofs[{0, 0}] = 0;
        target_element_indices_to_dofs[{1, 0}] = 1;
        target_number_of_dofs_by_elements      = 1;
    } else if (element_type == ElementType::P1) {
        target_dofs_to_elements[0]             = {0, 0};
        target_dofs_to_elements[1]             = {0, 1, 1, 0};
        target_dofs_to_elements[2]             = {1, 1};
        target_element_indices_to_dofs[{0, 0}] = 0;
        target_element_indices_to_dofs[{0, 1}] = 1;
        target_element_indices_to_dofs[{1, 0}] = 1;
        target_element_indices_to_dofs[{1, 1}] = 2;
        target_number_of_dofs_by_elements      = 2;
    }
    std::vector<int> target_elements_to_points = {0, 1, 1, 2};
    int target_number_of_elements              = 2;
    int target_number_of_points_per_element    = 2;
    std::vector<double> target_points;
    if constexpr (dimension == 3) {
        target_points = {0, 0, 0, 0, 1, 0, 0, 2, 0};
    }
    if constexpr (dimension == 2) {
        target_points = {0, 0, 0, 1, 0, 2};
    }
    std::vector<int> target_permutation(3);
    std::iota(target_permutation.begin(), target_permutation.end(), 0);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 5;
    distance[1] = 6;
    distance[2] = 7;
    distance[3] = 8;

    bool is_error = 0;
    for (int idist = 0; idist < ndistance; idist++) {
        std::map<int, std::vector<int>> source_dofs_to_elements;
        std::map<std::pair<int, int>, int> source_element_indices_to_dofs;
        int source_number_of_dofs_by_elements;
        if (element_type == ElementType::P0) {
            source_dofs_to_elements[0]             = {0, 0};
            source_dofs_to_elements[1]             = {1, 0};
            source_element_indices_to_dofs[{0, 0}] = 0;
            source_element_indices_to_dofs[{1, 0}] = 1;
            source_number_of_dofs_by_elements      = 1;
        } else if (element_type == ElementType::P1) {
            source_dofs_to_elements[0]             = {0, 0};
            source_dofs_to_elements[1]             = {0, 1, 1, 0};
            source_dofs_to_elements[2]             = {1, 1};
            source_element_indices_to_dofs[{0, 0}] = 0;
            source_element_indices_to_dofs[{0, 1}] = 1;
            source_element_indices_to_dofs[{1, 0}] = 1;
            source_element_indices_to_dofs[{1, 1}] = 2;
            source_number_of_dofs_by_elements      = 2;
        }
        std::vector<int> source_elements_to_points{0, 1, 1, 2};
        int source_number_of_elements           = 2;
        int source_number_of_points_per_element = 2;
        std::vector<double> source_points;
        source_points = target_points;
        if constexpr (dimension == 3) {
            source_points[2] += distance[idist];
            source_points[5] += distance[idist];
            source_points[8] += distance[idist];
        } else if constexpr (dimension == 2) {
            source_points[1] += distance[idist];
            source_points[3] += distance[idist];
            source_points[5] += distance[idist];
        }
        std::vector<int> source_permutation(3);
        std::iota(source_permutation.begin(), source_permutation.end(), 0);

        BEMHCA<CoefficientPrecision, double, dimension> compressor(kernel, target_basis_function, target_dofs_to_elements, target_number_of_dofs_by_elements, target_elements_to_points.data(), target_number_of_points_per_element, target_points.data(), target_points.size(), target_permutation.data(), quadrature_order, source_basis_function, source_dofs_to_elements, source_number_of_dofs_by_elements, source_elements_to_points.data(), source_number_of_points_per_element, source_points.data(), source_points.size(), source_permutation.data(), quadrature_order);
        compressor.check_size = false;
        LowRankMatrix<CoefficientPrecision> A(target_number_of_dofs, source_number_of_dofs, epsilon);
        compressor.copy_low_rank_approximation(target_number_of_dofs, source_number_of_dofs, 0, 0, A);

        // Compute reference
        Matrix<CoefficientPrecision> mat_ref(target_number_of_dofs, source_number_of_dofs);
        for (int target_element_index = 0; target_element_index < target_number_of_elements; target_element_index++) {
            for (int source_element_index = 0; source_element_index < source_number_of_elements; source_element_index++) {
                auto local_mat_ref = compute_reference_matrix_on_segment<CoefficientPrecision, dimension>(kernel, target_points, target_element_index, target_elements_to_points, target_number_of_dofs_by_elements, target_basis_function, source_points, source_element_index, source_elements_to_points, source_number_of_dofs_by_elements, source_basis_function, quadrature_order);
                for (int target_local_index = 0; target_local_index < target_number_of_dofs_by_elements; target_local_index++) {
                    for (int source_local_index = 0; source_local_index < source_number_of_dofs_by_elements; source_local_index++) {
                        mat_ref(target_element_indices_to_dofs[{target_element_index, target_local_index}], source_element_indices_to_dofs[{source_element_index, source_local_index}]) += local_mat_ref(target_local_index, source_local_index);
                    }
                }
            }
        }

        // Compute error
        double approximation_error;
        Matrix<CoefficientPrecision> dense_lrmat(target_number_of_dofs, source_number_of_dofs);
        A.copy_to_dense(dense_lrmat.data());
        // print(dense_lrmat, std::cout, ",");
        // print(mat_ref, std::cout, ",");
        approximation_error = normFrob(mat_ref - dense_lrmat) / normFrob(mat_ref);
        cout << "approximation error : " << approximation_error << endl;
        is_error = is_error || !(approximation_error < epsilon * (1 + tolerance));
    }
    return is_error;
}
