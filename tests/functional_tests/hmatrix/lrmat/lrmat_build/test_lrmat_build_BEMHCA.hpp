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

template <typename CoordinatePrecision, typename CoefficientPrecision, int dimension>
using basis_function_type =
    const std::function<
        CoefficientPrecision(
            std::vector<Point<CoordinatePrecision, dimension>>,
            int,
            Point<CoordinatePrecision, dimension>)>;

template <
    typename CoordinatePrecision,
    typename CoefficientPrecision,
    std::size_t dimension>
basis_function_type<CoordinatePrecision, CoefficientPrecision, dimension> make_p0_basis_on_triangle() {
    return [](
               std::vector<std::array<CoordinatePrecision, dimension>> /*vertices*/,
               int i,
               std::array<CoordinatePrecision, dimension> /*x*/)
               -> CoefficientPrecision {
        if (i != 0) {
            throw std::out_of_range(
                "P0 basis index must be 0");
        }

        return CoefficientPrecision(1);
    };
}

template <typename CoordinatePrecision,
          typename CoefficientPrecision,
          int dimension>
basis_function_type<CoordinatePrecision,
                    CoefficientPrecision,
                    dimension>
make_p1_basis_on_segment() {

    return [](
               const std::vector<
                   std::array<CoordinatePrecision, dimension>> &vertices,
               int i,
               const std::array<CoordinatePrecision, dimension> &x)
               -> CoefficientPrecision {
        static_assert(
            dimension >= 1,
            "segment must live in dimension >= 1");

        const auto &A = vertices[0];
        const auto &B = vertices[1];
        auto dot      = [](const auto &u, const auto &v) {
            CoordinatePrecision result = 0;

            for (std::size_t d = 0; d < dimension; ++d)
                result += u[d] * v[d];

            return result;
        };

        auto sub = [](const auto &u, const auto &v) {
            std::array<CoordinatePrecision, dimension> out{};

            for (std::size_t d = 0; d < dimension; ++d)
                out[d] = u[d] - v[d];

            return out;
        };

        const auto e = sub(B, A);
        const auto r = sub(x, A);

        // Solve:
        // x = A + xi * e

        const auto ee = dot(e, e);
        const auto re = dot(r, e);

        const auto xi = re / ee;

        const auto lambda0 =
            CoefficientPrecision(1) - CoefficientPrecision(xi);

        const auto lambda1 =
            CoefficientPrecision(xi);

        switch (i) {
        case 0:
            return lambda0;
        case 1:
            return lambda1;
        default:
            throw std::out_of_range(
                "P1 segment basis index must be 0 or 1");
        }
    };
}
template <typename CoordinatePrecision, typename CoefficientPrecision, int dimension>
basis_function_type<CoordinatePrecision, CoefficientPrecision, dimension> make_p1_basis_on_triangle() {
    return [](
               const std::vector<std::array<CoordinatePrecision, dimension>> &vertices,
               int i,
               const std::array<CoordinatePrecision, dimension> &x)
               -> CoefficientPrecision {
        static_assert(
            dimension == 2 || dimension == 3,
            "Triangle must live in 2D or 3D");

        const auto &A = vertices[0];
        const auto &B = vertices[1];
        const auto &C = vertices[2];

        auto dot = [](const auto &u, const auto &v) {
            CoordinatePrecision result = 0;

            for (std::size_t d = 0; d < dimension; ++d)
                result += u[d] * v[d];

            return result;
        };

        auto sub = [](const auto &u, const auto &v) {
            std::array<CoordinatePrecision, dimension> out{};

            for (std::size_t d = 0; d < dimension; ++d)
                out[d] = u[d] - v[d];

            return out;
        };

        const auto e1 = sub(B, A);
        const auto e2 = sub(C, A);
        const auto r  = sub(x, A);

        // Solve:
        // x = A + xi e1 + eta e2

        const auto g11 = dot(e1, e1);
        const auto g12 = dot(e1, e2);
        const auto g22 = dot(e2, e2);

        const auto b1 = dot(r, e1);
        const auto b2 = dot(r, e2);

        const auto det = g11 * g22 - g12 * g12;

        const auto xi =
            (g22 * b1 - g12 * b2) / det;

        const auto eta =
            (g11 * b2 - g12 * b1) / det;

        const auto lambda0 =
            CoefficientPrecision(1) - xi - eta;

        const auto lambda1 =
            CoefficientPrecision(xi);

        const auto lambda2 =
            CoefficientPrecision(eta);

        switch (i) {
        case 0:
            return lambda0;
        case 1:
            return lambda1;
        case 2:
            return lambda2;
        default:
            throw std::out_of_range(
                "P1 basis index must be 0,1,2");
        }
    };
}

template <typename CoefficientPrecision, int dimension>
Matrix<CoefficientPrecision> compute_reference_matrix_on_triangle(typename BEMHCA<CoefficientPrecision, double, dimension>::kernel_type kernel, const std::vector<double> &target_points, int target_element_index, const std::vector<int> &target_elements_to_points, basis_function_type<double, CoefficientPrecision, dimension> target_basis_function, const std::vector<double> &source_points, int source_element_index, const std::vector<int> &source_elements_to_points, basis_function_type<double, CoefficientPrecision, dimension> source_basis_function, int quadrature_order) {
    Matrix<CoefficientPrecision> mat_ref(3, 3);
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
    std::vector<double> target_weights(triangle_rules<double>[quadrature_order].nb_points);
    for (int i = 0; i < triangle_rules<double>[quadrature_order].nb_points; i++) {
        target_weights[i] = triangle_rules<double>[quadrature_order].quad_points[i].w;
    }
    double source_jacobian = 0;
    std::vector<double> source_weights(triangle_rules<CoefficientPrecision>[quadrature_order].nb_points);
    for (int i = 0; i < triangle_rules<CoefficientPrecision>[quadrature_order].nb_points; i++) {
        source_weights[i] = triangle_rules<double>[quadrature_order].quad_points[i].w;
    }
    target_jacobian                = triangle_jacobian(tmp_target_cell_points[0], tmp_target_cell_points[1], tmp_target_cell_points[2]);
    source_jacobian                = triangle_jacobian(tmp_source_cell_points[0], tmp_source_cell_points[1], tmp_source_cell_points[2]);
    local_target_quadrature_points = map_reference_to_triangle(tmp_target_cell_points[0], tmp_target_cell_points[1], tmp_target_cell_points[2], triangle_rules<double>[quadrature_order]);
    local_source_quadrature_points = map_reference_to_triangle(tmp_source_cell_points[0], tmp_source_cell_points[1], tmp_source_cell_points[2], triangle_rules<double>[quadrature_order]);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int p = 0; p < local_target_quadrature_points.size(); p++) {
                for (int q = 0; q < local_source_quadrature_points.size(); q++) {
                    CoefficientPrecision kernel_eval;
                    kernel(&local_target_quadrature_points[p], 1, &local_source_quadrature_points[q], 1, &kernel_eval);
                    mat_ref(i, j) += target_jacobian * source_jacobian * target_weights[p] * source_weights[q] * kernel_eval * target_basis_function(tmp_target_cell_points, i, local_target_quadrature_points[p]) * source_basis_function(tmp_source_cell_points, j, local_source_quadrature_points[q]);
                }
            }
        }
    }
    return mat_ref;
}

template <typename CoefficientPrecision, int dimension>
Matrix<CoefficientPrecision> compute_reference_matrix_on_segment(typename BEMHCA<CoefficientPrecision, double, dimension>::kernel_type kernel, const std::vector<double> &target_points, int target_element_index, const std::vector<int> &target_elements_to_points, basis_function_type<double, CoefficientPrecision, dimension> target_basis_function, const std::vector<double> &source_points, int source_element_index, const std::vector<int> &source_elements_to_points, basis_function_type<double, CoefficientPrecision, dimension> source_basis_function, int quadrature_order) {
    Matrix<CoefficientPrecision> mat_ref(2, 2);
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
    std::vector<double> target_weights(gauss_legendre_rules<double>[quadrature_order].nb_points);
    for (int i = 0; i < gauss_legendre_rules<double>[quadrature_order].nb_points; i++) {
        target_weights[i] = gauss_legendre_rules<double>[quadrature_order].quad_points[i].w;
    }
    double source_jacobian = 0;
    std::vector<double> source_weights(gauss_legendre_rules<CoefficientPrecision>[quadrature_order].nb_points);
    for (int i = 0; i < gauss_legendre_rules<CoefficientPrecision>[quadrature_order].nb_points; i++) {
        source_weights[i] = gauss_legendre_rules<double>[quadrature_order].quad_points[i].w;
    }
    target_jacobian                = segment_jacobian(tmp_target_cell_points[0], tmp_target_cell_points[1]);
    source_jacobian                = segment_jacobian(tmp_source_cell_points[0], tmp_source_cell_points[1]);
    local_target_quadrature_points = map_reference_to_segment(tmp_target_cell_points[0], tmp_target_cell_points[1], gauss_legendre_rules<double>[quadrature_order]);
    local_source_quadrature_points = map_reference_to_segment(tmp_source_cell_points[0], tmp_source_cell_points[1], gauss_legendre_rules<double>[quadrature_order]);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int p = 0; p < local_target_quadrature_points.size(); p++) {
                for (int q = 0; q < local_source_quadrature_points.size(); q++) {
                    CoefficientPrecision kernel_eval;
                    kernel(&local_target_quadrature_points[p], 1, &local_source_quadrature_points[q], 1, &kernel_eval);
                    mat_ref(i, j) += target_jacobian * source_jacobian * target_weights[p] * source_weights[q] * kernel_eval * target_basis_function(tmp_target_cell_points, i, local_target_quadrature_points[p]) * source_basis_function(tmp_source_cell_points, j, local_source_quadrature_points[q]);
                }
            }
        }
    }
    return mat_ref;
}

template <typename CoefficientPrecision, int dimension>
bool test_lrmat_build_BEMHCA_triangle(double epsilon, int quadrature_order, double tolerance) {

    auto kernel = std::function([](std::array<double, dimension> *, int, std::array<double, dimension> *, int, CoefficientPrecision *) -> void {});
    if constexpr (is_complex<CoefficientPrecision>()) {
        kernel = std::function([](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, std::complex<double> *mat) {
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    mat[i + j * Nx] = std::exp(std::complex<double>(0, 1) * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))) / (4 * M_PI * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
                }
            }
        });
    } else {
        kernel = std::function([](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, double *mat) {
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    mat[i + j * Nx] = 1. / (4 * M_PI * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
                }
            }
        });
    }

    auto target_basis_function = make_p1_basis_on_triangle<double, CoefficientPrecision, dimension>();
    auto source_basis_function = make_p1_basis_on_triangle<double, CoefficientPrecision, dimension>();

    // -----------------------------------------------------------------------------
    // Unit square mesh with two triangles
    //
    // 2 ----- 3
    // | \     |
    // |   \   |
    // |     \ |
    // 0 ----- 1
    // -----------------------------------------------------------------------------
    std::vector<double> target_points;
    std::vector<int> target_elements_to_points; // also to dofs because P1
    std::map<int, std::vector<int>> target_dofs_to_elements;
    std::vector<int> target_permutation;
    int target_number_of_dofs_by_elements;
    target_elements_to_points         = {0, 1, 2, 1, 3, 2};
    target_dofs_to_elements[0]        = {0, 0};
    target_dofs_to_elements[1]        = {0, 1, 1, 0};
    target_dofs_to_elements[2]        = {0, 2, 1, 2};
    target_dofs_to_elements[3]        = {1, 1};
    target_number_of_dofs_by_elements = 3;
    target_permutation.resize(4);
    if constexpr (dimension == 3) {
        target_points = {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0};
    }
    if constexpr (dimension == 2) {
        target_points = {0, 0, 0, 1, 1, 0, 1, 1};
    }
    std::iota(target_permutation.begin(), target_permutation.end(), 0);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    bool is_error = 0;
    for (int idist = 0; idist < ndistance; idist++) {
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

        std::vector<int> source_elements_to_points = {0, 1, 2, 1, 3, 2};
        std::map<int, std::vector<int>> source_dofs_to_elements;
        source_dofs_to_elements[0]            = {0, 0};
        source_dofs_to_elements[1]            = {0, 1, 1, 0};
        source_dofs_to_elements[2]            = {0, 2, 1, 2};
        source_dofs_to_elements[3]            = {1, 1};
        int source_number_of_dofs_by_elements = 3;
        std::vector<int> source_permutation(4);
        std::iota(source_permutation.begin(), source_permutation.end(), 0);

        BEMHCA<CoefficientPrecision, double, dimension> compressor(kernel, target_basis_function, target_dofs_to_elements, target_number_of_dofs_by_elements, target_elements_to_points.data(), 3, target_points.data(), target_points.size(), target_permutation.data(), quadrature_order, source_basis_function, source_dofs_to_elements, source_number_of_dofs_by_elements, source_elements_to_points.data(), 3, source_points.data(), source_points.size(), source_permutation.data(), quadrature_order);

        LowRankMatrix<CoefficientPrecision> A(4, 4, epsilon);
        compressor.copy_low_rank_approximation(4, 4, 0, 0, A);

        // Compute reference
        Matrix<CoefficientPrecision> mat_ref(4, 4);
        for (int target_element_index = 0; target_element_index < 2; target_element_index++) {
            for (int source_element_index = 0; source_element_index < 2; source_element_index++) {
                auto local_mat_ref = compute_reference_matrix_on_triangle<CoefficientPrecision, dimension>(kernel, target_points, target_element_index, target_elements_to_points, target_basis_function, source_points, source_element_index, source_elements_to_points, source_basis_function, quadrature_order);
                for (int target_local_index = 0; target_local_index < 3; target_local_index++) {
                    for (int source_local_index = 0; source_local_index < 3; source_local_index++) {
                        mat_ref(target_elements_to_points[target_element_index * 3 + target_local_index], source_elements_to_points[source_element_index * 3 + source_local_index]) += local_mat_ref(target_local_index, source_local_index);
                    }
                }
            }
        }

        // Compute error
        double approximation_error;
        Matrix<CoefficientPrecision> dense_lrmat(4, 4);
        A.copy_to_dense(dense_lrmat.data());
        approximation_error = normFrob(mat_ref - dense_lrmat) / normFrob(mat_ref);
        cout << "approximation error : " << approximation_error << endl;
        is_error = is_error || !(approximation_error < epsilon * (1 + tolerance));
    }
    return is_error;
}

template <typename CoefficientPrecision, int dimension>
bool test_lrmat_build_BEMHCA_segment(double epsilon, int quadrature_order, double tolerance) {

    auto kernel = std::function([](std::array<double, dimension> *, int, std::array<double, dimension> *, int, CoefficientPrecision *) -> void {});
    if constexpr (is_complex<CoefficientPrecision>()) {
        kernel = std::function([](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, std::complex<double> *mat) {
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    mat[i + j * Nx] = std::exp(std::complex<double>(0, 1) * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))) / (4 * M_PI * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
                }
            }
        });
    } else {
        kernel = std::function([](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, double *mat) {
            for (int i = 0; i < Nx; i++) {
                for (int j = 0; j < Ny; j++) {
                    mat[i + j * Nx] = 1. / (4 * M_PI * std::sqrt(std::inner_product(target_points[i].begin(), target_points[i].end(), source_points[j].begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
                }
            }
        });
    }

    auto target_basis_function = make_p1_basis_on_segment<double, CoefficientPrecision, dimension>();
    auto source_basis_function = make_p1_basis_on_segment<double, CoefficientPrecision, dimension>();

    std::vector<double> target_points;
    std::vector<int> target_elements_to_points;
    std::map<int, std::vector<int>> target_dofs_to_elements;
    std::vector<int> target_permutation;
    int target_number_of_dofs_by_elements;
    target_elements_to_points         = {0, 1, 1, 2};
    target_dofs_to_elements[0]        = {0, 0};
    target_dofs_to_elements[1]        = {0, 1, 1, 0};
    target_dofs_to_elements[2]        = {1, 1};
    target_number_of_dofs_by_elements = 2;
    target_permutation.resize(3);
    if constexpr (dimension == 3) {
        target_points = {0, 0, 0, 0, 1, 0, 0, 2, 0};
    }
    if constexpr (dimension == 2) {
        target_points = {0, 0, 0, 1, 0, 2};
    }
    std::iota(target_permutation.begin(), target_permutation.end(), 0);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    bool is_error = 0;
    for (int idist = 0; idist < ndistance; idist++) {
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

        std::vector<int> source_elements_to_points{0, 1, 1, 2};
        std::map<int, std::vector<int>> source_dofs_to_elements;
        source_dofs_to_elements[0]            = {0, 0};
        source_dofs_to_elements[1]            = {0, 1, 1, 0};
        source_dofs_to_elements[2]            = {1, 1};
        int source_number_of_dofs_by_elements = 2;
        std::vector<int> source_permutation(3);
        std::iota(source_permutation.begin(), source_permutation.end(), 0);

        BEMHCA<CoefficientPrecision, double, dimension> compressor(kernel, target_basis_function, target_dofs_to_elements, target_number_of_dofs_by_elements, target_elements_to_points.data(), 2, target_points.data(), target_points.size(), target_permutation.data(), quadrature_order, source_basis_function, source_dofs_to_elements, source_number_of_dofs_by_elements, source_elements_to_points.data(), 2, source_points.data(), source_points.size(), source_permutation.data(), quadrature_order);

        LowRankMatrix<CoefficientPrecision> A(3, 3, epsilon);
        compressor.copy_low_rank_approximation(3, 3, 0, 0, A);

        // Compute reference
        Matrix<CoefficientPrecision> mat_ref(3, 3);
        for (int target_element_index = 0; target_element_index < 2; target_element_index++) {
            for (int source_element_index = 0; source_element_index < 2; source_element_index++) {
                auto local_mat_ref = compute_reference_matrix_on_segment<CoefficientPrecision, dimension>(kernel, target_points, target_element_index, target_elements_to_points, target_basis_function, source_points, source_element_index, source_elements_to_points, source_basis_function, quadrature_order);
                for (int target_local_index = 0; target_local_index < 2; target_local_index++) {
                    for (int source_local_index = 0; source_local_index < 2; source_local_index++) {
                        mat_ref(target_elements_to_points[target_element_index * 2 + target_local_index], source_elements_to_points[source_element_index * 2 + source_local_index]) += local_mat_ref(target_local_index, source_local_index);
                    }
                }
            }
        }

        // Compute error
        double approximation_error;
        Matrix<CoefficientPrecision> dense_lrmat(3, 3);
        A.copy_to_dense(dense_lrmat.data());
        approximation_error = normFrob(mat_ref - dense_lrmat) / normFrob(mat_ref);
        cout << "approximation error : " << approximation_error << endl;
        is_error = is_error || !(approximation_error < epsilon * (1 + tolerance));
    }
    return is_error;
}
