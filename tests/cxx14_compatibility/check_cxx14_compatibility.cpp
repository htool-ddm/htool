// Built with CXX_STANDARD 14 to check that the public headers stay C++14-compatible (FreeFEM builds htool under C++14).
// Self-contained on purpose: no htool/testing header, and no C++17-only syntax here.
#include <cmath>
#include <htool/hmatrix/utility.hpp>
#include <htool/htool.hpp>
#include <vector>

using namespace htool;

class Generator : public VirtualGenerator<double> {
    const std::vector<double> &m_coordinates;

  public:
    explicit Generator(const std::vector<double> &coordinates) : m_coordinates(coordinates) {}

    void copy_submatrix(int M, int N, const int *rows, const int *cols, double *ptr) const override {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < M; i++) {
                double dx      = m_coordinates[3 * rows[i]] - m_coordinates[3 * cols[j]];
                double dy      = m_coordinates[3 * rows[i] + 1] - m_coordinates[3 * cols[j] + 1];
                double dz      = m_coordinates[3 * rows[i] + 2] - m_coordinates[3 * cols[j] + 2];
                ptr[i + M * j] = std::exp(-std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
    }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int size_world;
    MPI_Comm_size(MPI_COMM_WORLD, &size_world);

    const int number_of_points = 500;
    std::vector<double> coordinates(3 * number_of_points);
    for (int i = 0; i < number_of_points; i++) {
        double t               = 0.05 * i;
        coordinates[3 * i]     = std::cos(t);
        coordinates[3 * i + 1] = std::sin(t);
        coordinates[3 * i + 2] = 0.01 * i;
    }
    Generator generator(coordinates);
    std::vector<double> x(number_of_points, 1), y(number_of_points, 0);

    // Sequential HMatrix: LU and Cholesky
    HMatrixBuilder<double> hmatrix_builder(number_of_points, 3, coordinates.data());

    HMatrix<double> hmatrix = hmatrix_builder.build(generator, HMatrixTreeBuilder<double>(1e-6, 10, 'N', 'N'));
    add_hmatrix_vector_product('N', 1., hmatrix, x.data(), 0., y.data());
    MatrixView<double> y_view(number_of_points, 1, y.data());
    lu_factorization(hmatrix);
    lu_solve('N', hmatrix, y_view);

    HMatrix<double> sym_hmatrix = hmatrix_builder.build(generator, HMatrixTreeBuilder<double>(1e-6, 10, 'S', 'L'));
    std::vector<double> z(number_of_points, 1);
    MatrixView<double> z_view(number_of_points, 1, z.data());
    cholesky_factorization('L', sym_hmatrix);
    cholesky_solve('L', sym_hmatrix, z_view);

    // Distributed operator
    Cluster<double> cluster = ClusterTreeBuilder<double>().create_cluster_tree(number_of_points, 3, coordinates.data(), 2, size_world);
    DefaultApproximationBuilder<double> approximation_builder(generator, cluster, cluster, HMatrixTreeBuilder<double>(1e-6, 10, 'S', 'U'), MPI_COMM_WORLD);
    DistributedOperator<double> &distributed_operator = approximation_builder.distributed_operator;
    double *work                                      = nullptr;
    add_distributed_operator_vector_product_global_to_global('N', 1., distributed_operator, x.data(), 0., y.data(), work);

#ifdef HTOOL_WITH_HPDDM
    // DDM solver (block Jacobi, local LDLt since is_positive_definite defaults to false)
    HMatrix<double> local_hmatrix = *approximation_builder.block_diagonal_hmatrix;
    DDMSolverBuilder<double> ddm_solver_builder(distributed_operator, local_hmatrix);
    ddm_solver_builder.solver.facto_one_level();
    std::vector<double> solution(number_of_points, 0);
    ddm_solver_builder.solver.solve(y.data(), solution.data(), 1);
#endif

    MPI_Finalize();
    return 0;
}
