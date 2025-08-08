#include <execution>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg.hpp>
#include <htool/hmatrix/utility.hpp>
#include <htool/matrix/matrix_view.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

class UserOperator : public VirtualGenerator<double> {
    const vector<double> &m_target_coordinates;
    const vector<double> &m_source_coordinates;
    int m_spatial_dimension;
    int m_nr;
    int m_nc;

  public:
    // Constructor
    UserOperator(int space_dim, const vector<double> &target_coordinates, const vector<double> &source_coordinates) : m_target_coordinates(target_coordinates), m_source_coordinates(source_coordinates), m_spatial_dimension(space_dim), m_nr(target_coordinates.size() / m_spatial_dimension), m_nc(source_coordinates.size() / m_spatial_dimension) {}

    // Virtual function to overload, necessary
    void copy_submatrix(int M, int N, const int *rows, const int *cols, double *ptr) const override {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                ptr[j + M * k] = this->get_coef(rows[j], cols[k]);
            }
        }
    }

    double get_coef(const int &k, const int &j) const {
        return 1. / (1e-5 + std::sqrt(std::inner_product(m_target_coordinates.begin() + m_spatial_dimension * k, m_target_coordinates.begin() + m_spatial_dimension * k + m_spatial_dimension, m_source_coordinates.begin() + m_spatial_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    // Matrix vector product, not necessary
    std::vector<double> operator*(std::vector<double> a) {
        std::vector<double> result(m_nr, 0);
        for (int j = 0; j < m_nr; j++) {
            for (int k = 0; k < m_nc; k++) {
                result[j] += this->get_coef(j, k) * a[k];
            }
        }
        return result;
    }

    // Frobenius norm, not necessary
    double norm() {
        double norm = 0;
        for (int j = 0; j < m_nr; j++) {
            for (int k = 0; k < m_nc; k++) {
                norm += this->get_coef(j, k);
            }
        }
        return norm;
    }
};

int main(int argc, char *argv[]) {

    // Check the number of parameters
    if (argc < 1) {
        // Tell the user how to run the program
        cerr << "Usage: " << argv[0] << " outputpath" << endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }

    std::string outputpath = argv[1];

    // Execution policy
    auto &policy = exec_compat::par;

    // Geometry
    const int number_points     = 10000;
    const int spatial_dimension = 3;
    vector<double> coordinates(spatial_dimension * number_points);
    create_sphere(number_points, coordinates.data());

    // Cluster tree builder
    ClusterTreeBuilder<double> recursive_build_strategy;
    recursive_build_strategy.set_maximal_leaf_size(500);

    // HMatrix parameters
    const double epsilon = 0.01;
    const double eta     = 10;
    char symmetry        = 'S';
    char UPLO            = 'L';

    // Generator
    UserOperator A(spatial_dimension, coordinates, coordinates);

    // HMatrix
    HMatrixBuilder<double> hmatrix_builder(number_points, spatial_dimension, coordinates.data(), &recursive_build_strategy);
    HMatrix<double> hmatrix = hmatrix_builder.build(policy, A, htool::HMatrixTreeBuilder<double>(epsilon, eta, symmetry, UPLO));

    // Output
    // save_leaves_with_rank(hmatrix, outputpath + "hmatrix");
    print_tree_parameters(hmatrix, std::cout);
    print_hmatrix_information(hmatrix, std::cout);

    // sequential y= A*x
    std::vector<double> x(number_points, 1), y(number_points, 0), ref(number_points, 0);
    add_hmatrix_vector_product(policy, 'N', double(1), hmatrix, x.data(), double(0), y.data());
    ref = A * x;
    std::cout << "relative error on matrix vector product : ";
    std::cout << norm2(ref - y) / norm2(ref) << "\n";

    // sequential z = A^-1 y
    std::vector<double> z(number_points, 0);
    z = y;
    MatrixView<double> z_view(number_points, 1, z.data());
    cholesky_factorization(policy, UPLO, hmatrix);
    cholesky_solve(UPLO, hmatrix, z_view);
    std::cout << "relative error on cholesky solve : ";
    std::cout << norm2(z - x) / norm2(x) << "\n";
}
