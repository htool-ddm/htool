#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/utility.hpp>  // for HMatrixBuilder
#include <htool/testing/geometry.hpp> // for create_...
#include <vector>                     // for vector

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestTypeInUserNumbering>
bool test_hmatrix_builder(int nr, int nc, char Symmetry, char UPLO, double epsilon) {
    bool is_error = false;

    // Geometry
    double z1 = 1;
    vector<double> p1(3 * nr), p1_permuted, off_diagonal_p1;
    vector<double> p2(Symmetry == 'N' ? 3 * nc : 1), p2_permuted, off_diagonal_p2;
    create_disk(3, z1, nr, p1.data());

    // HMatrix builder
    std::unique_ptr<HMatrixBuilder<T>> hmatrix_builder;
    if (Symmetry == 'N') {
        hmatrix_builder = std::make_unique<HMatrixBuilder<T>>(nr, 3, p1.data(), nc, 3, p2.data());

    } else {
        hmatrix_builder = std::make_unique<HMatrixBuilder<T>>(nr, 3, p1.data());
    }

    // HMatrixTreeBuilder
    double eta = 10;
    HMatrixTreeBuilder<T> hmatrix_tree_builder(epsilon, eta, Symmetry, UPLO);

    // Generator
    GeneratorTestTypeInUserNumbering generator(3, p1, (Symmetry == 'N' ? p2 : p1));

    // HMatrix
    HMatrix<T> hmatrix = hmatrix_builder->build(generator, hmatrix_tree_builder);

    // Densified hmatrix
    Matrix<T> dense_hmatrix(nr, nc);
    copy_to_dense(hmatrix, dense_hmatrix.data());

    // Test
    std::vector<T> ones(nc, 1), result(nr, 0), reference(nr, 0);
    sequential_internal_add_hmatrix_vector_product('N', T(1), hmatrix, ones.data(), T(0), result.data());

    if (Symmetry == 'N') {
        add_matrix_vector_product('N', T(1), dense_hmatrix, ones.data(), T(0), reference.data());
    } else if (Symmetry == 'S') {
        add_symmetric_matrix_vector_product(UPLO, T(1), dense_hmatrix, ones.data(), T(0), reference.data());
    } else {
        add_hermitian_matrix_vector_product(UPLO, T(1), dense_hmatrix, ones.data(), T(0), reference.data());
    }

    double error = norm2(result - reference) / norm2(reference);
    is_error     = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix vector product: " << error << endl;

    return is_error;
}
