#include <htool/hmatrix/linalg/add_hmatrix_matrix_product.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/utility.hpp> // for HMatrixBuilder
#include <htool/testing/generator_input.hpp>
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

    // Reference solution
    Matrix<T> dense_hmatrix(nr, nc);
    std::vector<T> vec_in(nc, 1), vec_out(nr, 1), vec_result(nr, 0), vec_reference(nr, 0);
    std::vector<T> vec_buffer(nr + nc);
    Matrix<T> mat_in(nc, 10), mat_out(nr, 10), mat_result(nr, 10), mat_reference(nr, 10);
    vector<T> mat_buffer((nr + nc) * 10);
    double error;
    T alpha, beta;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_vector(vec_in);
    generate_random_vector(vec_out);
    generate_random_matrix(mat_in);
    generate_random_matrix(mat_out);
    vec_reference = vec_out;
    mat_reference = mat_out;
    copy_to_dense_in_user_numbering(hmatrix, dense_hmatrix.data());

    if (Symmetry == 'N') {
        add_matrix_vector_product('N', alpha, dense_hmatrix, vec_in.data(), beta, vec_reference.data());
        add_matrix_matrix_product('N', 'N', alpha, dense_hmatrix, mat_in, beta, mat_reference);
    } else if (Symmetry == 'S') {
        add_symmetric_matrix_vector_product(UPLO, alpha, dense_hmatrix, vec_in.data(), beta, vec_reference.data());
        add_symmetric_matrix_matrix_product('L', UPLO, alpha, dense_hmatrix, mat_in, beta, mat_reference);
    } else {
        if constexpr (is_complex<T>()) {
            add_hermitian_matrix_vector_product(UPLO, alpha, dense_hmatrix, vec_in.data(), beta, vec_reference.data());
            add_hermitian_matrix_matrix_product('L', UPLO, alpha, dense_hmatrix, mat_in, beta, mat_reference);
        }
    }

#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
    // Test sequential vector product without buffer
    vec_result = vec_out;
    add_hmatrix_vector_product(std::execution::seq, 'N', alpha, hmatrix, vec_in.data(), beta, vec_result.data());

    error    = norm2(vec_result - vec_reference) / norm2(vec_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a sequential hmatrix vector product wo buffer: " << error << endl;

    // Test parallel vector product without buffer
    vec_result = vec_out;
    add_hmatrix_vector_product(std::execution::par, 'N', alpha, hmatrix, vec_in.data(), beta, vec_result.data());

    error    = norm2(vec_result - vec_reference) / norm2(vec_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a parallel hmatrix vector product wo buffer: " << error << endl;

    // Test sequential vector product with buffer
    vec_result = vec_out;
    add_hmatrix_vector_product(std::execution::seq, 'N', alpha, hmatrix, vec_in.data(), beta, vec_result.data(), vec_buffer.data());

    error    = norm2(vec_result - vec_reference) / norm2(vec_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a sequential hmatrix vector product with buffer: " << error << endl;

    // Test parallel vector product with buffer
    vec_result = vec_out;
    add_hmatrix_vector_product(std::execution::par, 'N', alpha, hmatrix, vec_in.data(), beta, vec_result.data(), vec_buffer.data());

    error    = norm2(vec_result - vec_reference) / norm2(vec_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a paralel hmatrix vector product with buffer: " << error << endl;

    // Test sequential matrix product without buffer
    mat_result = mat_out;
    add_hmatrix_matrix_product(std::execution::seq, 'N', 'N', alpha, hmatrix, mat_in, beta, mat_result);

    error    = normFrob(mat_result - mat_reference) / normFrob(mat_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a sequential hmatrix matrix product wo buffer: " << error << endl;

    // Test parallel matrix product without buffer
    mat_result = mat_out;
    add_hmatrix_matrix_product(std::execution::par, 'N', 'N', alpha, hmatrix, mat_in, beta, mat_result);

    error    = normFrob(mat_result - mat_reference) / normFrob(mat_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a parallel hmatrix matrix product wo buffer: " << error << endl;

    // Test sequential matrix product with buffer
    mat_result = mat_out;
    add_hmatrix_matrix_product(std::execution::seq, 'N', 'N', alpha, hmatrix, mat_in, beta, mat_result, mat_buffer.data());

    error    = normFrob(mat_result - mat_reference) / normFrob(mat_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a sequential hmatrix vector product with buffer: " << error << endl;

    // Test parallel matrix product with buffer
    mat_result = mat_out;
    add_hmatrix_matrix_product(std::execution::par, 'N', 'N', alpha, hmatrix, mat_in, beta, mat_result, mat_buffer.data());

    error    = normFrob(mat_result - mat_reference) / normFrob(mat_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a paralel hmatrix matrix product with buffer: " << error << endl;
#else
    // Test vector product without buffer
    vec_result = vec_out;
    add_hmatrix_vector_product('N', alpha, hmatrix, vec_in.data(), beta, vec_result.data());

    error    = norm2(vec_result - vec_reference) / norm2(vec_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix vector product wo buffer: " << error << endl;

    // Test vector product with buffer
    vec_result = vec_out;
    add_hmatrix_vector_product('N', alpha, hmatrix, vec_in.data(), beta, vec_result.data(), vec_buffer.data());

    error    = norm2(vec_result - vec_reference) / norm2(vec_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix vector product with buffer: " << error << endl;

    // Test sequential matrix product without buffer
    mat_result = mat_out;
    add_hmatrix_matrix_product('N', 'N', alpha, hmatrix, mat_in, beta, mat_result);

    error    = normFrob(mat_result - mat_reference) / normFrob(mat_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix vector product wo buffer: " << error << endl;

    // Test parallel matrix product with buffer
    mat_result = mat_out;
    add_hmatrix_matrix_product('N', 'N', alpha, hmatrix, mat_in, beta, mat_result, mat_buffer.data());

    error    = normFrob(mat_result - mat_reference) / normFrob(mat_reference);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix matrix product with buffer: " << error << endl;
#endif
    return is_error;
}
