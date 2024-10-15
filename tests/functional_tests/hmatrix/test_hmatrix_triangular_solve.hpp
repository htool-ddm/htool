#include <htool/basic_types/tree.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_hmatrix_product.hpp>
#include <htool/hmatrix/linalg/triangular_hmatrix_hmatrix_solve.hpp>
#include <htool/hmatrix/linalg/triangular_hmatrix_lrmat_solve.hpp>
#include <htool/hmatrix/linalg/triangular_hmatrix_matrix_solve.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/lrmat/linalg/add_matrix_lrmat_product.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/misc/misc.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <iostream>
#include <memory>
#include <vector>
using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_triangular_solve(char side, char transa, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 10;
    htool::underlying_type<T> error;

    // Random input
    T alpha(1);
    generate_random_scalar(alpha);

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case(side, transa, n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_X(epsilon, eta, 'N', 'N');

    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    HMatrix<T, htool::underlying_type<T>> X = hmatrix_tree_builder_X.build(*test_case.operator_X, *test_case.root_cluster_X_output, *test_case.root_cluster_X_input);
    HMatrix<T, htool::underlying_type<T>> B(X), UB(X), LB(X);
    HMatrix<T, htool::underlying_type<T>> hmatrix_test(B);

    // Lrmat rhs
    SVD<T> compressor;
    htool::underlying_type<T> lrmat_tolerance = 1e-3;
    LowRankMatrix<T> X_lrmat(*test_case.operator_X, compressor, *test_case.root_cluster_X_output, *test_case.root_cluster_X_input, 15, lrmat_tolerance);
    LowRankMatrix<T> lrmat_test(epsilon);

    // Triangular hmatrices
    HMatrix<T, htool::underlying_type<T>> LA(A);
    HMatrix<T, htool::underlying_type<T>> UA(A);
    preorder_tree_traversal(LA, [](HMatrix<T, htool::underlying_type<T>> &hmatrix) {
        if (hmatrix.is_leaf() and hmatrix.get_target_cluster() == hmatrix.get_source_cluster()) {
            Matrix<T> &dense_data = *hmatrix.get_dense_data();
            for (int i = 0; i < dense_data.nb_rows(); i++) {
                for (int j = i + 1; j < dense_data.nb_cols(); j++) {
                    dense_data(i, j) = 0;
                }
            }
        } else {
            std::vector<std::unique_ptr<HMatrix<T, htool::underlying_type<T>>>> filtered_children;
            for (auto &child : hmatrix.get_children_with_ownership()) {
                if (child->get_target_cluster().get_offset() >= child->get_source_cluster().get_offset()) {
                    filtered_children.push_back(std::move(child));
                }
            }
            if (filtered_children.size() > 0) {
                hmatrix.delete_children();
                hmatrix.assign_children(filtered_children);
            }
        }
    });

    preorder_tree_traversal(UA, [](HMatrix<T, htool::underlying_type<T>> &hmatrix) {
        if (hmatrix.is_leaf() and hmatrix.get_target_cluster() == hmatrix.get_source_cluster()) {
            Matrix<T> &dense_data = *hmatrix.get_dense_data();
            for (int j = 0; j < dense_data.nb_cols(); j++) {
                for (int i = j + 1; i < dense_data.nb_rows(); i++) {
                    dense_data(i, j) = 0;
                }
            }
        } else {
            std::vector<std::unique_ptr<HMatrix<T, htool::underlying_type<T>>>> filtered_children;
            for (auto &child : hmatrix.get_children_with_ownership()) {
                if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
                    filtered_children.push_back(std::move(child));
                }
            }
            if (filtered_children.size() > 0) {
                hmatrix.delete_children();
                hmatrix.assign_children(filtered_children);
            }
        }
    });

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test, dense_lrmat_test, dense_X_lrmat;
    test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());
    test_case.operator_X->copy_submatrix(no_X, ni_X, test_case.root_cluster_X_output->get_offset(), test_case.root_cluster_X_input->get_offset(), X_dense.data());
    dense_X_lrmat.resize(X_lrmat.nb_rows(), X_lrmat.nb_cols());
    X_lrmat.copy_to_dense(dense_X_lrmat.data());

    // Triangular matrices
    Matrix<T> LA_dense(A_dense), UA_dense(A_dense), LB_dense(B.nb_rows(), B.nb_cols()), UB_dense(B.nb_rows(), B.nb_cols());
    for (int i = 0; i < A.nb_rows(); i++) {
        for (int j = 0; j < A.nb_cols(); j++) {
            if (i > j) {
                UA_dense(i, j) = 0;
            }
            if (i < j) {
                LA_dense(i, j) = 0;
            }
        }
    }
    LowRankMatrix<T> UB_lrmat(epsilon), LB_lrmat(epsilon);
    if (side == 'L') {
        add_matrix_matrix_product(transa, 'N', T(1) / alpha, UA_dense, X_dense, T(0), UB_dense);
        add_matrix_matrix_product(transa, 'N', T(1) / alpha, LA_dense, X_dense, T(0), LB_dense);
        add_matrix_lrmat_product(transa, 'N', T(1) / alpha, UA_dense, X_lrmat, T(0), UB_lrmat);
        add_matrix_lrmat_product(transa, 'N', T(1) / alpha, LA_dense, X_lrmat, T(0), LB_lrmat);
        internal_add_hmatrix_hmatrix_product(transa, 'N', T(1) / alpha, UA, X, T(0), UB);
        internal_add_hmatrix_hmatrix_product(transa, 'N', T(1) / alpha, LA, X, T(0), LB);

    } else {
        add_matrix_matrix_product('N', transa, T(1) / alpha, X_dense, UA_dense, T(0), UB_dense);
        add_matrix_matrix_product('N', transa, T(1) / alpha, X_dense, LA_dense, T(0), LB_dense);

        // TODO: use add_lrmat_matrix_product when transb will work

        UB_lrmat.get_U() = X_lrmat.get_U();
        UB_lrmat.get_V().resize(X_lrmat.get_V().nb_rows(), A_dense.nb_cols());
        LB_lrmat.get_U() = X_lrmat.get_U();
        LB_lrmat.get_V().resize(X_lrmat.get_V().nb_rows(), A_dense.nb_cols());

        add_matrix_matrix_product('N', transa, T(1) / alpha, X_lrmat.get_V(), UA_dense, T(0), UB_lrmat.get_V());
        add_matrix_matrix_product('N', transa, T(1) / alpha, X_lrmat.get_V(), LA_dense, T(0), LB_lrmat.get_V());
        // add_lrmat_matrix_product('N', transa, T(1) / alpha, X_lrmat, UA_dense, T(0), UB_lrmat);
        // add_lrmat_matrix_product('N', transa, T(1) / alpha, X_lrmat, LA_dense, T(0), LB_lrmat);

        internal_add_hmatrix_hmatrix_product('N', transa, T(1) / alpha, X, UA, T(0), UB);
        internal_add_hmatrix_hmatrix_product('N', transa, T(1) / alpha, X, LA, T(0), LB);
    }

    save_leaves_with_rank(LA, "LA");
    save_leaves_with_rank(LB, "LB");

    // triangular_matrix_matrix_solve
    matrix_test = LB_dense;
    internal_triangular_hmatrix_matrix_solve(side, 'L', transa, 'N', alpha, LA, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on lower triangular hmatrix matrix solve: " << error << endl;

    matrix_test = UB_dense;
    internal_triangular_hmatrix_matrix_solve(side, 'U', transa, 'N', alpha, UA, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on upper triangular hmatrix matrix solve: " << error << endl;

    lrmat_test = LB_lrmat;
    internal_triangular_hmatrix_lrmat_solve(side, 'L', transa, 'N', alpha, LA, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(dense_X_lrmat - dense_lrmat_test) / normFrob(dense_X_lrmat);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on lower triangular hmatrix lrmat solve: " << error << endl;

    lrmat_test = UB_lrmat;
    internal_triangular_hmatrix_lrmat_solve(side, 'U', transa, 'N', alpha, UA, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(dense_X_lrmat - dense_lrmat_test) / normFrob(dense_X_lrmat);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on upper triangular hmatrix lrmat solve: " << error << endl;

    hmatrix_test = LB;
    internal_triangular_hmatrix_hmatrix_solve(side, 'L', transa, 'N', alpha, LA, hmatrix_test);
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(X_dense - densified_hmatrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on lower triangular hmatrix hmatrix solve: " << error << endl;

    hmatrix_test = UB;
    internal_triangular_hmatrix_hmatrix_solve(side, 'U', transa, 'N', alpha, UA, hmatrix_test);
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(X_dense - densified_hmatrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on upper triangular hmatrix hmatrix solve: " << error << endl;

    return is_error;
}
