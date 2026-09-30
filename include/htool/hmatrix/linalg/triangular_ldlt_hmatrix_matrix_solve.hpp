#ifndef HTOOL_HMATRIX_LINALG_TRIANGULAR_LDLT_HMATRIX_HMATRIX_SOLVE_ROW_MAJOR_HPP
#define HTOOL_HMATRIX_LINALG_TRIANGULAR_LDLT_HMATRIX_HMATRIX_SOLVE_ROW_MAJOR_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/linalg/factorization.hpp"
#include "../../matrix/linalg/transpose.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/misc.hpp"
#include "../hmatrix.hpp"
#include "add_hmatrix_matrix_product_row_major.hpp"
#include <vector>

namespace htool {
// Same recursion as internal_triangular_hmatrix_matrix_solve_row_major, solving with the unit L/U of an
// LDLt factorization (D is applied separately), transa in {N,T,C}; the dense leaf goes through
// triangular_ldlt_matrix_matrix_solve.
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_triangular_ldlt_hmatrix_matrix_solve_row_major(char side, char UPLO, char transa, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &Bt) {
    if (alpha != CoefficientPrecision(1)) {
        scale(alpha, Bt);
    }

    if (!A.is_leaf()) {
        bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0);

        std::vector<const Cluster<CoordinatePrecision> *> output_clusters, input_clusters;
        auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
        auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

        if (transa != 'N') {
            get_input_cluster_A  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
            get_output_cluster_A = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        }
        const Cluster<CoordinatePrecision> &output_cluster = (A.*get_output_cluster_A)();
        const Cluster<CoordinatePrecision> &input_cluster  = (A.*get_input_cluster_A)();

        if (output_cluster.is_leaf() || (block_tree_not_consistent and output_cluster.get_rank() >= 0)) {
            output_clusters.push_back(&output_cluster); // LCOV_EXCL_LINE (A is a hierarchical diagonal block: its cluster is never a leaf)
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : output_cluster.get_clusters_on_partition()) {
                output_clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : output_cluster.get_children()) {
                output_clusters.push_back(output_cluster_child.get());
            }
        }

        if (input_cluster.is_leaf() || (block_tree_not_consistent and input_cluster.get_rank() >= 0)) {
            input_clusters.push_back(&input_cluster); // LCOV_EXCL_LINE (A is a hierarchical diagonal block: its cluster is never a leaf)
        } else if (block_tree_not_consistent) {
            for (auto &input_cluster_child : input_cluster.get_clusters_on_partition()) {
                input_clusters.push_back(input_cluster_child);
            }
        } else {
            for (auto &input_cluster_child : input_cluster.get_children()) {
                input_clusters.push_back(input_cluster_child.get());
            }
        }

        // Forward
        if ((UPLO == 'L' && transa == 'N') || (UPLO == 'U' && transa != 'N')) {
            for (auto &output_cluster_child : output_clusters) {
                Matrix<CoefficientPrecision> current_Bt_to_modify;
                current_Bt_to_modify.assign(Bt.nb_rows(), output_cluster_child->get_size(), Bt.data() + Bt.nb_rows() * (output_cluster_child->get_offset() - output_cluster.get_offset()), false);

                for (auto &input_cluster_child : input_clusters) {

                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*output_cluster_child, *input_cluster_child) : A.get_sub_hmatrix(*input_cluster_child, *output_cluster_child);
                    if (*output_cluster_child == *input_cluster_child) {
                        internal_triangular_ldlt_hmatrix_matrix_solve_row_major(side, UPLO, transa, CoefficientPrecision(1), *A_child, current_Bt_to_modify);

                    } else if (output_cluster_child->get_offset() > input_cluster_child->get_offset()) {
                        Matrix<CoefficientPrecision> current_Bt;
                        current_Bt.assign(Bt.nb_rows(), input_cluster_child->get_size(), Bt.data() + Bt.nb_rows() * (input_cluster_child->get_offset() - input_cluster.get_offset()), false);

                        internal_add_hmatrix_matrix_product_row_major(transa, 'N', CoefficientPrecision(-1), *A_child, current_Bt.data(), CoefficientPrecision(1), current_Bt_to_modify.data(), current_Bt.nb_rows());
                    }
                }
            }
        } // Backward
        else if ((UPLO == 'U' && transa == 'N') || (UPLO == 'L' && transa != 'N')) {
            for (auto output_it = output_clusters.rbegin(); output_it != output_clusters.rend(); ++output_it) {
                auto &output_cluster_child = *output_it;

                Matrix<CoefficientPrecision> current_Bt_to_modify;
                current_Bt_to_modify.assign(Bt.nb_rows(), output_cluster_child->get_size(), Bt.data() + Bt.nb_rows() * (output_cluster_child->get_offset() - output_cluster.get_offset()), false);

                for (auto input_it = input_clusters.rbegin(); input_it != input_clusters.rend(); ++input_it) {
                    auto &input_cluster_child = *input_it;

                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*output_cluster_child, *input_cluster_child) : A.get_sub_hmatrix(*input_cluster_child, *output_cluster_child);
                    if (*output_cluster_child == *input_cluster_child) {
                        internal_triangular_ldlt_hmatrix_matrix_solve_row_major(side, UPLO, transa, CoefficientPrecision(1), *A_child, current_Bt_to_modify);

                    } else if (output_cluster_child->get_offset() < input_cluster_child->get_offset()) {
                        Matrix<CoefficientPrecision> current_Bt;
                        current_Bt.assign(Bt.nb_rows(), input_cluster_child->get_size(), Bt.data() + Bt.nb_rows() * (input_cluster_child->get_offset() - input_cluster.get_offset()), false);

                        internal_add_hmatrix_matrix_product_row_major(transa, 'N', CoefficientPrecision(-1), *A_child, current_Bt.data(), CoefficientPrecision(1), current_Bt_to_modify.data(), current_Bt.nb_rows());
                    }
                }
            }
        }
    } else {
        Matrix<CoefficientPrecision> B(Bt.nb_cols(), Bt.nb_rows());
        transpose(Bt, B);
        triangular_ldlt_matrix_matrix_solve(side, UPLO, transa, CoefficientPrecision(1), *A.get_dense_data(), B);
        transpose(B, Bt);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_triangular_ldlt_hmatrix_matrix_solve(char side, char UPLO, char transa, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &B) {
    if (side == 'L') {
        Matrix<CoefficientPrecision> transposed_B(B.nb_cols(), B.nb_rows());
        transpose(B, transposed_B);
        internal_triangular_ldlt_hmatrix_matrix_solve_row_major(side, UPLO, transa, alpha, A, transposed_B);
        transpose(transposed_B, B);
    } else {
        // Same reduction to side='L' as internal_triangular_hmatrix_matrix_solve: B^T for 'N'/'T', and
        // conj(B)^T for 'C', whose transposed op (conj(A)) is not a transa.
        char transposed_transa = transa == 'N' ? 'T' : 'N';
        if (transa == 'C') {
            conj_if_complex(B.data(), B.nb_rows() * B.nb_cols());
        }
        internal_triangular_ldlt_hmatrix_matrix_solve_row_major('L', UPLO, transposed_transa, transa == 'C' ? conj_if_complex(alpha) : alpha, A, B);
        if (transa == 'C') {
            conj_if_complex(B.data(), B.nb_rows() * B.nb_cols());
        }
    }
}

} // namespace htool
#endif
