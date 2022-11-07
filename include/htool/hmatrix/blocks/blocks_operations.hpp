#ifndef HTOOL_BLOCKS_BLOCKS_OPERATIONS_HPP
#define HTOOL_BLOCKS_BLOCKS_OPERATIONS_HPP

#include "blocks.hpp"

namespace htool {

// C = alpha*C
template <typename T>
void Hscale(Block<T> &C, T alpha) {
    if (C.IsLeaf() && C.get_dense_block_data() != nullptr) {
        C.get_dense_block_data()->scale(alpha);
    } else if (C.IsLeaf() && C.get_low_rank_block_data() != nullptr) {
        C.get_low_rank_block_data()->scale(alpha);
    } else {
        for (int i = 0; i < C.get_nb_sons(); i++) {
            Hscale(C.get_son(i), alpha);
        }
    }
}

// y = aC*x
template <typename T>
void Hmatvec(Block<T> &C, const T *x, T *y) {
    if (C.IsLeaf() && C.get_dense_block_data() != nullptr) {
        C.get_dense_block_data()->mvprod(x + C.get_source_cluster().get_offset(), y + C.get_target_cluster().get_offset());
    } else if (C.IsLeaf() && C.get_low_rank_block_data() != nullptr) {
        C.get_low_rank_block_data()->mvprod(x + C.get_source_cluster().get_offset(), y + C.get_target_cluster().get_offset());
    } else {
        for (int i = 0; i < C.get_nb_sons(); i++) {
            Hmatvec(C.get_son(i), x, y);
        }
    }
}

// y = A*B
template <typename T>
void HmatmatToDense(Block<T> &A, Block<T> &B, T out) {
    if (A.get_dense_block_data() != nullptr && B.get_dense_block_data() != nullptr) { // only if A or B is dense
        throw std::logic_error("Cannot call HmatmatToDense")
    }

    if (B.get_dense_block_data() != nullptr) { // if B is dense, compute B^T A^T
    }

    //     C.get_dense_block_data()->mvprod(x + C.get_source_cluster().get_offset(), y + C.get_target_cluster().get_offset());
    // } else if (C.IsLeaf() && C.get_low_rank_block_data() != nullptr) {
    //     C.get_low_rank_block_data()->mvprod(x + C.get_source_cluster().get_offset(), y + C.get_target_cluster().get_offset());
    // } else {
    //     for (int i = 0; i < C.get_nb_sons(); i++) {
    //         Hmatvec(C.get_son(i), x, y);
    //     }
    // }
}

// C = alpha*A*B + beta*C
template <typename T>
void Hgemm(Block<T> &C, T beta) {
    // if (C.get_target_cluster() == A.get_target_cluster() && C.get_source_cluster() == B.get_source_cluster() && A.get_source_cluster() == B.get_target_cluster()) {
    //     throw std::logic_error("[Htool error] Hgemm needs compatible clusters for input blocks");
    // }

    if (C.IsRoot()) {
        Hscale(C, beta);
    }

    if (!C.IsLeaf()) {
        for (int i = 0; i < C.get_nb_sons(); i++) {

            // if (C.get_son(i).get_target_cluster() == A.get_son(j).get_target_cluster() && C.get_son(i).get_source_cluster() == B.get_son(k).get_source_cluster() && A.get_son(j).get_source_cluster() == B.get_son(k).get_target_cluster())
            C.restrict_sum_expressions(i);
            Hgemm(C.get_son(i));
        }
        else {
            if (C.IsAdmissible()) {
                C.compute_approximate_sum_expressions();
            } else {
                C.compute_dense_sum_expressions();
            }
        }
        // } else {
        //     if (C.IsLeaf() && C.get_dense_block_data() != nullptr) { // C is dense
        //         char transa = 'N';
        //         char transb = 'N';
        //         int M       = C.get_target_cluster().get_size();
        //         int N       = C.get_source_cluster().get_size();
        //         int K       = A.get_source_cluster().get_size();
        //         int ldb     = K;
        //         int ldc     = M;
        //         T *A_ptr    = nullptr;
        //         T *B_ptr    = nullptr;
        //         if (A.IsLeaf() && A.get_dense_block_data() != nullptr) { // A is dense
        //             A_ptr = A.get_dense_block_data()->data();
        //         } else if (A.IsLeaf() && A.get_low_rank_block_data() != nullptr) { // A is low rank
        //             A_ptr = new T[A.get_size()];
        //             A.get_low_rank_block_data()->get_whole_matrix(A_ptr);
        //         } else { // A is hierarchical
        //         }

        //         if (B.IsLeaf() && B.get_dense_block_data() != nullptr) { // B is dense
        //             B_ptr = B.get_dense_block_data()->data();
        //         } else if (B.IsLeaf() && B.get_low_rank_block_data() != nullptr) { // B is low rank
        //             B_ptr = new T[B.get_size()];
        //             B.get_low_rank_block_data()->get_whole_matrix(A_ptr);
        //         } else { // B is hierarchical
        //         }

        //         Blas::gemm(&transa, &transb, &M, &N, &K, &alpha, A_ptr, lda, B_ptr, ldb, &beta, C.get_dense_block_data()->data(), ldc);

        //         if (A.IsLeaf() && A.get_low_dense_data() == nullptr) { // A is not dense
        //             delete[] A_ptr;
        //         }
        //         if (B.IsLeaf() && B.get_low_dense_data() == nullptr) { // B is not dense
        //             delete[] B_ptr;
        //         }

        //     } else if (C.IsLeaf() && C.get_low_rank_block_data() != nullptr) { // C is low rank
        //     }
    }
}

} // namespace htool

#endif
