#ifndef HTOOL_HMATRIX_LINALG_APPLY_LDLT_DIAGONAL_HPP
#define HTOOL_HMATRIX_LINALG_APPLY_LDLT_DIAGONAL_HPP

#include "../../matrix/linalg/factorization.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/logger.hpp"
#include "../hmatrix.hpp"
#include <algorithm>
#include <memory>
#include <vector>

namespace htool {
// B := D^-1 B (side='L') or B D^-1 (side='R'), D being the block-diagonal factor of an LDLt
// factorization held at A's dense diagonal leaves (see apply_ldlt_diagonal). B's rows (side='L')
// or columns (side='R') are numbered from offset.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void internal_apply_ldlt_diagonal(char symmetry, char side, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &B, int offset) {
    if (A.is_hierarchical()) {
        for (auto &child : A.get_children()) {
            if (child->get_target_cluster() == child->get_source_cluster()) {
                internal_apply_ldlt_diagonal(symmetry, side, UPLO, *child, B, offset);
            }
        }
    } else {
        int local_offset = A.get_target_cluster().get_offset() - offset;
        int size         = A.get_target_cluster().get_size();
        if (side == 'R') {
            Matrix<CoefficientPrecision> B_columns;
            B_columns.assign(B.nb_rows(), size, B.data() + B.nb_rows() * local_offset, false);
            apply_ldlt_diagonal(symmetry, 'R', UPLO, *A.get_dense_data(), B_columns);
        } else {
            Matrix<CoefficientPrecision> B_rows(size, B.nb_cols());
            for (int j = 0; j < B.nb_cols(); j++) {
                std::copy_n(B.data() + local_offset + j * B.nb_rows(), size, B_rows.data() + j * size);
            }
            apply_ldlt_diagonal(symmetry, 'L', UPLO, *A.get_dense_data(), B_rows);
            for (int j = 0; j < B.nb_cols(); j++) {
                std::copy_n(B_rows.data() + j * size, size, B.data() + local_offset + j * B.nb_rows());
            }
        }
    }
}

// Same with an HMatrix B, following B's own block structure so that its blocks keep their compression.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void internal_apply_ldlt_diagonal(char symmetry, char side, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, HMatrix<CoefficientPrecision, CoordinatePrecision> &B) {
    int offset = side == 'L' ? B.get_target_cluster().get_offset() : B.get_source_cluster().get_offset();
    if (B.is_dense()) {
        internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, *B.get_dense_data(), offset);
    } else if (B.is_low_rank()) {
        internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, side == 'L' ? B.get_low_rank_data()->get_U() : B.get_low_rank_data()->get_V(), offset);
    } else {
        std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> A_children;
        for (auto &child : B.get_children()) {
            const Cluster<CoordinatePrecision> &cluster = side == 'L' ? child->get_target_cluster() : child->get_source_cluster();
            A_children.push_back(A.get_sub_hmatrix(cluster, cluster));
        }
        if (std::find(A_children.begin(), A_children.end(), nullptr) == A_children.end()) {
            for (std::size_t c = 0; c < A_children.size(); c++) {
                internal_apply_ldlt_diagonal(symmetry, side, UPLO, *A_children[c], *B.get_children()[c]);
            }
            return;
        }
        // B is refined below A's dense leaves, where a 2x2 pivot block of D may straddle B's children.
        htool::Logger::get_instance().log(LogLevel::WARNING, "In internal_apply_ldlt_diagonal, B is refined below the dense diagonal leaves holding D and is densified, losing its compression.");
        auto B_dense = std::make_unique<Matrix<CoefficientPrecision>>(B.nb_rows(), B.nb_cols());
        copy_to_dense(B, B_dense->data());
        internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, *B_dense, offset);
        B.set_dense_data(std::move(B_dense));
    }
}
} // namespace htool
#endif
