#ifndef HTOOL_LRMAT_LINALG_ADD_LRMAT_LRMAT_HPP
#define HTOOL_LRMAT_LINALG_ADD_LRMAT_LRMAT_HPP
#include "../../../matrix/linalg/add_matrix_matrix_product.hpp"
#include "../lrmat.hpp"
#include "add_lrmat_matrix_product.hpp"

namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision>
void add_lrmat_lrmat(LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &C, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &R, const Cluster<CoordinatePrecision> &target_cluster_child, const Cluster<CoordinatePrecision> &source_cluster_child) {
    int row_offset = target_cluster_child.get_offset() - target_cluster.get_offset();
    int col_offset = source_cluster_child.get_offset() - source_cluster.get_offset();

    bool C_is_overwritten = (C.rank_of() == 0);

    auto &U_R = R.get_U();
    auto &V_R = R.get_V();
    auto &U_C = C.get_U();
    auto &V_C = C.get_V();

    if (C_is_overwritten) {
        U_C.resize(C.nb_rows(), R.rank_of());
        V_C.resize(R.rank_of(), C.nb_cols());

        for (int i = 0; i < U_R.nb_cols(); i++) {
            std::copy_n(U_R.data() + U_R.nb_rows() * i, U_R.nb_rows(), U_C.data() + i * U_C.nb_rows() + row_offset);
        }
        for (int j = 0; j < V_R.nb_cols(); j++) {
            std::copy_n(V_R.data() + j * V_R.nb_rows(), V_R.nb_rows(), V_C.data() + (j + col_offset) * V_C.nb_rows());
        }

    } else {

        // Concatenate U_C and U_R
        Matrix<CoefficientPrecision> new_U(C.nb_rows(), U_R.nb_cols() + U_C.nb_cols());
        std::copy_n(U_C.data(), U_C.nb_cols() * U_C.nb_rows(), new_U.data());

        for (int i = 0; i < U_R.nb_cols(); i++) {
            std::copy_n(U_R.data() + U_R.nb_rows() * i, U_R.nb_rows(), new_U.data() + U_C.nb_cols() * C.nb_rows() + i * new_U.nb_rows() + row_offset);
        }
        // Concatenate V_C and V_R
        Matrix<CoefficientPrecision> new_V(V_C.nb_rows() + V_R.nb_rows(), C.nb_cols());

        for (int j = 0; j < new_V.nb_cols(); j++) {
            std::copy_n(V_C.data() + j * V_C.nb_rows(), V_C.nb_rows(), new_V.data() + j * new_V.nb_rows());
        }
        for (int j = 0; j < V_R.nb_cols(); j++) {
            std::copy_n(V_R.data() + j * V_R.nb_rows(), V_R.nb_rows(), new_V.data() + (j + col_offset) * new_V.nb_rows() + V_C.nb_rows());
        }

        // Set C
        C.get_U() = new_U;
        C.get_V() = new_V;
        recompression(C);
    }
}

} // namespace htool

#endif
