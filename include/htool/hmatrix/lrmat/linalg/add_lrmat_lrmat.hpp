#ifndef HTOOL_LRMAT_LINALG_ADD_LRMAT_LRMAT_HPP
#define HTOOL_LRMAT_LINALG_ADD_LRMAT_LRMAT_HPP
#include "../../../clustering/cluster_node.hpp"               // for Cluster (ptr ...
#include "../../../hmatrix/lrmat/utils/SVD_recompression.hpp" // for recompression
#include "../../../matrix/matrix.hpp"                         // for Matrix
#include "../../../misc/logger.hpp"                           // for Logger, LogLevel
#include "../../../misc/misc.hpp"                             // for underlying_type
#include "../lrmat.hpp"                                       // for LowRankMatrix
#include <algorithm>                                          // for copy_n
#include <string>                                             // for basic_string

namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_lrmat_lrmat(const LowRankMatrix<CoefficientPrecision> &X_lrmat, const Cluster<CoordinatePrecision> &X_target_cluster, const Cluster<CoordinatePrecision> &X_source_cluster, LowRankMatrix<CoefficientPrecision> &Y_lrmat, const Cluster<CoordinatePrecision> &Y_target_cluster, const Cluster<CoordinatePrecision> &Y_source_cluster) {
    if (left_cluster_contains_right_cluster(Y_target_cluster, X_target_cluster) && left_cluster_contains_right_cluster(Y_source_cluster, X_source_cluster)) { // extends X and add to Y
        int row_offset = X_target_cluster.get_offset() - Y_target_cluster.get_offset();
        int col_offset = X_source_cluster.get_offset() - Y_source_cluster.get_offset();

        bool C_is_overwritten = (Y_lrmat.rank_of() == 0);

        auto &U_X = X_lrmat.get_U();
        auto &V_X = X_lrmat.get_V();
        auto &U_Y = Y_lrmat.get_U();
        auto &V_Y = Y_lrmat.get_V();

        if (C_is_overwritten) {
            U_Y.resize(Y_lrmat.nb_rows(), X_lrmat.rank_of());
            V_Y.resize(X_lrmat.rank_of(), Y_lrmat.nb_cols());

            for (int i = 0; i < U_X.nb_cols(); i++) {
                std::copy_n(U_X.data() + U_X.nb_rows() * i, U_X.nb_rows(), U_Y.data() + i * U_Y.nb_rows() + row_offset);
            }
            for (int j = 0; j < V_X.nb_cols(); j++) {
                std::copy_n(V_X.data() + j * V_X.nb_rows(), V_X.nb_rows(), V_Y.data() + (j + col_offset) * V_Y.nb_rows());
            }

        } else {

            // Concatenate U_X and U_Y
            Matrix<CoefficientPrecision> new_U(Y_lrmat.nb_rows(), U_X.nb_cols() + U_Y.nb_cols());
            std::copy_n(U_Y.data(), U_Y.nb_cols() * U_Y.nb_rows(), new_U.data());

            for (int i = 0; i < U_X.nb_cols(); i++) {
                std::copy_n(U_X.data() + U_X.nb_rows() * i, U_X.nb_rows(), new_U.data() + U_Y.nb_cols() * Y_lrmat.nb_rows() + i * new_U.nb_rows() + row_offset);
            }
            // Concatenate V_X and V_Y
            Matrix<CoefficientPrecision> new_V(V_Y.nb_rows() + V_X.nb_rows(), Y_lrmat.nb_cols());

            for (int j = 0; j < new_V.nb_cols(); j++) {
                std::copy_n(V_Y.data() + j * V_Y.nb_rows(), V_Y.nb_rows(), new_V.data() + j * new_V.nb_rows());
            }
            for (int j = 0; j < V_X.nb_cols(); j++) {
                std::copy_n(V_X.data() + j * V_X.nb_rows(), V_X.nb_rows(), new_V.data() + (j + col_offset) * new_V.nb_rows() + V_Y.nb_rows());
            }

            // Set C
            Y_lrmat.get_U() = new_U;
            Y_lrmat.get_V() = new_V;
            SVD_recompression(Y_lrmat);
        }
    } else if (left_cluster_contains_right_cluster(X_target_cluster, Y_target_cluster) && left_cluster_contains_right_cluster(X_source_cluster, Y_source_cluster)) { // restrict X and add to Y
        int row_offset = Y_target_cluster.get_offset() - X_target_cluster.get_offset();
        int col_offset = Y_source_cluster.get_offset() - X_source_cluster.get_offset();

        bool C_is_overwritten = (Y_lrmat.rank_of() == 0);

        auto &U_X = X_lrmat.get_U();
        auto &V_X = X_lrmat.get_V();
        auto &U_Y = Y_lrmat.get_U();
        auto &V_Y = Y_lrmat.get_V();

        if (C_is_overwritten) {
            U_Y.resize(Y_lrmat.nb_rows(), X_lrmat.rank_of());
            V_Y.resize(X_lrmat.rank_of(), Y_lrmat.nb_cols());

            for (int i = 0; i < U_X.nb_cols(); i++) {
                std::copy_n(U_X.data() + U_X.nb_rows() * i + row_offset, U_Y.nb_rows(), U_Y.data() + i * U_Y.nb_rows());
            }
            for (int j = 0; j < V_Y.nb_cols(); j++) {
                std::copy_n(V_X.data() + (j + col_offset) * V_X.nb_rows(), V_X.nb_rows(), V_Y.data() + j * V_Y.nb_rows());
            }
        } else {
            // Concatenate U_X and U_Y
            Matrix<CoefficientPrecision> new_U(Y_lrmat.nb_rows(), U_X.nb_cols() + U_Y.nb_cols());
            std::copy_n(U_Y.data(), U_Y.nb_cols() * U_Y.nb_rows(), new_U.data());

            for (int i = 0; i < U_X.nb_cols(); i++) {
                std::copy_n(U_X.data() + U_X.nb_rows() * i + row_offset, U_Y.nb_rows(), new_U.data() + U_Y.nb_cols() * Y_lrmat.nb_rows() + i * new_U.nb_rows());
            }
            // Concatenate V_C and V_R
            Matrix<CoefficientPrecision> new_V(V_Y.nb_rows() + V_X.nb_rows(), Y_lrmat.nb_cols());

            for (int j = 0; j < new_V.nb_cols(); j++) {
                std::copy_n(V_Y.data() + j * V_Y.nb_rows(), V_Y.nb_rows(), new_V.data() + j * new_V.nb_rows());
            }
            for (int j = 0; j < V_Y.nb_cols(); j++) {
                std::copy_n(V_X.data() + (j + col_offset) * V_X.nb_rows(), V_X.nb_rows(), new_V.data() + j * new_V.nb_rows() + V_Y.nb_rows());
            }

            // Set C
            Y_lrmat.get_U() = new_U;
            Y_lrmat.get_V() = new_V;
            SVD_recompression(Y_lrmat);
        }
    } else {
        Logger::get_instance().log(LogLevel::ERROR, "Operation not implemented in add_lrmat_lrmat.");
    }
}

} // namespace htool

#endif
