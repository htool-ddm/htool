#ifndef HTOOL_BLOC_ACA_HPP
#define HTOOL_BLOC_ACA_HPP

#include "../../misc/logger.hpp"
#include "../../wrappers/wrapper_lapack.hpp"
#include "lrmat.hpp"
#include <cassert>
#include <iomanip> // std::setprecision, std::setw
#include <vector>

// #    include <chrono>
// using namespace std::chrono;

namespace htool {
//================================//
//   CLASSE MATRICE RANG FAIBLE   //
//================================//
//
// Refs biblio:
//
//  -> article de Liu et al. :
//           https://journals.sagepub.com/doi/10.1177/1094342020918305
//           et en particulier l'algorithme 2
//
//=================================//
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class blocACA final : public VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> {

  public:
    //=========================//
    //    BLOC ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision>::VirtualLowRankGenerator;

    int m_bloc_size;
    blocACA(int d) : m_bloc_size(d) {}

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &target_cluster, const Cluster<CoordinatesPrecision> &source_cluster, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {
        bool tmp_bool;

        const int bloc_size                  = m_bloc_size;
        const CoefficientPrecision one       = 1;
        const CoefficientPrecision minus_one = -1;
        const CoefficientPrecision zero      = 0;

        underlying_type<CoefficientPrecision> Nu = 0;
        underlying_type<CoefficientPrecision> Mu = 0;
        CoefficientPrecision s                   = 0;

        int target_size   = target_cluster.get_size();
        int source_size   = source_cluster.get_size();
        int target_offset = target_cluster.get_offset();
        int source_offset = source_cluster.get_offset();
        int q             = 0;
        int r             = 0;
        int reqrank       = rank;
        int lwork, info;
        int tmp_int;
        int uu_size;
        int q_times_r;
        int incx(1), incy(1);

        std::vector<Matrix<CoefficientPrecision>> uu, vv;

        Matrix<CoefficientPrecision> Ck(target_size, bloc_size), CkT(bloc_size, target_size), Rk(bloc_size, source_size), RkT(source_size, bloc_size), holed_Rk(bloc_size, source_size), Wk(bloc_size, bloc_size), Uk, Vk, UkTUk, VkVkT, part_uu, part_vv, vvVkT, uuTUk;
        // Matrix<underlying_type<CoefficientPrecision>> vvVkT, uuTUk;

        std::vector<bool> visited_row(target_size, false);
        std::vector<bool> visited_col(source_size, false);
        std::vector<int> Jpvt, JpvtJk;
        std::vector<CoefficientPrecision> tau, work;
        std::vector<underlying_type<CoefficientPrecision>> rwork;

        //// First bloc
        std::vector<int> Ik(bloc_size), Jk(bloc_size), JkNext(bloc_size), Jbar;
        std::iota(Jk.begin(), Jk.end(), 0);

        // =====================================================================================
        // std::cout << std::fixed             // fix the number of decimal digits...
        //           << std::setprecision(22); // ... to 11
        // std::cout << "reqrank=" << reqrank << endl;
        // std::cout << "++++++++++++++++++ \nA=" << endl;
        // Matrix<CoefficientPrecision> TotMat(target_size, source_size);
        // A.copy_submatrix(target_size, source_size, target_offset, source_offset, TotMat.data());
        // TotMat.print(cout, ",");
        // std::cout << "++++++++++++++++++" << endl;
        // getchar();
        // =====================================================================================

        // Either we have a required rank
        // Or it is negative and we have to check the relative error between two iterations.
        // But to do that we need a least two iterations.

        while (((reqrank > 0) && (q < std::min(reqrank, std::min(target_size, source_size)))) || ((reqrank < 0) && (q == 0 || (Nu / Mu) > epsilon))) {
            // Test if the next current rank would not be advantageous
            if (q * (target_size + source_size) > (target_size * source_size)) {
                q = -1;
                break;
            }
            uu_size = uu.size(); // == vv.size()

            for (int i = 0; i < bloc_size; i++) {
                visited_col[Jk[i]] = true; // Could change later if bloc Wk is rank deficient
            }

            // std::cout << "visited_col=";
            // for (int j = 0; j < source_size; j++) {
            //     std::cout << visited_col[j] << ", ";
            // }
            // std::cout << endl;

            // Define Ck the current column bloc
            // // Copy A(:,Jk) into Ck by taking into account non-contiguous columns
            for (int j = 0; j < bloc_size; j++) {
                A.copy_submatrix(target_size, 1, target_offset, Jk[j] + source_offset, Ck.data() + j * target_size); // column major
            }

            // std::cout << "++++++++++++++++++ \nA(:,Jk)=" << endl;
            // Ck.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;

            // // Add -uu*vv(:,Jk) into Ck
            for (int k = 0; k < uu_size; k++) { // loop on uu / vv items because uu.size() == vv.size()
                tmp_int = (uu[k]).nb_cols();
                part_uu.resize(target_size, tmp_int);
                part_vv.resize(tmp_int, bloc_size);

                for (int j = 0; j < tmp_int; j++) {       // loop on uu[k] columns / vv[k] rows
                    for (int p = 0; p < bloc_size; p++) { // loop on vv non-contiguous columns
                        part_vv(j, p) = vv[k](j, Jk[p]);
                    }
                    for (int i = 0; i < target_size; i++) { // loop on uu[k] rows
                        part_uu(i, j) = uu[k](i, j);
                    }
                }

                Blas<CoefficientPrecision>::gemm("N", "N", &(target_size), &(bloc_size), &(tmp_int), &minus_one, part_uu.data(), &(target_size), part_vv.data(), &(tmp_int), &one, Ck.data(), &(target_size));
            }
            for (int p = 0; p < bloc_size; p++) {       // loop on vv non-contiguous columns
                for (int i = 0; i < target_size; i++) { // loop on uu[k] rows
                    if (!visited_row[i]) {
                        CkT(p, i) = Ck(i, p);
                    } else {
                        CkT(p, i) = 0;
                    }
                }
            }

            // std::cout << "++++++++++++++++++ \nCk(:,Jk)=" << endl;
            // Ck.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;
            // std::cout << "++++++++++++++++++ \nCkT(Jk,:)=" << endl;
            // CkT.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;

            // Define Ik the current vector of row indices definning the current bloc row
            Jpvt.resize(target_size);                     // /!\ does not remove existing elements
            std::fill(Jpvt.begin(), Jpvt.end(), int(0));  // because in/out arg
            tau.resize(std::min(target_size, bloc_size)); // out arg
            work.resize(1);                               // out arg
            rwork.resize(2 * target_size);                // out arg
            lwork = -1;

            Lapack<CoefficientPrecision>::geqp3(&(bloc_size), &(target_size), CkT.data(), &(bloc_size), Jpvt.data(), tau.data(), work.data(), &lwork, rwork.data(), &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::geqp3(&(bloc_size), &(target_size), CkT.data(), &(bloc_size), Jpvt.data(), tau.data(), work.data(), &lwork, rwork.data(), &info);
            tmp_int = 0;
            for (int j = 0; j < target_size; j++) {
                if (!visited_row[Jpvt[j] - 1]) {
                    Ik[tmp_int] = Jpvt[j] - 1; // because Jpvt start with 1
                    tmp_int += 1;
                }
                if (tmp_int == bloc_size) {
                    break;
                }
            }

            for (int i = 0; i < bloc_size; i++) {
                visited_row[Ik[i]] = true; // Could change later if bloc Wk is rank deficient
            }

            // std::cout << "++++++++++++++++++ \nQR(CkT)=" << endl;
            // CkT.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;
            // std::cout << "Jpvt-1=\n";
            // for (int j = 0; j < target_size; j++) {
            //     std::cout << Jpvt[j] - 1 << ",";
            // }
            // std::cout << endl;

            // std::cout << "Jk=[";
            // for (int i = 0; i < bloc_size; i++) {
            //     std::cout << Jk[i] << ",";
            // }
            // std::cout << "]" << endl;
            // std::cout << "Ik=[";
            // for (int i = 0; i < bloc_size; i++) {
            //     std::cout << Ik[i] << ",";
            // }
            // std::cout << "]" << endl;

            // std::cout << "visited_row=";
            // for (int j = 0; j < target_size; j++) {
            //     std::cout << visited_row[j] << ", ";
            // }
            // std::cout << endl;

            // Define Rk the current row bloc
            // Copy A(Ik,:) into Rk by taking into account non-contiguous rows
            for (int i = 0; i < bloc_size; i++) {
                A.copy_submatrix(1, source_size, Ik[i] + target_offset, source_offset, RkT.data() + i * source_size); // columns major
            }

            // Add A(Ik,:)-uu(Ik,:)*vv into Rk
            for (int p = 0; p < source_size; p++) {   // loop on vv[k] columns
                for (int i = 0; i < bloc_size; i++) { // loop on uu[k] non-contiguous rows
                    Rk(i, p) = RkT(p, i);
                }
            }

            for (int k = 0; k < uu_size; k++) { // loop on uu / vv items because uu.size() == vv.size()
                tmp_int = (uu[k]).nb_cols();
                part_uu.resize(bloc_size, tmp_int);
                part_vv.resize(tmp_int, source_size);

                for (int j = 0; j < tmp_int; j++) {         // loop on uu[k] columns / vv[k] rows
                    for (int p = 0; p < source_size; p++) { // loop on vv non-contiguous columns
                        part_vv(j, p) = vv[k](j, p);
                    }
                    for (int i = 0; i < bloc_size; i++) { // loop on uu[k] rows
                        part_uu(i, j) = uu[k](Ik[i], j);
                    }
                }

                Blas<CoefficientPrecision>::gemm("N", "N", &(bloc_size), &(source_size), &(tmp_int), &minus_one, part_uu.data(), &(bloc_size), part_vv.data(), &(tmp_int), &one, Rk.data(), &(bloc_size));
            }
            for (int p = 0; p < source_size; p++) {   // loop on vv[k] columns
                for (int i = 0; i < bloc_size; i++) { // loop on uu[k] non-contiguous rows
                    if (!visited_col[p]) {
                        holed_Rk(i, p) = Rk(i, p);
                    } else {
                        holed_Rk(i, p) = 0;
                    }
                }
            }

            // std::cout << "++++++++++++++++++ \nRk=" << endl;
            // Rk.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;
            // std::cout << "++++++++++++++++++ \nholedRk=" << endl;
            // holed_Rk.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;

            // Define JkNext the next vector of indices definning the next bloc column
            Jpvt.resize(source_size);
            std::fill(Jpvt.begin(), Jpvt.end(), int(0)); // because in/out arg
            tau.resize(std::min(source_size, bloc_size));
            work.resize(1);
            rwork.resize(2 * source_size);
            lwork = -1;
            Lapack<CoefficientPrecision>::geqp3(&(bloc_size), &(source_size), holed_Rk.data(), &(bloc_size), Jpvt.data(), tau.data(), work.data(), &lwork, rwork.data(), &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::geqp3(&(bloc_size), &(source_size), holed_Rk.data(), &(bloc_size), Jpvt.data(), tau.data(), work.data(), &lwork, rwork.data(), &info);
            JpvtJk  = Jpvt;
            tmp_int = 0;
            for (int j = 0; j < source_size; j++) {
                if (!visited_col[Jpvt[j] - 1]) {
                    JkNext[tmp_int] = Jpvt[j] - 1; // := J_{k+1}, needed later to update Jk
                    tmp_int += 1;
                }
                if (tmp_int == bloc_size) {
                    break;
                }
            }

            // Define Wk the current square bloc: Copy Ck(Ik,:) into Wk by taking into account non-contiguous rows and columns.
            for (int p = 0; p < bloc_size; p++) {
                for (int i = 0; i < bloc_size; i++) {
                    Wk(i, p) = Ck(Ik[i], p);
                }
            }

            // std::cout << "JkNext=" << JkNext[0] << "," << JkNext[1] << endl; // must be = to Jpvt[0]-1
            // std::cout << "++++++++++++++++++ \nWk=" << endl;
            // Wk.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;

            // LRID
            // // QR factorization of Wk for epsilon tolerance
            Jpvt.resize(bloc_size);
            std::fill(Jpvt.begin(), Jpvt.end(), int(0)); // because in/out arg
            work.resize(1);
            rwork.resize(2 * bloc_size);
            lwork = -1;
            Lapack<CoefficientPrecision>::gelsy(&bloc_size, &bloc_size, &source_size, Wk.data(), &bloc_size, Rk.data(), &bloc_size, Jpvt.data(), &epsilon, &r, work.data(), &lwork, rwork.data(), &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::gelsy(&bloc_size, &bloc_size, &source_size, Wk.data(), &bloc_size, Rk.data(), &bloc_size, Jpvt.data(), &epsilon, &r, work.data(), &lwork, rwork.data(), &info); // max_i { |T_{i,i}| < epsilon * |T_{1,1}| }
            if (r <= 0) {
                std::cerr << "!!! In blocACA.hpp effective rank of bloc Wk is lower or equal to 0 !!!" << std::endl;
            }

            // std::cout << "++++++++++++++++++ \nQR(Wk)=" << endl;
            // Wk.print(cout, ", ");
            // std::cout << "++++++++++++++++++" << endl;

            // std::cout << "r=" << r << endl;
            if ((q + r > reqrank) && (reqrank > 0)) {
                r = reqrank - q;
            }
            // std::cout << "Corrected r=" << r << endl;

            // // Define Jbar
            Jbar.resize(r);
            for (int i = 0; i < r; i++) {
                Jbar[i] = Jpvt[i] - 1; // because Jpvt start with 1
            }

            // std::cout << "Jbar=" << Jbar[0] << endl;
            //"," << Jbar[1] << endl;
            // std::cout << "++++++++++++++++++ \nX=" << endl;
            // Rk.print(cout, ", ");
            // std::cout << "++++++++++++++++++" << endl;

            // // Define Vk :
            Vk.resize(r, source_size);
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < source_size; j++) {
                    Vk(i, j) = Rk(i, j);
                }
            }

            // // Define Uk
            Uk.resize(target_size, r);
            for (int i = 0; i < target_size; i++) {
                for (int j = 0; j < r; j++) {
                    Uk(i, j) = Ck(i, j); // todo pk on a enlevé Jbar ?
                }
            }

            // std::cout << "++++++++++++++++++ \nUk=" << endl;
            // Uk.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;
            // std::cout << "++++++++++++++++++ \nVk=" << endl;
            // Vk.print(cout, ",");
            // std::cout << "++++++++++++++++++" << endl;

            // Update visited_row and visited_col
            for (int i = r; i < bloc_size; i++) {
                visited_row[Ik[i]] = false;
            }
            for (int j = 0; j < bloc_size; j++) {
                tmp_bool = true;
                for (int i = 0; i < r; i++) {
                    if (j == Jbar[i]) {
                        tmp_bool = false;
                        break;
                    }
                }
                if (tmp_bool) {
                    visited_col[Jk[j]] = false;
                }
            }

            // std::cout << "UpdatedVisited_col=";
            // for (int j = 0; j < source_size; j++) {
            //     std::cout << visited_col[j] << ", ";
            // }
            // std::cout << endl;
            // std::cout << "UpdatedVisited_row=";
            // for (int j = 0; j < target_size; j++) {
            //     std::cout << visited_row[j] << ", ";
            // }
            // std::cout << endl;

            if (reqrank < 0) { // need to compute Mu and Nu
                // LRnorm
                UkTUk.resize(r, r);
                Blas<CoefficientPrecision>::gemm("C", "N", &(r), &(r), &(target_size), &one, Uk.data(), &(target_size), Uk.data(), &(target_size), &zero, UkTUk.data(), &(r)); // does not pass test if  C <-> T (different from Liu2020)

                // std::cout << "++++++++++++++++++ \nUkTUk=" << endl;
                // UkTUk.print(cout, ",");

                Lapack<CoefficientPrecision>::potrf("U", &r, UkTUk.data(), &r, &info); // T1

                // std::cout << "++++++++++++++++++ \nChol(UkTUk)=" << endl;
                // UkTUk.print(cout, ",");
                // std::cout << "++++++++++++++++++" << endl;

                // // Vk*Vk**T
                VkVkT.resize(r, r);
                Blas<CoefficientPrecision>::gemm("N", "C", &(r), &(r), &(source_size), &one, Vk.data(), &(r), Vk.data(), &(r), &zero, VkVkT.data(), &(r)); // does not pass test if  C <-> T (different from Liu2020)

                // std::cout << "++++++++++++++++++ \nVk=" << endl;
                // Vk.print(cout, ",");

                // std::cout << "++++++++++++++++++ \nVkVkT=" << endl;
                // VkVkT.print(cout, ",");

                Lapack<CoefficientPrecision>::potrf("U", &r, VkVkT.data(), &r, &info); // T2

                // std::cout << "++++++++++++++++++ \nChol(VkVkT)=" << endl;
                // VkVkT.print(cout, ",");
                // std::cout << "++++++++++++++++++" << endl;

                // // T1*T2**T
                Blas<CoefficientPrecision>::trmm("R", "U", "C", "N", &r, &r, &one, VkVkT.data(), &r, UkTUk.data(), &r);

                // std::cout << "++++++++++++++++++ \nT1*T2**T=" << endl;
                // UkTUk.print(cout, ",");
                // std::cout << "++++++++++++++++++" << endl;

                // // Nu
                Nu = normFrob(UkTUk);

                // Define uuTUk
                uuTUk.resize(q, r);
                for (int k = 0; k < uu_size; k++) { // loop on uu / vv items because uu.size() == vv.size()
                    tmp_int = (uu[k]).nb_cols();
                    Blas<CoefficientPrecision>::gemm("C", "N", &(tmp_int), &(r), &(target_size), &one, uu[k].data(), &(target_size), Uk.data(), &(target_size), &one, uuTUk.data(), &(tmp_int));
                }

                //  Define vvVkT
                vvVkT.resize(q, r);
                for (int k = 0; k < uu_size; k++) { // loop on uu / vv items because uu.size() == vv.size()
                    tmp_int = (vv[k]).nb_rows();
                    Blas<CoefficientPrecision>::gemm("N", "C", &(tmp_int), &(r), &(source_size), &one, vv[k].data(), &(tmp_int), Vk.data(), &(r), &one, vvVkT.data(), &(tmp_int));
                }

                // Define tilde{V}
                q_times_r = q * r;
                s         = Blas<CoefficientPrecision>::dot(&(q_times_r), uuTUk.data(), &incx, vvVkT.data(), &incy);

                Mu = std::sqrt(Mu * Mu + Nu * Nu + 2 * std::real(s));
                std::cout << "Nu=" << Nu << endl;
                std::cout << "Mu=" << Mu << endl;
                // std::cout << "sqrt(s)=" << std::sqrt(std::real(s)) << endl;
                std::cout << "(s)=" << (std::real(s)) << endl;

                // getchar();
            }

            // Update
            uu.push_back(Uk);
            vv.push_back(Vk);
            q += r;
            Jk = JkNext;

            // std::cout << "q=" << q << endl;
            // std::cout << "q * (target_size + source_size)=" << q * (target_size + source_size) << endl;
            // std::cout << "========== Fin d'iteration ===========\n";
            // getchar();

        } // End while
        // Final rank
        rank = q;
        if (rank > 0) {
            U.resize(target_size, rank);
            V.resize(rank, source_size);
            int offset_U = 0;
            int offset_V = 0;

            for (int k = 0; k < vv.size(); k++) { // uu.size == vv.size
                r = (vv[k]).nb_rows();            // r == (uu[k]).nb_cols()
                for (int p = 0; p < source_size; p++) {
                    for (int i = 0; i < r; i++) {
                        V(offset_V + i, p) = vv[k](i, p);
                    }
                }
                offset_V += r;

                for (int p = 0; p < target_size; p++) {
                    for (int j = 0; j < r; j++) {
                        U(p, offset_U + j) = uu[k](p, j);
                    }
                }
                offset_U += r;
            }
        }

        // std::cout << "++++++++++++++++++ \nU=" << endl;
        // U.print(cout, ",");
        // std::cout << "++++++++++++++++++" << endl;
        // std::cout << "++++++++++++++++++ \nV=" << endl;
        // V.print(cout, ",");
        // std::cout << "++++++++++++++++++" << endl;
        // std::cout << "++++++++++++++++++ \nUV=" << endl;
        // (U * V).print(cout, ",");
        // std::cout << "++++++++++++++++++" << endl;
        // std::cout << "normF(A-U*V)= " << normFrob(TotMat - U * V) << endl;

    } // End "copy_low_rank_approximation" function
};    // End blocACA class
} // namespace htool
#endif
