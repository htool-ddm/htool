#ifndef HTOOL_GENEO_COARSE_SPACE_BUILDER_HPP
#define HTOOL_GENEO_COARSE_SPACE_BUILDER_HPP

#include "../../matrix/matrix.hpp"
// #include "../../wrappers/wrapper_hpddm.hpp"
#include "../../wrappers/wrapper_lapack.hpp"
// #include "../hmatrix/hmatrix.hpp"
#include "../interfaces/virtual_coarse_space_builder.hpp"
namespace htool {

template <typename CoefficientPrecision>
class GeneoCoarseSpaceDenseBuilder : public VirtualCoarseSpaceBuilder<CoefficientPrecision> {

  protected:
    int m_size_wo_overlap;
    int m_size_with_overlap;
    Matrix<CoefficientPrecision> m_DAiD;
    Matrix<CoefficientPrecision> m_Bi;
    char m_symmetry                                                = 'N';
    char m_uplo                                                    = 'N';
    int m_geneo_nu                                                 = 2;
    htool::underlying_type<CoefficientPrecision> m_geneo_threshold = -1.;
    explicit GeneoCoarseSpaceDenseBuilder(int size_wo_overlap, const Matrix<CoefficientPrecision> &Ai, Matrix<CoefficientPrecision> &Bi, char symmetry, char uplo, int geneo_nu, htool::underlying_type<CoefficientPrecision> geneo_threshold) : m_size_wo_overlap(size_wo_overlap), m_size_with_overlap(Ai.nb_cols()), m_DAiD(m_size_with_overlap, m_size_with_overlap), m_Bi(Bi), m_symmetry(symmetry), m_uplo(uplo), m_geneo_nu(geneo_nu), m_geneo_threshold(geneo_threshold) {
        for (int i = 0; i < m_size_wo_overlap; i++) {
            std::copy_n(Ai.data() + i * m_size_with_overlap, m_size_wo_overlap, &(m_DAiD(0, i)));
        }
    }

  public:
    static GeneoCoarseSpaceDenseBuilder GeneoWithNu(int size_wo_overlap, const Matrix<CoefficientPrecision> &Ai, Matrix<CoefficientPrecision> &Bi, char symmetry, char uplo, int geneo_nu) { return GeneoCoarseSpaceDenseBuilder{size_wo_overlap, Ai, Bi, symmetry, uplo, geneo_nu, -1}; }

    static GeneoCoarseSpaceDenseBuilder GeneoWithThreshold(int size_wo_overlap, const Matrix<CoefficientPrecision> &Ai, Matrix<CoefficientPrecision> &Bi, char symmetry, char uplo, htool::underlying_type<CoefficientPrecision> geneo_threshold) { return GeneoCoarseSpaceDenseBuilder{size_wo_overlap, Ai, Bi, symmetry, uplo, 0, geneo_threshold}; }

    virtual Matrix<CoefficientPrecision> build_coarse_space() override {

        int n    = m_size_with_overlap;
        int ldvl = n, ldvr = n, lwork = -1;
        int lda = n, ldb = n;
        std::vector<CoefficientPrecision> work(n);
        std::vector<double> rwork;
        std::vector<int> index(n, 0);
        std::iota(index.begin(), index.end(), int(0));
        int nevi = m_geneo_nu;
        int info;
        // int rankWorld;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
        // m_DAiD.csv_save("m_DAiD_" + NbrToStr(rankWorld));
        // std::cout << "?????? " << m_Bi.nb_rows() << " " << m_Bi.nb_cols() << "\n";
        // m_Bi.csv_save("m_Bi_" + NbrToStr(rankWorld));
        // std::cout << "BOUH " << m_geneo_nu << " " << m_geneo_threshold << "\n";
        if (m_symmetry == 'S' || m_symmetry == 'H') {
            int itype = 1;
            std::vector<underlying_type<CoefficientPrecision>> w(n);
            if (is_complex<CoefficientPrecision>()) {
                rwork.resize(3 * n - 2);
            }

            // std::cout << "OUAAAAH\n";
            Lapack<CoefficientPrecision>::gv(&itype, "V", &m_uplo, &n, m_DAiD.data(), &lda, m_Bi.data(), &ldb, w.data(), work.data(), &lwork, rwork.data(), &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::gv(&itype, "V", &m_uplo, &n, m_DAiD.data(), &lda, m_Bi.data(), &ldb, w.data(), work.data(), &lwork, rwork.data(), &info);

            // std::cout << "OUAAAAH 2\n";
            std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
                return (std::abs(w[a]) > std::abs(w[b]));
            });
            if (m_geneo_threshold > 0.0) {
                nevi = 0;
                while (std::abs(w[index[nevi]]) > m_geneo_threshold && nevi < index.size()) {
                    nevi++;
                }
            }

            // if (rankWorld == 0) {
            //     for (int i = 0; i < index.size(); i++) {
            //         std::cout << std::abs(w[index[i]]) << " ";
            //     }
            //     std::cout << "\n";
            //     // std::cout << vr << "\n";
            // }
            // MPI_Barrier(MPI_COMM_WORLD);
            // if (rankWorld == 1) {
            //     std::cout << "w : " << w << "\n";
            //     std::cout << "info: " << info << "\n";
            //     for (int i = 0; i < index.size(); i++) {
            //         std::cout << std::abs(w[index[i]]) << " ";
            //     }
            //     std::cout << "\n";
            //     // std::cout << vr << "\n";
            // }

            Matrix<CoefficientPrecision> Z(n, nevi);
            for (int i = 0; i < m_size_wo_overlap; i++) {
                for (int j = 0; j < nevi; j++) {
                    Z(i, j) = m_DAiD(i, index[j]);
                }
            }
            return Z;
        }

        if (is_complex<CoefficientPrecision>()) {
            rwork.resize(8 * n);
        }
        std::vector<CoefficientPrecision> alphar(n), alphai((is_complex<CoefficientPrecision>() ? 0 : n)), beta(n);
        std::vector<CoefficientPrecision> vl(n * n), vr(n * n);

        Lapack<CoefficientPrecision>::ggev("N", "V", &n, m_DAiD.data(), &lda, m_Bi.data(), &ldb, alphar.data(), alphai.data(), beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        Lapack<CoefficientPrecision>::ggev("N", "V", &n, m_DAiD.data(), &lda, m_Bi.data(), &ldb, alphar.data(), alphai.data(), beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);

        std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
            return ((std::abs(beta[a]) < 1e-15 || (std::abs(alphar[a] / beta[a]) > std::abs(alphar[b] / beta[b]))) && !(std::abs(beta[b]) < 1e-15));
        });
        if (m_geneo_threshold > 0.0) {
            nevi = 0;
            while (std::abs(beta[index[nevi]]) < 1e-15 || (std::abs(alphar[index[nevi]] / beta[index[nevi]]) > m_geneo_threshold && nevi < index.size())) {
                nevi++;
            }
        }

        // if (rankWorld == 0) {
        //     for (int i = 0; i < index.size(); i++) {
        //         std::cout << std::abs(alphar[index[i]] / beta[index[i]]) << " ";
        //     }
        //     std::cout << "\n";
        //     // std::cout << vr << "\n";
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
        // if (rankWorld == 1) {
        //     // std::cout << "alphar : " << alphar << "\n";
        //     // std::cout << "alphai : " << alphai << "\n";
        //     // std::cout << "beta : " << beta << "\n";
        //     // std::cout << "info: " << info << "\n";
        //     for (int i = 0; i < index.size(); i++) {
        //         std::cout << std::abs(alphar[index[i]] / beta[index[i]]) << " ";
        //     }
        //     std::cout << "\n";
        //     // std::cout << vr << "\n";
        // }

        Matrix<CoefficientPrecision> Z(n, nevi);
        for (int i = 0; i < m_size_wo_overlap; i++) {
            for (int j = 0; j < nevi; j++) {
                Z(i, j) = vr[index[j] * n + i];
            }
        }
        return Z;
    }
};

} // namespace htool

#endif
