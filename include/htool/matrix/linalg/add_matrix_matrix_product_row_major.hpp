#ifndef HTOOL_MATRIX_LINALG_ADD_MATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_MATRIX_LINALG_ADD_MATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP

#include "../matrix.hpp"
namespace htool {

template <typename T>
void add_matrix_matrix_product_row_major(char transa, char transb, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, int mu) {

    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_row_major (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    int nr               = A.nb_rows();
    int nc               = A.nb_cols();
    char inverted_transa = 'N';
    char inverted_transb = 'T';
    int M                = mu;
    int N                = nr;
    int K                = nc;
    int lda              = mu;
    int ldb              = nr;
    int ldc              = mu;
    if (transa != 'N') {
        inverted_transb = 'N';
        N               = nc;
        K               = nr;
    }
    if (transa == 'C' && is_complex<T>()) {
        std::vector<T> conjugate_in(nr * mu);
        T conjugate_alpha = conj_if_complex<T>(alpha);
        T conjugate_beta  = conj_if_complex<T>(beta);
        std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return conj_if_complex<T>(c); });
        conj_if_complex<T>(out, nc * mu);
        Blas<T>::gemm(&inverted_transa, &inverted_transb, &M, &N, &K, &conjugate_alpha, conjugate_in.data(), &lda, A.data(), &ldb, &conjugate_beta, out, &ldc);
        conj_if_complex<T>(out, nc * mu);
        return;
    }
    Blas<T>::gemm(&inverted_transa, &inverted_transb, &M, &N, &K, &alpha, in, &lda, A.data(), &ldb, &beta, out, &ldc);
}

template <typename T>
void add_matrix_matrix_product_symmetric_row_major(char side, char, char transb, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, const int &mu, char UPLO, char) {

    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_symmetric (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    if (side != 'L') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_symmetric (side=" + std::string(1, side) + ")"); // LCOV_EXCL_LINE
    }

    int nr = A.nb_rows();

    if (nr) {
        int lda            = nr;
        char inverted_side = 'R';
        int M              = mu;
        int N              = nr;
        int ldb            = mu;
        int ldc            = mu;

        Blas<T>::symm(&inverted_side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
    }
}

template <typename T>
void add_matrix_matrix_product_symmetric_row_major(char side, char transa, char transb, std::complex<T> alpha, const Matrix<std::complex<T>> &A, const std::complex<T> *in, std::complex<T> beta, std::complex<T> *out, const int &mu, char UPLO, char symmetry) {
    int nr = A.nb_rows();

    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_symmetric (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    if (side != 'L') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_symmetric (side=" + std::string(1, side) + ")"); // LCOV_EXCL_LINE
    }

    if (nr) {
        int lda            = nr;
        char inverted_side = 'R';
        int M              = mu;
        int N              = nr;
        int ldb            = mu;
        int ldc            = mu;

        if (symmetry == 'S' && (transa == 'N' || transa == 'T')) {
            Blas<std::complex<T>>::symm(&inverted_side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
        } else if (symmetry == 'H' && transa == 'T') {
            Blas<std::complex<T>>::hemm(&inverted_side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
        } else if (symmetry == 'S' && transa == 'C') {
            std::vector<std::complex<T>> conjugate_in(nr * mu);
            std::complex<T> conjugate_alpha = std::conj(alpha);
            std::complex<T> conjugate_beta  = std::conj(beta);
            std::transform(in, in + nr * mu, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
            conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
            Blas<std::complex<T>>::symm(&inverted_side, &UPLO, &M, &N, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
            conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
        } else if (symmetry == 'H' && (transa == 'N' || transa == 'C')) {
            std::vector<std::complex<T>> conjugate_in(nr * mu);
            std::complex<T> conjugate_alpha = std::conj(alpha);
            std::complex<T> conjugate_beta  = std::conj(beta);
            std::transform(in, in + nr * mu, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
            conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
            Blas<std::complex<T>>::hemm(&inverted_side, &UPLO, &M, &N, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
            conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
        } else {
            htool::Logger::get_instance().log(LogLevel::ERROR, "Invalid arguments for add_matrix_product_symmetric_row_major: " + std::string(1, transa) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
            // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")"); // LCOV_EXCL_LINE
        }
    }
}

} // namespace htool

#endif
