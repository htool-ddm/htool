#ifndef HTOOL_MATRIX_LINALG_ADD_MATRIX_MATRIX_PRODUCT_HPP
#define HTOOL_MATRIX_LINALG_ADD_MATRIX_MATRIX_PRODUCT_HPP

#include "../../matrix/linalg/scale.hpp"   // for scale
#include "../../matrix/matrix.hpp"         // for Matrix
#include "../../wrappers/wrapper_blas.hpp" // for Blas
namespace htool {

// template <typename T>
// void add_matrix_matrix_product(char transa, char transb, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, int mu) {
//     int nr     = A.nb_rows();
//     int nc     = A.nb_cols();
//     char trans = transa;
//     int lda    = nr;
//     int M      = nr;
//     int N      = mu;
//     int K      = nc;
//     int ldb    = nc;
//     int ldc    = nr;
//     if (trans != 'N') {
//         M   = nc;
//         N   = mu;
//         K   = nr;
//         ldb = nr;
//         ldc = nc;
//     }

//     Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
// }

template <typename T>
void add_matrix_matrix_product(char transa, char transb, T alpha, const Matrix<T> &A, const Matrix<T> &B, T beta, Matrix<T> &C) {
    int M   = C.nb_rows();
    int N   = C.nb_cols();
    int K   = transa == 'N' ? A.nb_cols() : A.nb_rows(); // also transa == 'N' ? B.nb_rows() : B.nb_cols();
    int lda = transa == 'N' ? M : K;
    int ldb = transb == 'N' ? K : N;
    int ldc = M;

    if (ldb > 0) { // AB is zero
        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, A.data(), &lda, B.data(), &ldb, &beta, C.data(), &ldc);
    } else {
        scale(beta, C);
    }

    // add_matrix_matrix_product(transa, transb, alpha, A, B.data(), beta, C.data(), C.nb_cols());
}

// template <typename T>
// void add_matrix_matrix_product_symmetric(char side, char UPLO, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, const int &mu) {

//     int M   = (side == 'L') ? A.nb_rows() : mu;
//     int N   = (side == 'L') ? mu : A.nb_cols();
//     int lda = A.nb_rows();
//     int ldb = (side == 'L') ? A.nb_cols() : mu;
//     int ldc = ldb;
//     Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
// }

// template <typename T>
// void add_symmetric_matrix_matrix_product(char side, char transa, char transb, std::complex<T> alpha, const Matrix<std::complex<T>> &A, const std::complex<T> *in, std::complex<T> beta, std::complex<T> *out, const int &mu, char UPLO, char symmetry) {
//     int nr = A.nb_rows();

//     if (transb != 'N') {
//         htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_symmetric (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
//     }
//     if (side != 'L') {
//         htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_matrix_product_symmetric (side=" + std::string(1, side) + ")"); // LCOV_EXCL_LINE
//     }

//     if (nr) {
//         int lda = nr;
//         int M   = nr;
//         int N   = mu;
//         int ldb = A.nb_cols();
//         int ldc = nr;

//         if (symmetry == 'S' && (transa == 'N' || transa == 'T')) {
//             Blas<std::complex<T>>::symm(&side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
//         } else if (symmetry == 'H' && (transa == 'N' || transa == 'C')) {
//             Blas<std::complex<T>>::hemm(&side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
//         } else if (symmetry == 'S' && transa == 'C') {
//             std::vector<std::complex<T>> conjugate_in(nr * mu);
//             std::complex<T> conjugate_alpha = std::conj(alpha);
//             std::complex<T> conjugate_beta  = std::conj(beta);
//             std::transform(in, in + nr * mu, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
//             conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
//             Blas<std::complex<T>>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
//             conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
//         } else if (symmetry == 'H' && transa == 'T') {
//             std::vector<std::complex<T>> conjugate_in(nr * mu);
//             std::complex<T> conjugate_alpha = std::conj(alpha);
//             std::complex<T> conjugate_beta  = std::conj(beta);
//             std::transform(in, in + nr * mu, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
//             std::transform(out, out + nr * mu, out, [](const std::complex<T> &c) { return std::conj(c); });
//             Blas<std::complex<T>>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
//             std::transform(out, out + nr * mu, out, [](const std::complex<T> &c) { return std::conj(c); });
//         } else {
//             htool::Logger::get_instance().log(LogLevel::ERROR, "Invalid arguments for add_matrix_product_symmetric: " + std::string(1, transa) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
//         }
//     }
// }

template <typename T>
void add_symmetric_matrix_matrix_product(char side, char UPLO, T alpha, const Matrix<T> &A, const Matrix<T> &B, T beta, Matrix<T> &C) {
    int M   = C.nb_rows();
    int N   = C.nb_cols();
    int lda = (side == 'L') ? M : N;
    int ldb = M;
    int ldc = M;
    Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, A.data(), &lda, B.data(), &ldb, &beta, C.data(), &ldc);
}

template <typename T>
void add_hermitian_matrix_matrix_product(char side, char UPLO, T alpha, const Matrix<T> &A, const Matrix<T> &B, T beta, Matrix<T> &C) {
    int M   = C.nb_rows();
    int N   = C.nb_cols();
    int lda = (side == 'L') ? M : N;
    int ldb = M;
    int ldc = M;
    Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, A.data(), &lda, B.data(), &ldb, &beta, C.data(), &ldc);
}

} // namespace htool

#endif
