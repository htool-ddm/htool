#ifndef HTOOL_MATRIX_LINALG_ADD_MATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_MATRIX_LINALG_ADD_MATRIX_VECTOR_PRODUCT_HPP

#include "../matrix.hpp"
namespace htool {

template <typename T>
void add_matrix_vector_product(char trans, T alpha, const Matrix<T> &A, const T *in, T beta, T *out) {
    int nr   = A.nb_rows();
    int nc   = A.nb_cols();
    int lda  = nr;
    int incx = 1;
    int incy = 1;
    Blas<T>::gemv(&trans, &nr, &nc, &alpha, A.data(), &lda, in, &incx, &beta, out, &incy);
}

template <typename T>
void add_matrix_vector_product_symmetric(char, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, char UPLO, char) {
    int nr  = A.nb_rows();
    int lda = nr;

    if (nr) {
        int incx = 1;
        int incy = 1;
        Blas<T>::symv(&UPLO, &nr, &alpha, A.data(), &lda, in, &incx, &beta, out, &incy);
    }
}

template <typename T>
void add_matrix_vector_product_symmetric(char trans, std::complex<T> alpha, const Matrix<std::complex<T>> &A, const std::complex<T> *in, std::complex<T> beta, std::complex<T> *out, char UPLO, char symmetry) {
    int nr = A.nb_rows();
    if (nr) {
        int lda  = nr;
        int incx = 1;
        int incy = 1;
        if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
            Blas<std::complex<T>>::symv(&UPLO, &nr, &alpha, A.data(), &lda, in, &incx, &beta, out, &incy);
        } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
            Blas<std::complex<T>>::hemv(&UPLO, &nr, &alpha, A.data(), &lda, in, &incx, &beta, out, &incy);
        } else if (symmetry == 'S' && trans == 'C') {
            std::vector<std::complex<T>> conjugate_in(nr);
            std::complex<T> conjugate_alpha = std::conj(alpha);
            std::complex<T> conjugate_beta  = std::conj(beta);
            std::transform(in, in + nr, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
            std::transform(out, out + nr, out, [](const std::complex<T> &c) { return std::conj(c); });
            Blas<std::complex<T>>::symv(&UPLO, &nr, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
            std::transform(out, out + nr, out, [](const std::complex<T> &c) { return std::conj(c); });
        } else if (symmetry == 'H' && trans == 'T') {
            std::vector<std::complex<T>> conjugate_in(nr);
            std::complex<T> conjugate_alpha = std::conj(alpha);
            std::complex<T> conjugate_beta  = std::conj(beta);
            std::transform(in, in + nr, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
            std::transform(out, out + nr, out, [](const std::complex<T> &c) { return std::conj(c); });
            Blas<std::complex<T>>::hemv(&UPLO, &nr, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
            std::transform(out, out + nr, out, [](const std::complex<T> &c) { return std::conj(c); });

        } else {
            htool::Logger::get_instance().log(LogLevel::ERROR, "Invalid arguments for add_vector_product_symmetric: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
            // throw std::invalid_argument("[Htool error] Invalid arguments for add_vector_product_symmetric");               // LCOV_EXCL_LINE
        }
    }
}

} // namespace htool
#endif
