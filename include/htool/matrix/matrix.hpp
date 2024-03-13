#ifndef HTOOL_BASIC_TYPES_MATRIX_HPP
#define HTOOL_BASIC_TYPES_MATRIX_HPP

#include "../basic_types/vector.hpp"
#include "../misc/logger.hpp"
#include "../misc/misc.hpp"
#include "../wrappers/wrapper_blas.hpp"
#include <cassert>
#include <functional>
#include <iterator>

namespace htool {

template <typename T>
class Matrix {

  protected:
    int m_number_of_rows, m_number_of_cols;
    T *m_data;
    bool m_is_owning_data;

  public:
    Matrix() : m_number_of_rows(0), m_number_of_cols(0), m_data(nullptr), m_is_owning_data(true) {}
    Matrix(int nbr, int nbc, T value = 0) : m_number_of_rows(nbr), m_number_of_cols(nbc), m_is_owning_data(true) {
        m_data = new T[nbr * nbc];
        std::fill_n(m_data, nbr * nbc, value);
    }
    Matrix(const Matrix &rhs) : m_number_of_rows(rhs.m_number_of_rows), m_number_of_cols(rhs.m_number_of_cols), m_is_owning_data(true) {
        m_data = new T[rhs.m_number_of_rows * rhs.m_number_of_cols]();

        std::copy_n(rhs.m_data, rhs.m_number_of_rows * rhs.m_number_of_cols, m_data);
    }
    Matrix &operator=(const Matrix &rhs) {
        if (&rhs == this) {
            return *this;
        }
        if (m_number_of_rows * m_number_of_cols == rhs.m_number_of_cols * rhs.m_number_of_rows) {
            std::copy_n(rhs.m_data, m_number_of_rows * m_number_of_cols, m_data);
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            // m_is_owning_data = true;
        } else {
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            if (m_is_owning_data)
                delete[] m_data;
            m_data = new T[m_number_of_rows * m_number_of_cols]();
            std::copy_n(rhs.m_data, m_number_of_rows * m_number_of_cols, m_data);
            m_is_owning_data = true;
        }
        return *this;
    }
    Matrix(Matrix &&rhs) : m_number_of_rows(rhs.m_number_of_rows), m_number_of_cols(rhs.m_number_of_cols), m_data(rhs.m_data), m_is_owning_data(rhs.m_is_owning_data) {
        rhs.m_data = nullptr;
    }

    Matrix &operator=(Matrix &&rhs) {
        if (this != &rhs) {
            if (m_is_owning_data)
                delete[] m_data;
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            m_data           = rhs.m_data;
            m_is_owning_data = rhs.m_is_owning_data;
            rhs.m_data       = nullptr;
        }
        return *this;
    }

    ~Matrix() {
        if (m_data != nullptr && m_is_owning_data)
            delete[] m_data;
    }

    void operator=(const T &z) {
        if (m_number_of_rows * m_number_of_cols > 0)
            std::fill_n(m_data, m_number_of_rows * m_number_of_cols, z);
    }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the entries
    are allowed.
    */
    T &operator()(const int &j, const int &k) {
        return m_data[j + k * m_number_of_rows];
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the
    entries are forbidden.
    */
    const T &operator()(const int &j, const int &k) const {
        return m_data[j + k * m_number_of_rows];
    }

    //! ### Access operator
    /*!
     */

    T *data() {
        return m_data;
    }
    const T *data() const {
        return m_data;
    }

    T *release() {
        T *result = m_data;
        m_data    = nullptr;
        return result;
    }

    void assign(int number_of_rows, int number_of_cols, T *ptr, bool owning_data) {
        if (m_number_of_rows * m_number_of_cols > 0 && m_is_owning_data)
            delete[] m_data;

        m_number_of_rows = number_of_rows;
        m_number_of_cols = number_of_cols;
        m_data           = ptr;
        m_is_owning_data = owning_data;
    }

    int nb_cols() const { return m_number_of_cols; }
    int nb_rows() const { return m_number_of_rows; }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_stridedslice(i,j,k)_ returns the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_. Modification forbidden
    */

    std::vector<T> get_stridedslice(int start, int length, int stride) const {
        std::vector<T> result;
        result.reserve(length);
        const T *pos = &m_data[start];
        for (int i = 0; i < length; i++) {
            result.push_back(*pos);
            pos += stride;
        }
        return result;
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_row(j)_ returns the jth row of _A_.
    */

    std::vector<T> get_row(int row) const {
        return this->get_stridedslice(row, m_number_of_cols, m_number_of_rows);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_col(j)_ returns the jth col of _A_.
    */

    std::vector<T> get_col(int col) const {
        return this->get_stridedslice(col * m_number_of_rows, m_number_of_rows, 1);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_stridedslice(i,j,k,a)_ puts a in the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_.
    */

    void set_stridedslice(int start, int length, int stride, const std::vector<T> &a) {
        assert(length == a.size());
        T *pos = &m_data[start];
        for (int i = 0; i < length; i++) {
            *pos = a[i];
            pos += stride;
        }
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the ith row of _A_.
    */
    void set_row(int row, const std::vector<T> &a) {
        set_stridedslice(row, m_number_of_cols, m_number_of_rows, a);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the row of _A_.
    */
    void set_col(int col, const std::vector<T> &a) {
        set_stridedslice(col * m_number_of_rows, m_number_of_rows, 1, a);
    }

    void set_size(int nr0, int nc0) {
        m_number_of_rows = nr0;
        m_number_of_cols = nc0;
    }

    //! ### Modifies the size of the matrix
    /*!
    Changes the size of the matrix so that
    the number of rows is set to _nbr_ and
    the number of columns is set to _nbc_.
    */
    void resize(int nbr, int nbc, T value = 0) {
        if (m_data != nullptr and m_is_owning_data and m_number_of_rows * m_number_of_cols != nbr * nbc) {
            delete[] m_data;
            m_data = nullptr;
            m_data = new T[nbr * nbc];
        } else if (m_number_of_rows * m_number_of_cols != nbr * nbc) {
            m_data = new T[nbr * nbc];
        }

        m_number_of_rows = nbr;
        m_number_of_cols = nbc;
        std::fill_n(m_data, nbr * nbc, value);
        m_is_owning_data = true;
    }

    //! ### Matrix-scalar product
    /*!
     */

    friend Matrix
    operator*(const Matrix &A, const T &a) {
        Matrix R(A.m_number_of_rows, A.m_number_of_cols);
        for (int i = 0; i < A.m_number_of_rows; i++) {
            for (int j = 0; j < A.m_number_of_cols; j++) {
                R(i, j) = A(i, j) * a;
            }
        }
        return R;
    }
    friend Matrix operator*(const T &a, const Matrix &A) {
        return A * a;
    }

    //! ### Matrix sum
    /*!
     */

    Matrix operator+(const Matrix &A) const {
        assert(m_number_of_rows == A.m_number_of_rows && m_number_of_cols == A.m_number_of_cols);
        Matrix R(A.m_number_of_rows, A.m_number_of_cols);
        for (int i = 0; i < A.m_number_of_rows; i++) {
            for (int j = 0; j < A.m_number_of_cols; j++) {
                R(i, j) = m_data[i + j * m_number_of_rows] + A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix -
    /*!
     */

    Matrix operator-(const Matrix &A) const {
        assert(m_number_of_rows == A.m_number_of_rows && m_number_of_cols == A.m_number_of_cols);
        Matrix R(A.m_number_of_rows, A.m_number_of_cols);
        for (int i = 0; i < A.m_number_of_rows; i++) {
            for (int j = 0; j < A.m_number_of_cols; j++) {
                R(i, j) = m_data[i + j * m_number_of_rows] - A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix-std::vector product
    /*!
     */

    std::vector<T> operator*(const std::vector<T> &rhs) const {
        std::vector<T> lhs(m_number_of_rows);
        this->mvprod(rhs.data(), lhs.data(), 1);
        return lhs;
    }

    //! ### Matrix-Matrix product
    /*!
     */
    Matrix operator*(const Matrix &B) const {
        assert(m_number_of_cols == B.m_number_of_rows);
        Matrix R(m_number_of_rows, B.m_number_of_cols);
        this->mvprod(&(B.m_data[0]), &(R.m_data[0]), B.m_number_of_cols);
        return R;
    }

    //! ### Interface with blas gemm
    /*!
     */
    void mvprod(const T *in, T *out, const int &mu = 1) const {
        int nr  = m_number_of_rows;
        int nc  = m_number_of_cols;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            char n   = 'N';
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&n, &nr, &nc, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
        } else {
            char transa = 'N';
            char transb = 'N';
            int M       = nr;
            int N       = mu;
            int K       = nc;
            int ldb     = nc;
            int ldc     = nr;
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    void add_vector_product(char trans, T alpha, const T *in, T beta, T *out) const {
        int nr   = m_number_of_rows;
        int nc   = m_number_of_cols;
        int lda  = nr;
        int incx = 1;
        int incy = 1;
        Blas<T>::gemv(&trans, &nr, &nc, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
    }

    void add_matrix_product(char trans, T alpha, const T *in, T beta, T *out, int mu) const {
        int nr      = m_number_of_rows;
        int nc      = m_number_of_cols;
        char transa = trans;
        char transb = 'N';
        int lda     = nr;
        int M       = nr;
        int N       = mu;
        int K       = nc;
        int ldb     = nc;
        int ldc     = nr;
        if (transa != 'N') {
            M   = nc;
            N   = mu;
            K   = nr;
            ldb = nr;
            ldc = nc;
        }

        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
    }

    void add_matrix_product_row_major(char trans, T alpha, const T *in, T beta, T *out, int mu) const {
        int nr      = m_number_of_rows;
        int nc      = m_number_of_cols;
        char transa = 'N';
        char transb = 'T';
        int M       = mu;
        int N       = nr;
        int K       = nc;
        int lda     = mu;
        int ldb     = nr;
        int ldc     = mu;
        if (trans != 'N') {
            transb = 'N';
            N      = nc;
            K      = nr;
        }
        if (trans == 'C' && is_complex<T>()) {
            std::vector<T> conjugate_in(nr * mu);
            T conjugate_alpha = conj_if_complex<T>(alpha);
            T conjugate_beta  = conj_if_complex<T>(beta);
            std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return conj_if_complex<T>(c); });
            conj_if_complex<T>(out, nc * mu);
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &conjugate_alpha, conjugate_in.data(), &lda, m_data, &ldb, &conjugate_beta, out, &ldc);
            conj_if_complex<T>(out, nc * mu);
            return;
        }
        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, m_data, &ldb, &beta, out, &ldc);
    }

    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_vector_product_symmetric(char, T alpha, const T *in, T beta, T *out, char UPLO, char) const {
        int nr  = m_number_of_rows;
        int lda = nr;

        if (nr) {
            int incx = 1;
            int incy = 1;
            Blas<T>::symv(&UPLO, &nr, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_vector_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, char UPLO, char symmetry) const {
        int nr = m_number_of_rows;
        if (nr) {
            int lda  = nr;
            int incx = 1;
            int incy = 1;
            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symv(&UPLO, &nr, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                Blas<T>::hemv(&UPLO, &nr, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
                Blas<T>::symv(&UPLO, &nr, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
            } else if (symmetry == 'H' && trans == 'T') {
                std::vector<T> conjugate_in(nr);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });
                Blas<T>::hemv(&UPLO, &nr, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &incx, &conjugate_beta, out, &incy);
                std::transform(out, out + nr, out, [](const T &c) { return std::conj(c); });

            } else {
                htool::Logger::get_instance().log(LogLevel::ERROR, "Invalid arguments for add_vector_product_symmetric: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] Invalid arguments for add_vector_product_symmetric");               // LCOV_EXCL_LINE
            }
        }
    }

    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric(char, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char) const {
        int nr  = m_number_of_rows;
        int lda = nr;

        if (nr) {
            char side = 'L';
            int M     = nr;
            int N     = mu;
            int ldb   = m_number_of_cols;
            int ldc   = nr;
            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr = m_number_of_rows;

        if (nr) {
            int lda   = nr;
            char side = 'L';
            int M     = nr;
            int N     = mu;
            int ldb   = m_number_of_cols;
            int ldc   = nr;

            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, m_number_of_cols * mu);
                Blas<T>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, m_number_of_cols * mu);
            } else if (symmetry == 'H' && trans == 'T') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                std::transform(out, out + nr * mu, out, [](const T &c) { return std::conj(c); });
                Blas<T>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                std::transform(out, out + nr * mu, out, [](const T &c) { return std::conj(c); });
            } else {
                htool::Logger::get_instance().log(LogLevel::ERROR, "Invalid arguments for add_matrix_product_symmetric: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE

                // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")");                                            // LCOV_EXCL_LINE
            }
        }
    }

    //! ### Special mvprod with row major input and output
    /*!
     */
    void mvprod_row_major(const T *in, T *out, const int &mu, char transb, char op = 'N') const {
        int nr  = m_number_of_rows;
        int nc  = m_number_of_cols;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&op, &nr, &nc, &alpha, m_data, &lda, in, &incx, &beta, out, &incy);
        } else {
            lda         = mu;
            char transa = 'N';
            int M       = mu;
            int N       = nr;
            int K       = nc;
            int ldb     = nr;
            int ldc     = mu;

            if (op == 'T' || op == 'C') {
                transb = 'N';
                N      = nc;
                K      = nr;
            }

            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, m_data, &ldb, &beta, out, &ldc);
        }
    }

    // void add_matrix_product_row_major(T alpha, const T *in, T beta, T *out, const int &mu, char transb, char op = 'N') const {
    //     int nr = this->nr;
    //     int nc = this->nc;

    //     if (nr && nc) {
    //         if (mu == 1) {
    //             int lda  = nr;
    //             int incx = 1;
    //             int incy = 1;
    //             Blas<T>::gemv(&op, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
    //         } else {
    //             int lda     = mu;
    //             char transa = 'N';
    //             int M       = mu;
    //             int N       = nr;
    //             int K       = nc;
    //             int ldb     = nr;
    //             int ldc     = mu;

    //             if (op == 'T' || op == 'C') {
    //                 transb = 'N';
    //                 N      = nc;
    //                 K      = nr;
    //             }

    //             Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, mat, &ldb, &beta, out, &ldc);
    //         }
    //     }
    // }

    // see https://stackoverflow.com/questions/6972368/stdenable-if-to-conditionally-compile-a-member-function for why  Q template parameter
    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric_row_major(char, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char) const {
        int nr = m_number_of_rows;

        if (nr) {
            int lda   = nr;
            char side = 'R';
            int M     = mu;
            int N     = nr;
            int ldb   = mu;
            int ldc   = mu;

            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_matrix_product_symmetric_row_major(char trans, T alpha, const T *in, T beta, T *out, const int &mu, char UPLO, char symmetry) const {
        int nr = m_number_of_rows;

        if (nr) {
            int lda   = nr;
            char side = 'R';
            int M     = mu;
            int N     = nr;
            int ldb   = mu;
            int ldc   = mu;

            if (symmetry == 'S' && (trans == 'N' || trans == 'T')) {
                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'H' && trans == 'T') {
                Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, m_data, &lda, in, &ldb, &beta, out, &ldc);
            } else if (symmetry == 'S' && trans == 'C') {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, m_number_of_cols * mu);
                Blas<T>::symm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, m_number_of_cols * mu);
            } else if (symmetry == 'H' && (trans == 'N' || trans == 'C')) {
                std::vector<T> conjugate_in(nr * mu);
                T conjugate_alpha = std::conj(alpha);
                T conjugate_beta  = std::conj(beta);
                std::transform(in, in + nr * mu, conjugate_in.data(), [](const T &c) { return std::conj(c); });
                conj_if_complex<T>(out, m_number_of_cols * mu);
                Blas<T>::hemm(&side, &UPLO, &M, &N, &conjugate_alpha, m_data, &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
                conj_if_complex<T>(out, m_number_of_cols * mu);
            } else {
                htool::Logger::get_instance().log(LogLevel::ERROR, "Invalid arguments for add_matrix_product_symmetric_row_major: " + std::string(1, trans) + " with " + symmetry + ")\n"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + symmetry + ")"); // LCOV_EXCL_LINE
            }
        }
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Returns the number of row and column of the entry
    of maximal modulus in the matrix _A_.
    */
    friend std::pair<int, int> argmax(const Matrix<T> &M) {
        int p = std::max_element(M.data(), M.data() + M.nb_cols() * M.nb_rows(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - M.data();
        return std::pair<int, int>(p % M.m_number_of_rows, (int)p / M.m_number_of_rows);
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Save a Matrix in a file (bytes)
    */
    int matrix_to_bytes(const std::string &file) {
        std::ofstream out(file, std::ios::out | std::ios::binary | std::ios::trunc);

        if (!out) {
            std::cout << "Cannot open file." << std::endl; // LCOV_EXCL_LINE
            return 1;                                      // LCOV_EXCL_LINE
        }
        int rows = m_number_of_rows;
        int cols = m_number_of_cols;
        out.write((char *)(&rows), sizeof(int));
        out.write((char *)(&cols), sizeof(int));
        out.write((char *)m_data, rows * cols * sizeof(T));

        out.close();
        return 0;
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Load a matrix from a file (bytes)
    */
    int bytes_to_matrix(const std::string &file) {
        std::ifstream in(file, std::ios::in | std::ios::binary);

        if (!in) {
            std::cout << "Cannot open file." << std::endl; // LCOV_EXCL_LINE
            return 1;                                      // LCOV_EXCL_LINE
        }

        int rows = 0, cols = 0;
        in.read((char *)(&rows), sizeof(int));
        in.read((char *)(&cols), sizeof(int));
        if (m_number_of_rows != 0 && m_number_of_cols != 0 && m_is_owning_data)
            delete[] m_data;
        m_data           = new T[rows * cols];
        m_number_of_rows = rows;
        m_number_of_cols = cols;
        m_is_owning_data = true;
        in.read((char *)&(m_data[0]), rows * cols * sizeof(T));

        in.close();
        return 0;
    }

    int print(std::ostream &os, const std::string &delimiter) const {
        int rows = m_number_of_rows;

        if (m_number_of_cols > 0) {
            for (int i = 0; i < rows; i++) {
                std::vector<T> row = this->get_row(i);
                std::copy(row.begin(), row.end() - 1, std::ostream_iterator<T>(os, delimiter.c_str()));
                os << row.back();
                os << '\n';
            }
        }
        return 0;
    }

    int csv_save(const std::string &file, const std::string &delimiter = ",") const {
        std::ofstream os(file);
        try {
            if (!os) {
                htool::Logger::get_instance().log(LogLevel::WARNING, "Cannot create file " + file); // LCOV_EXCL_LINE
                // throw std::string("Cannot create file " + file);
            }
        } catch (std::string const &error) {
            std::cerr << error << std::endl;
            return 1;
        }

        this->print(os, delimiter);

        os.close();
        return 0;
    }
};

//! ### Computation of the Frobenius norm
/*!
Computes the Frobenius norm of the input matrix _A_.
*/
template <typename T>
double normFrob(const Matrix<T> &A) {
    double norm = 0;
    for (int j = 0; j < A.nb_rows(); j++) {
        for (int k = 0; k < A.nb_cols(); k++) {
            norm = norm + std::pow(std::abs(A(j, k)), 2);
        }
    }
    return sqrt(norm);
}

} // namespace htool

#endif
