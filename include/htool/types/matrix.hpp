#ifndef HTOOL_MATRIX_HPP
#define HTOOL_MATRIX_HPP

#include "../misc/misc.hpp"
#include "../types/virtual_generator.hpp"
#include "../wrappers/wrapper_blas.hpp"
#include "vector.hpp"
#include <cassert>
#include <iterator>

namespace htool {

//=================================================================//
//                         CLASS MATRIX
//*****************************************************************//

template <typename T>
class IMatrix {
  protected:
    // Data members
    int nr;
    int nc;

  public:
    IMatrix(int nr0, int nc0) : nr(nr0), nc(nc0) {}

    int nb_rows() const { return nr; }
    int nb_cols() const { return nc; }

    virtual int get_offset_i() const = 0;
    virtual int get_offset_j() const = 0;

    virtual void add_mvprod_row_major(const T *const in, T *const out, const int &mu, char transb = 'T', char op = 'N') const = 0;

    virtual ~IMatrix(){};
};

template <typename T>
class Matrix : public IMatrix<T> {

  protected:
    T *mat;

  public:
    Matrix() : IMatrix<T>(0, 0), mat(nullptr) {}
    Matrix(const int &nbr, const int &nbc) : IMatrix<T>(nbr, nbc) {
        this->mat = new T[nbr * nbc];
        std::fill_n(this->mat, nbr * nbc, 0);
    }
    Matrix(const Matrix &rhs) : IMatrix<T>(rhs.nb_rows(), rhs.nb_cols()) {
        mat = new T[rhs.nb_rows() * rhs.nb_cols()]();

        std::copy_n(rhs.mat, rhs.nb_rows() * rhs.nb_cols(), mat);
    }
    Matrix &operator=(const Matrix &rhs) {
        if (&rhs == this) {
            return *this;
        }
        if (this->nr * this->nc == rhs.nb_cols() * rhs.nb_rows()) {
            std::copy_n(rhs.mat, this->nr * this->nc, mat);
            this->nr = rhs.nr;
            this->nc = rhs.nc;
        } else {
            this->nr = rhs.nr;
            this->nc = rhs.nc;
            delete[] mat;
            mat = new T[this->nr * this->nc]();
            std::copy_n(rhs.mat, this->nr * this->nc, mat);
        }
        return *this;
    }
    Matrix(Matrix &&rhs) : IMatrix<T>(rhs.nb_rows(), rhs.nb_cols()), mat(rhs.mat) {
        rhs.mat = nullptr;
    }

    Matrix &operator=(Matrix &&rhs) {
        if (this != &rhs) {
            delete[] this->mat;
            this->nr  = rhs.nr;
            this->nc  = rhs.nc;
            this->mat = rhs.mat;
            rhs.mat   = nullptr;
        }
        return *this;
    }

    ~Matrix() {
        if (mat != nullptr)
            delete[] mat;
    }

    void operator=(const T &z) {
        if (this->nr * this->nc > 0)
            std::fill_n(this->mat, this->nr * this->nc, z);
    }
    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the entries
    are allowed.
    */
    T &operator()(const int &j, const int &k) {
        return this->mat[j + k * this->nr];
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A(j,k)_ returns the entry of _A_ located
    jth row and kth column. Modification of the
    entries are forbidden.
    */
    const T &operator()(const int &j, const int &k) const {
        return this->mat[j + k * this->nr];
    }

    //! ### Access operator
    /*!
    */

    T *data() { return this->mat; }
    T *data() const { return this->mat; }

    void assign(int nr, int nc, T *ptr) {
        if (this->nr * this->nc > 0)
            delete[] this->mat;

        this->nr = nr;
        this->nc = nc;

        this->mat = ptr;
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_stridedslice(i,j,k)_ returns the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_. Modification forbidden
    */

    std::vector<T> get_stridedslice(int start, int length, int stride) const {
        std::vector<T> result;
        result.reserve(length);
        const T *pos = &mat[start];
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
        return this->get_stridedslice(row, this->nc, this->nr);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_col(j)_ returns the jth col of _A_.
    */

    std::vector<T> get_col(int col) const {
        return this->get_stridedslice(col * this->nr, this->nr, 1);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_stridedslice(i,j,k,a)_ puts a in the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_.
    */

    void set_stridedslice(int start, int length, int stride, const std::vector<T> &a) {
        assert(length == a.size());
        T *pos = &mat[start];
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
        set_stridedslice(row, this->nc, this->nr, a);
    }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.set_row(i,a)_ puts a in the row of _A_.
    */
    void set_col(int col, const std::vector<T> &a) {
        set_stridedslice(col * this->nr, this->nr, 1, a);
    }

    void set_size(int nr0, int nc0) {
        this->nr = nr0;
        this->nc = nc0;
    }

    // //! ### Modifies the size of the matrix
    // /*!
    // Changes the size of the matrix so that
    // the number of rows is set to _nbr_ and
    // the number of columns is set to _nbc_.
    // */
    // void resize(const int nbr, const int nbc, T value = 0) {
    //     this->mat.resize(nbr * nbc, value);
    //     this->nr = nbr;
    //     this->nc = nbc;
    // }

    //! ### Matrix-scalar product
    /*!
    */

    friend Matrix
    operator*(const Matrix &A, const T &a) {
        Matrix R(A.nr, A.nc);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
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
        assert(this->nr == A.nr && this->nc == A.nc);
        Matrix R(A.nr, A.nc);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
                R(i, j) = this->mat[i + j * this->nr] + A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix -
    /*!
    */

    Matrix operator-(const Matrix &A) const {
        assert(this->nr == A.nr && this->nc == A.nc);
        Matrix R(A.nr, A.nc);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
                R(i, j) = this->mat[i + j * this->nr] - A(i, j);
            }
        }
        return R;
    }

    //! ### Matrix-std::vector product
    /*!
    */

    std::vector<T> operator*(const std::vector<T> &rhs) const {
        std::vector<T> lhs(this->nr);
        this->mvprod(rhs.data(), lhs.data(), 1);
        return lhs;
    }

    //! ### Matrix-Matrix product
    /*!
    */
    Matrix operator*(const Matrix &B) const {
        assert(this->nc == B.nr);
        Matrix R(this->nr, B.nc);
        this->mvprod(&(B.mat[0]), &(R.mat[0]), B.nc);
        return R;
    }

    //! ### Interface with blas gemm
    /*!
    */
    void mvprod(const T *const in, T *const out, const int &mu = 1) const {
        int nr  = this->nr;
        int nc  = this->nc;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            char n   = 'N';
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&n, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
        } else {
            char transa = 'N';
            char transb = 'N';
            int M       = nr;
            int N       = mu;
            int K       = nc;
            int ldb     = nc;
            int ldc     = nr;
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
        }
    }

    //! ### Special mvprod
    /*!
    */
    void mvprod_row_major(const T *const in, T *const out, const int &mu, char transb, char op = 'N') const {
        int nr  = this->nr;
        int nc  = this->nc;
        T alpha = 1;
        T beta  = 0;
        int lda = nr;

        if (mu == 1) {
            int incx = 1;
            int incy = 1;
            Blas<T>::gemv(&op, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
        } else {
            int lda     = mu;
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

            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, mat, &ldb, &beta, out, &ldc);
        }
    }

    //! ### Special add_mvprod
    /*!
    */
    void add_mvprod_row_major(const T *const in, T *const out, const int &mu, char transb, char op = 'N') const {
        int nr  = this->nr;
        int nc  = this->nc;
        T alpha = 1;
        T beta  = 1;

        if (nr && nc) {
            if (mu == 1) {
                int lda  = nr;
                int incx = 1;
                int incy = 1;
                Blas<T>::gemv(&op, &nr, &nc, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
            } else {
                int lda     = mu;
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

                Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, mat, &ldb, &beta, out, &ldc);
            }
        }
    }

    // see https://stackoverflow.com/questions/6972368/stdenable-if-to-conditionally-compile-a-member-function for why  Q template parameter
    template <typename Q = T, typename std::enable_if<!is_complex_t<Q>::value, int>::type = 0>
    void add_mvprod_row_major_sym(const T *const in, T *const out, const int &mu, char UPLO, char) const {
        int nr  = this->nr;
        T alpha = 1;
        T beta  = 1;

        if (nr) {
            if (mu == 1) {
                int lda  = nr;
                int incx = 1;
                int incy = 1;
                Blas<T>::symv(&UPLO, &nr, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
            } else {
                int lda   = nr;
                char side = 'R';
                int M     = mu;
                int N     = nr;
                int ldb   = mu;
                int ldc   = mu;

                Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
            }
        }
    }

    template <typename Q = T, typename std::enable_if<is_complex_t<Q>::value, int>::type = 0>
    void add_mvprod_row_major_sym(const T *const in, T *const out, const int &mu, char UPLO, char symmetry) const {
        int nr  = this->nr;
        T alpha = 1;
        T beta  = 1;

        if (nr) {
            if (mu == 1) {
                int lda  = nr;
                int incx = 1;
                int incy = 1;
                if (symmetry == 'S') {
                    Blas<T>::symv(&UPLO, &nr, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
                } else if (symmetry == 'H') {
                    Blas<T>::hemv(&UPLO, &nr, &alpha, mat, &lda, in, &incx, &beta, out, &incy);
                } else {
                    throw std::invalid_argument("[Htool error] Invalid arguments for add_mvprod_row_major_sym"); // LCOV_EXCL_LINE
                }

            } else {
                int lda   = nr;
                char side = 'R';
                int M     = mu;
                int N     = nr;
                int ldb   = mu;
                int ldc   = mu;

                if (symmetry == 'S') {
                    Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
                } else if (symmetry == 'H') {
                    Blas<T>::hemm(&side, &UPLO, &M, &N, &alpha, mat, &lda, in, &ldb, &beta, out, &ldc);
                } else {
                    throw std::invalid_argument("[Htool error] Invalid arguments for add_mvprod_row_major_sym"); // LCOV_EXCL_LINE
                }
            }
        }
    }

    int get_offset_i() const { return 0; }
    int get_offset_j() const { return 0; }

    //! ### Looking for the entry of maximal modulus
    /*!
    Returns the number of row and column of the entry
    of maximal modulus in the matrix _A_.
    */
    friend std::pair<int, int> argmax(const Matrix<T> &M) {
        int p = std::max_element(M.data(), M.data() + M.nb_cols() * M.nb_rows(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - M.data();
        return std::pair<int, int>(p % M.nr, (int)p / M.nr);
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
        int rows = this->nr;
        int cols = this->nc;
        out.write((char *)(&rows), sizeof(int));
        out.write((char *)(&cols), sizeof(int));
        out.write((char *)mat, rows * cols * sizeof(T));

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
        if (this->nr != 0 && this->nc != 0)
            delete[] mat;
        mat      = new T[rows * cols];
        this->nr = rows;
        this->nc = cols;
        in.read((char *)&(mat[0]), rows * cols * sizeof(T));

        in.close();
        return 0;
    }

    int print(std::ostream &os, const std::string &delimiter) {

        int rows = this->nr;
        for (int i = 0; i < rows; i++) {
            std::vector<T> row = this->get_row(i);
            std::copy(row.begin(), row.end() - 1, std::ostream_iterator<T>(os, delimiter.c_str()));
            os << row.back();
            os << '\n';
        }
        return 0;
    }

    int csv_save(const std::string &file, const std::string &delimiter = ",") {
        std::ofstream os(file);
        try {
            if (!os) {
                throw std::string("Cannot create file " + file);
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

//================================//
//      CLASSE SOUS-MATRICE       //
//================================//
template <typename T>
class SubMatrix : public Matrix<T> {
    // TODO: remove ir and ic
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;

  public:
    // C style
    SubMatrix(int M, int N, const int *const rows, const int *const cols, int offset_i0 = 0, int offset_j0 = 0) : Matrix<T>(M, N), ir(rows, rows + M), ic(cols, cols + N), offset_i(offset_i0), offset_j(offset_j0) {
    }

    SubMatrix(const VirtualGenerator<T> &mat0, int M, int N, const int *const rows, const int *const cols, int offset_i0 = 0, int offset_j0 = 0) : SubMatrix(M, N, rows, cols, offset_i0, offset_j0) {
        mat0.copy_submatrix(M, N, rows, cols, this->mat);
    }

    // C++ style
    SubMatrix(std::vector<int>::const_iterator first_ir, std::vector<int>::const_iterator last_ir, std::vector<int>::const_iterator first_ic, std::vector<int>::const_iterator last_ic, int offset_i0 = 0, int offset_j0 = 0) : Matrix<T>(std::distance(first_ir, last_ir), std::distance(first_ic, last_ic)), ir(first_ir, last_ir), ic(first_ic, last_ic), offset_i(offset_i0), offset_j(offset_j0) {}

    SubMatrix(const std::vector<int> &ir0, const std::vector<int> &ic0) : SubMatrix(ir0.begin(), ir0.end(), ic0.begin(), ic0.end(), 0, 0) {}

    SubMatrix(const std::vector<int> &ir0, const std::vector<int> &ic0, const int &offset_i0, const int &offset_j0) : SubMatrix(ir0.begin(), ir0.end(), ic0.begin(), ic0.end(), offset_i0, offset_j0) {}

    SubMatrix(const SubMatrix &m) {
        this->mat      = m.mat;
        this->ir       = m.ir;
        this->ic       = m.ic;
        this->offset_i = m.offset_i;
        this->offset_j = m.offset_j;
        this->nr       = m.nr;
        this->nc       = m.nc;
    }

    // Mostly same operators as in Matrix, need CRTP to factorize
    // Operators
    SubMatrix operator+(const SubMatrix &A) {
        assert(this->nr == A.nr && this->nc == A.nc);
        SubMatrix R(*this);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
                R(i, j) += A(i, j);
            }
        }
        return R;
    }
    friend SubMatrix operator*(const SubMatrix &A, const T &a) {
        SubMatrix R(A);
        for (int i = 0; i < A.nr; i++) {
            for (int j = 0; j < A.nc; j++) {
                R(i, j) = A(i, j) * a;
            }
        }
        return R;
    }
    friend SubMatrix operator*(const T &a, const SubMatrix &A) {
        return A * a;
    }
    // Getters
    std::vector<int> get_ir() const { return this->ir; }
    std::vector<int> get_ic() const { return this->ic; }
    const int *data_ir() const { return this->ir.data(); }
    const int *data_ic() const { return this->ic.data(); }
    int get_offset_i() const { return this->offset_i; }
    int get_offset_j() const { return this->offset_j; }
    void set_offset_i(int offset) { this->offset_i = offset; }
    void set_offset_j(int offset) { this->offset_j = offset; }
};
} // namespace htool

#endif
