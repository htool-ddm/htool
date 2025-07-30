#ifndef HTOOL_BASIC_TYPES_MATRIX_HPP
#define HTOOL_BASIC_TYPES_MATRIX_HPP

#include "../misc/misc.hpp"             // for conj_if_complex, is_complex_t
#include "../wrappers/wrapper_blas.hpp" // for Blas
#include <algorithm>                    // for transform, copy_n, fill_n
#include <cmath>                        // for sqrt
#include <complex>                      // for conj, complex
#include <fstream>                      // for basic_ofstream, basic_ostream
#include <iostream>                     // for cout, cerr
#include <iterator>                     // for ostream_iterator
#include <string>                       // for operator+, basic_string, all...
#include <type_traits>                  // for enable_if
#include <utility>                      // for pair
#include <vector>                       // for vector

namespace htool {

template <typename T>
class Matrix {

  protected:
    int m_number_of_rows, m_number_of_cols;
    T *m_data;
    bool m_is_owning_data;
    std::vector<int> m_pivots;

  public:
    using element_type = T;
    using value_type   = std::remove_cv_t<T>;

    Matrix() : m_number_of_rows(0), m_number_of_cols(0), m_data(nullptr), m_is_owning_data(true), m_pivots(0) {}
    Matrix(int nbr, int nbc, T value = 0) : m_number_of_rows(nbr), m_number_of_cols(nbc), m_is_owning_data(true), m_pivots(0) {
        std::size_t size = std::size_t(nbr) * std::size_t(nbc);
        m_data           = size != 0 ? new T[size] : nullptr;
        std::fill_n(m_data, std::size_t(nbr) * std::size_t(nbc), value);
    }
    Matrix(const Matrix &rhs) : m_number_of_rows(rhs.m_number_of_rows), m_number_of_cols(rhs.m_number_of_cols), m_is_owning_data(true), m_pivots(rhs.m_pivots) {
        std::size_t size = std::size_t(rhs.m_number_of_rows) * std::size_t(rhs.m_number_of_cols);
        m_data           = size != 0 ? new T[rhs.m_number_of_rows * rhs.m_number_of_cols]() : nullptr;

        std::copy_n(rhs.m_data, rhs.m_number_of_rows * rhs.m_number_of_cols, m_data);
    }
    Matrix &operator=(const Matrix &rhs) {
        if (&rhs == this) {
            return *this;
        }
        std::size_t size     = std::size_t(m_number_of_rows) * std::size_t(m_number_of_cols);
        std::size_t size_rhs = std::size_t(rhs.m_number_of_rows) * std::size_t(rhs.m_number_of_cols);
        if (size == size_rhs) {
            std::copy_n(rhs.m_data, m_number_of_rows * m_number_of_cols, m_data);
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
        } else {
            m_number_of_rows = rhs.m_number_of_rows;
            m_number_of_cols = rhs.m_number_of_cols;
            if (m_is_owning_data)
                delete[] m_data;
            m_data = size_rhs != 0 ? new T[m_number_of_rows * m_number_of_cols]() : nullptr;
            std::copy_n(rhs.m_data, m_number_of_rows * m_number_of_cols, m_data);
            m_is_owning_data = true;
        }
        m_pivots = rhs.m_pivots;
        return *this;
    }
    Matrix(Matrix &&rhs) : m_number_of_rows(rhs.m_number_of_rows), m_number_of_cols(rhs.m_number_of_cols), m_data(rhs.m_data), m_is_owning_data(rhs.m_is_owning_data), m_pivots(rhs.m_pivots) {
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
            m_pivots         = rhs.m_pivots;
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

    std::vector<int> &get_pivots() { return m_pivots; }
    const std::vector<int> &get_pivots() const { return m_pivots; }
    bool is_owning_data() const { return m_is_owning_data; }

    //! ### Modifies the size of the matrix
    /*!
    Changes the size of the matrix so that
    the number of rows is set to _nbr_ and
    the number of columns is set to _nbc_.
    */
    void resize(int nbr, int nbc, T value = 0) {
        if (m_data != nullptr and m_is_owning_data and m_number_of_rows * m_number_of_cols != nbr * nbc) {
            delete[] m_data;
            m_data           = nullptr;
            m_data           = new T[nbr * nbc];
            m_is_owning_data = true;
        } else if (m_number_of_rows * m_number_of_cols != nbr * nbc) {
            m_data           = new T[nbr * nbc];
            m_is_owning_data = true;
        } else if (!m_is_owning_data and m_number_of_rows * m_number_of_cols == nbr * nbc) {
            m_data           = new T[nbr * nbc];
            m_is_owning_data = true;
        }

        m_number_of_rows = nbr;
        m_number_of_cols = nbc;
        std::fill_n(m_data, nbr * nbc, value);
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
        std::vector<T> lhs(m_number_of_rows, 0);
        for (int i = 0; i < m_number_of_rows; i++) {
            for (int k = 0; k < m_number_of_cols; k++) {
                lhs[i] += m_data[i + k * m_number_of_rows] * rhs[k];
            }
        }
        return lhs;
    }

    //! ### Matrix-Matrix product
    /*!
     */
    Matrix operator*(const Matrix &B) const {
        Matrix R(m_number_of_rows, B.m_number_of_cols);
        for (int i = 0; i < m_number_of_rows; i++) {
            for (int j = 0; j < B.m_number_of_cols; j++) {
                for (int k = 0; k < m_number_of_cols; k++) {
                    R(i, j) += m_data[i + k * m_number_of_rows] * B(k, j);
                }
            }
        }
        return R;
    }
};

} // namespace htool

#endif
