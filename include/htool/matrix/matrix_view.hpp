#ifndef HTOOL_BASIC_TYPES_MATRIX_VIEW_HPP
#define HTOOL_BASIC_TYPES_MATRIX_VIEW_HPP

#include "../misc/logger.hpp" // for Logger, LogLevel
#include "../misc/misc.hpp"   // for conj_if_complex, is_complex_t
#include "matrix.hpp"         // for Matrix
#include <algorithm>          // for transform, copy_n, fill_n
#include <cmath>              // for sqrt
#include <complex>            // for conj, complex
#include <fstream>            // for basic_ofstream, basic_ostream
#include <iostream>           // for cout, cerr
#include <iterator>           // for ostream_iterator
#include <string>             // for operator+, basic_string, all...
#include <type_traits>        // for enable_if
#include <utility>            // for pair
#include <vector>             // for vector
namespace htool {

template <typename T>
class MatrixView {

  protected:
    int m_number_of_rows, m_number_of_cols;
    T *m_data;

  public:
    using element_type = T;
    using value_type   = std::remove_cv_t<T>;
    MatrixView() : m_number_of_rows(0), m_number_of_cols(0), m_data(nullptr) {}
    MatrixView(int nb_rows, int nb_cols, T *data) : m_number_of_rows(nb_rows), m_number_of_cols(nb_cols), m_data(data) {}

    // Implicit conversion from Matrix<U> to MatrixView<T> if T = const U
    template <typename U, typename = std::enable_if_t<std::is_same<T, const U>::value>>
    MatrixView(const Matrix<U> &mat) : m_number_of_rows(mat.nb_rows()), m_number_of_cols(mat.nb_cols()), m_data(mat.data()) {}

    // Implicit conversion from Matrix<U> to MatrixView<T> if T =  U
    template <typename U, typename = std::enable_if_t<std::is_same<U, T>::value>>
    MatrixView(Matrix<U> &mat) : m_number_of_rows(mat.nb_rows()), m_number_of_cols(mat.nb_cols()), m_data(mat.data()) {}

    void assign(int number_of_rows, int number_of_cols, T *ptr) {
        m_number_of_rows = number_of_rows;
        m_number_of_cols = number_of_cols;
        m_data           = ptr;
    }

    void operator=(const T &z) {
        if (m_number_of_rows * m_number_of_cols > 0)
            std::fill_n(m_data, m_number_of_rows * m_number_of_cols, z);
    }
    T &operator()(const int &j, const int &k) {
        return m_data[j + k * m_number_of_rows];
    }
    const T &operator()(const int &j, const int &k) const {
        return m_data[j + k * m_number_of_rows];
    }
    T *data() {
        return m_data;
    }
    const T *data() const {
        return m_data;
    }
    int nb_cols() const { return m_number_of_cols; }
    int nb_rows() const { return m_number_of_rows; }
};

} // namespace htool

#endif
