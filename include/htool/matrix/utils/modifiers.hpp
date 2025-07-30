#ifndef HTOOL_MATRIX_UTILS_MODIFIERS_HPP
#define HTOOL_MATRIX_UTILS_MODIFIERS_HPP

#include <vector>

namespace htool {

template <typename Mat>
auto get_stridedslice(const Mat &mat, int start, int length, int stride) {
    using T = typename Mat::value_type;
    std::vector<T> result;
    result.reserve(length);
    const T *pos = &mat.data()[start];
    for (int i = 0; i < length; i++) {
        result.push_back(*pos);
        pos += stride;
    }
    return result;
}

template <typename Mat>
void set_stridedslice(Mat &mat, int start, int length, int stride, const std::vector<typename Mat::value_type> &a) {
    using T = typename Mat::value_type;
    T *pos  = &mat.data()[start];
    for (int i = 0; i < length; i++) {
        *pos = a[i];
        pos += stride;
    }
}
template <typename Mat>
auto get_row(const Mat &mat, int row) {
    return get_stridedslice(mat, row, mat.nb_cols(), mat.nb_rows());
}

template <typename Mat>
auto get_col(const Mat &mat, int col) {
    return get_stridedslice(mat, col * mat.nb_rows(), mat.nb_rows(), 1);
}

template <typename Mat>
void set_row(Mat &mat, int row, const std::vector<typename Mat::value_type> &a) {
    set_stridedslice(mat, row, mat.nb_cols(), mat.nb_rows(), a);
}
template <typename Mat>
void set_col(Mat &mat, int col, const std::vector<typename Mat::value_type> &a) {
    set_stridedslice(mat, col * mat.nb_rows(), mat.nb_rows(), 1, a);
}

} // namespace htool
#endif
