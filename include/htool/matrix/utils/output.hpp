#ifndef HTOOL_MATRIX_UTILS_OUTPUT_HPP
#define HTOOL_MATRIX_UTILS_OUTPUT_HPP

#include "../../misc/logger.hpp"
#include "modifiers.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace htool {

template <typename Mat>
void print(const Mat &mat, std::ostream &os, const std::string &delimiter) {
    using T  = typename Mat::value_type;
    int rows = mat.nb_rows();

    if (mat.nb_cols() > 0) {
        for (int i = 0; i < rows; i++) {
            std::vector<T> row = get_row(mat, i);
            std::copy(row.begin(), row.end() - 1, std::ostream_iterator<T>(os, delimiter.c_str()));
            os << row.back();
            os << '\n';
        }
    }
}

template <typename Mat>
void csv_save(const Mat &mat, const std::string &file, const std::string &delimiter = ",") {
    std::ofstream os(file);
    if (!os) {
        htool::Logger::get_instance().log(LogLevel::WARNING, "Cannot create file " + file); // LCOV_EXCL_LINE
    }

    print(mat, os, delimiter);

    os.close();
}

template <typename Mat>
void matrix_to_bytes(const Mat &mat, const std::string &file) {
    using T = typename Mat::value_type;
    std::ofstream out(file, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!out) {
        htool::Logger::get_instance().log(LogLevel::WARNING, "Cannot open file " + file); // LCOV_EXCL_LINE
    }
    int rows = mat.nb_rows();
    int cols = mat.nb_cols();
    out.write(reinterpret_cast<const char *>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char *>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char *>(mat.data()), rows * cols * sizeof(T));

    out.close();
}

template <typename T>
void bytes_to_matrix(const std::string &file, Matrix<T> &mat) {
    std::ifstream in(file, std::ios::in | std::ios::binary);

    if (!in) {
        htool::Logger::get_instance().log(LogLevel::WARNING, "Cannot open file " + file); // LCOV_EXCL_LINE
    }

    int rows = 0, cols = 0;
    in.read(reinterpret_cast<char *>(&rows), sizeof(int));
    in.read(reinterpret_cast<char *>(&cols), sizeof(int));
    T *new_data = new T[rows * cols];
    mat.assign(rows, cols, new_data, true);
    in.read(reinterpret_cast<char *>(&(new_data[0])), rows * cols * sizeof(T));

    in.close();
}

} // namespace htool
#endif
