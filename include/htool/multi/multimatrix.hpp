#ifndef HTOOL_MultiMatrix_HPP
#define HTOOL_MultiMatrix_HPP

#include "virtual_multi_generator.hpp"
#include <vector>

namespace htool {

// template <typename T>
// class MultiIMatrix {
//   protected:
//     // Data members
//     int nr;
//     int nc;
//     int nm;

//   public:
//     MultiIMatrix(int nr0, int nc0, int nm0) : nr(nr0), nc(nc0), nm(nm0) {}

//     int nb_rows() const { return nr; }
//     int nb_cols() const { return nc; }
//     int nb_matrix() const { return nm; }

//     virtual ~MultiIMatrix(){};
// };

template <typename T>
class MultiSubMatrix {
  private:
    std::vector<SubMatrix<T>> SubMatrices;
    int nr;
    int nc;
    int nm;
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;

  public:
    // C style
    MultiSubMatrix(int M, int N, const int *const rows, const int *const cols, int nm0, int offset_i0 = 0, int offset_j0 = 0) : nr(M), nc(N), ir(rows, rows + M), nm(nm0), ic(cols, cols + N), offset_i(offset_i0), offset_j(offset_j0) {
    }

    MultiSubMatrix(const VirtualMultiGenerator<T> &mat0, int M, int N, const int *const rows, const int *const cols, int nm0, int offset_i0 = 0, int offset_j0 = 0) : MultiSubMatrix(M, N, rows, cols, offset_i0, offset_j0) {
        mat0.copy_submatrices(M, N, rows, cols, this->mat);
    }

    // C++ style
    MultiSubMatrix(const std::vector<int> &ir0, const std::vector<int> &ic0, int nm0) : nr(ir0.size()), nr(ic0.size()), nm(nm0), SubMatrices(nm, SubMatrix<T>(ir0.size(), ic0.size(), ir0.data(), ic0.data())), ir(ir0), ic(ic0), offset_i(0), offset_j(0) {
    }

    MultiSubMatrix(const VirtualMultiGenerator<T> &mat0, const std::vector<int> &ir0, const std::vector<int> &ic0) : MultiSubMatrix(ir0, ic0, mat0.nb_matrix()) {
        mat0.copy_submatrices(ir0, ic0);
    }

    SubMatrix<T> &operator[](int j) { return SubMatrices[j]; };
    const SubMatrix<T> &operator[](int j) const { return SubMatrices[j]; };
};
} // namespace htool

#endif