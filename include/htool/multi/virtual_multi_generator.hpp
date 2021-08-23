#ifndef HTOOL_MULTI_GENERATPR_HPP
#define HTOOL_MULTI_GENERATPR_HPP

#include <vector>

namespace htool {

template <typename T>
class VirtualMultiGenerator {
  protected:
    // Data members
    int nr;
    int nc;
    int nm;
    int dimension;

  public:
    VirtualMultiGenerator(int nr0, int nc0, int nm0, int dimension0 = 1) : nr(nr0), nc(nc0), nm(nm0), dimension(dimension0) {}

    virtual void copy_submatrices(int M, int N, const int *const rows, const int *const cols, int nb_matrix, T *ptr) const {}

    int nb_rows() const { return nr; }
    int nb_cols() const { return nc; }
    int nb_matrix() const { return nm; }

    virtual ~VirtualMultiGenerator(){};
};
} // namespace htool
#endif