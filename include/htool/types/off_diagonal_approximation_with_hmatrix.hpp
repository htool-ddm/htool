#ifndef HTOOL_OFF_DIAGONAL_APPROXIMATION_WITH_HMATRIX_HPP
#define HTOOL_OFF_DIAGONAL_APPROXIMATION_WITH_HMATRIX_HPP

#include "hmatrix.hpp"
#include "vector.hpp"
#include "virtual_off_diagonal_approximation.hpp"
#include <cassert>
#include <iterator>
namespace htool {

template <typename T>
class OffDiagonalApproximationWithHMatrix : public VirtualOffDiagonalApproximation<T> {
    std::unique_ptr<HMatrix<T>> off_diagonal_hmatrix;

  public:
    OffDiagonalApproximationWithHMatrix(VirtualHMatrix<T> *HA0, std::shared_ptr<VirtualCluster> target_cluster, std::shared_ptr<VirtualCluster> source_cluster) {
        // HMatrix
        off_diagonal_hmatrix = std::unique_ptr<HMatrix<T>>(new HMatrix<T>(target_cluster, source_cluster, HA0->get_epsilon(), HA0->get_eta(), 'N', 'N', -1, MPI_COMM_SELF));
        off_diagonal_hmatrix->set_maxblocksize(HA0->get_maxblocksize());
        off_diagonal_hmatrix->set_minsourcedepth(HA0->get_minsourcedepth());
        off_diagonal_hmatrix->set_mintargetdepth(HA0->get_mintargetdepth());
        off_diagonal_hmatrix->set_maxblocksize(HA0->get_maxblocksize());
    }

    void build(VirtualGenerator<T> &generator, const double *const xt, const double *const xs) {
        off_diagonal_hmatrix->build(generator, xt, xs);
    }

    void mvprod_global_to_local(const T *const in, T *const out, const int &mu) override {
        off_diagonal_hmatrix->mvprod_global_to_global(in, out, mu);
    }

    void mvprod_subrhs_to_local(const T *const in, T *const out, const int &mu, const int &offset, const int &size) override {
        std::vector<T> in_global(off_diagonal_hmatrix->nb_cols() * mu, 0);
        for (int i = 0; i < mu; i++) {
            std::copy_n(in + size * i, size, in_global.data() + offset + off_diagonal_hmatrix->nb_cols() * i);
        }
        off_diagonal_hmatrix->mvprod_global_to_global(in_global.data(), out, mu);
    }

    // Setters
    void set_compression(std::shared_ptr<VirtualLowRankGenerator<T>> compressor) {
        off_diagonal_hmatrix->set_compression(compressor);
    }

    bool IsUsingRowMajorStorage() override { return false; }
    int nb_cols() const { return off_diagonal_hmatrix->nb_cols(); }
    int nb_rows() const { return off_diagonal_hmatrix->nb_rows(); }

    // Output
    const HMatrix<T> *get_HMatrix() const { return off_diagonal_hmatrix.get(); }
    void save_plot(std::string filename, MPI_Comm comm = MPI_COMM_WORLD) const {
        int rank;
        MPI_Comm_rank(comm, &rank);
        off_diagonal_hmatrix->save_plot(filename + NbrToStr(rank));
    }
    std::vector<int> get_output() const {
        return off_diagonal_hmatrix->get_output();
    }
};

} // namespace htool

#endif
