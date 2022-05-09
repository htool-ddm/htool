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
    VirtualHMatrix<T> *HA;
    std::unique_ptr<HMatrix<T>> off_diagonal_hmatrix;

  public:
    OffDiagonalApproximationWithHMatrix(VirtualHMatrix<T> *HA0, std::shared_ptr<VirtualCluster> target_cluster, std::shared_ptr<VirtualCluster> source_cluster) : HA(HA0) {
        // HMatrix
        off_diagonal_hmatrix = std::unique_ptr<HMatrix<T>>(new HMatrix<T>(target_cluster, source_cluster, HA->get_epsilon(), HA->get_eta(), 'N', 'N', -1, MPI_COMM_SELF));
        off_diagonal_hmatrix->set_maxblocksize(HA->get_maxblocksize());
        off_diagonal_hmatrix->set_minsourcedepth(HA->get_minsourcedepth());
        off_diagonal_hmatrix->set_mintargetdepth(HA->get_mintargetdepth());
        off_diagonal_hmatrix->set_maxblocksize(HA->get_maxblocksize());
    }

    void build(VirtualGenerator<T> &generator, const double *const xt, const double *const xs) {
        off_diagonal_hmatrix->build(generator, xt, xs);
    }

    void
    mvprod_global_to_local(const T *const in, T *const out, const int &mu) override {
        off_diagonal_hmatrix->mvprod_global_to_global(in, out, mu);
    }

    // Setters
    void set_compression(std::shared_ptr<VirtualLowRankGenerator<T>> compressor) {
        off_diagonal_hmatrix->set_compression(compressor);
    }

    bool IsUsingRowMajorStorage() override { return false; }

    // Output
    void save_plot(std::string filename) {
        int rank;
        MPI_Comm_rank(HA->get_comm(), &rank);
        off_diagonal_hmatrix->save_plot(filename + NbrToStr(rank));
    }
};

} // namespace htool

#endif
