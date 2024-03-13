#ifndef HTOOL_GENEO_COARSE_OPERATOR_BUILDER_HPP
#define HTOOL_GENEO_COARSE_OPERATOR_BUILDER_HPP

#include "../../distributed_operator/distributed_operator.hpp"
#include "../../misc/misc.hpp"
#include "../interfaces/virtual_coarse_operator_builder.hpp"
#include <algorithm>
#include <numeric>

namespace htool {

template <typename CoefficientPrecision>
void build_geneo_coarse_operator(const DistributedOperator<CoefficientPrecision> &HA, int nevi, int n, const CoefficientPrecision *const *Z, Matrix<CoefficientPrecision> &E) {
    //

    MPI_Comm comm = HA.get_comm();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    int n_inside = HA.get_target_partition().get_size_of_partition(rankWorld);

    // Allgather
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);
    MPI_Allgather(&nevi, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, HA.get_comm());

    displs[0] = 0;

    for (int i = 1; i < sizeWorld; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    int nevi_max = *std::max_element(recvcounts.begin(), recvcounts.end());
    int size_E   = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
    E.resize(size_E, size_E, 0);
    std::vector<CoefficientPrecision> evi(nevi * n, 0);
    for (int i = 0; i < nevi; i++) {
        // std::cout << i << "\n";
        std::copy_n(Z[i], n_inside, evi.data() + i * n);
    }

    // There should not be any bloc larger than the size of the subdmain, so margin should be useless
    // std::pair<int, int> max_local_size = HA.get_max_size_blocks();
    // int local_max_size_i               = max_local_size.first;
    // int local_max_size_j               = max_local_size.second;
    // int local_max_size                 = std::max(local_max_size_i, local_max_size_j);
    // int margin                         = local_max_size_j;
    // if (HA.get_symmetry_type() != 'N') {
    //     margin = local_max_size;
    // }
    int margin = 0;

    std::vector<CoefficientPrecision> AZ(nevi_max * n_inside, 0);
    std::vector<CoefficientPrecision> vec_ovr(n);

    // if (rankWorld == 0) {
    //     for (int i = 0; i < nevi; i++) {
    //         for (int j = 0; j < n_inside; j++) {
    //             std::cout << evi[i * n + j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rankWorld == 1) {
    //     for (int i = 0; i < nevi; i++) {
    //         for (int j = 0; j < n_inside; j++) {
    //             std::cout << evi[i * n + j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    // }

    for (int i = 0; i < sizeWorld; i++) {
        if (recvcounts[i] == 0)
            continue;
        std::vector<CoefficientPrecision> buffer((HA.get_target_partition().get_size_of_partition(i) + 2 * margin) * recvcounts[i], 0);
        std::fill_n(AZ.data(), recvcounts[i] * n_inside, 0);

        if (rankWorld == i) {
            // Transpose + partition of unity
            for (int j = 0; j < recvcounts[i]; j++) {
                for (int k = 0; k < n_inside; k++) {
                    buffer[recvcounts[i] * (k + margin) + j] = evi[j * n + k];
                }
            }
        }
        // if (rankWorld == 0) {
        //     // std::cout << buffer.size() << " " << HA.get_target_partition()->get_size_of_partition(i) << " " << recvcounts[i] << "\n";
        //     std::cout << buffer << "\n";
        // }
        MPI_Bcast(buffer.data() + margin * recvcounts[i], HA.get_target_partition().get_size_of_partition(i) * recvcounts[i], wrapper_mpi<CoefficientPrecision>::mpi_type(), i, HA.get_comm());
        if (HA.get_symmetry_type() == 'H') {
            conj_if_complex(buffer.data(), buffer.size());
        }
        HA.internal_sub_matrix_product_to_local(buffer.data(), AZ.data(), recvcounts[i], HA.get_target_partition().get_offset_of_partition(i), HA.get_target_partition().get_size_of_partition(i));

        // if (rankWorld == 0) {
        //     std::cout << "AZ " << i << "\n";
        //     std::cout << AZ << "\n";
        // }

        // Removed because complex scalar product afterward
        // if (HA.get_symmetry_type() == 'H') {
        //     conj_if_complex(AZ.data(), AZ.size());
        // }

        for (int j = 0; j < recvcounts[i]; j++) {
            for (int k = 0; k < n_inside; k++) {
                vec_ovr[k] = AZ[j + recvcounts[i] * k];
            }
            // Parce que partition de l'unité...
            // synchronize(true);
            for (int jj = 0; jj < nevi; jj++) {
                int coord_E_i           = displs[i] + j;
                int coord_E_j           = displs[rankWorld] + jj;
                E(coord_E_i, coord_E_j) = std::inner_product(evi.data() + jj * n, evi.data() + jj * n + n_inside, vec_ovr.data(), CoefficientPrecision(0), std::plus<CoefficientPrecision>(), [](CoefficientPrecision u, CoefficientPrecision v) { return u * v; });
            }
        }
    }

    if (rankWorld == 0)
        MPI_Reduce(MPI_IN_PLACE, E.data(), size_E * size_E, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, 0, HA.get_comm());
    else
        MPI_Reduce(E.data(), E.data(), size_E * size_E, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, 0, HA.get_comm());
}

template <typename CoefficientPrecision>
class GeneoCoarseOperatorBuilder : public VirtualCoarseOperatorBuilder<CoefficientPrecision> {
  private:
    const DistributedOperator<CoefficientPrecision> &m_distributed_operator;

  public:
    GeneoCoarseOperatorBuilder(const DistributedOperator<CoefficientPrecision> &HA) : m_distributed_operator(HA) {}

    Matrix<CoefficientPrecision> build_coarse_operator(int n, int nevi, CoefficientPrecision **coarse_space) override {
        Matrix<CoefficientPrecision> coarse_operator;
        htool::build_geneo_coarse_operator(m_distributed_operator, nevi, n, coarse_space, coarse_operator);
        return coarse_operator;
    }
};

} // namespace htool
#endif
