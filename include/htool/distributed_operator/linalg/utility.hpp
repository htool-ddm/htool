#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_UTILITY_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_UTILITY_HPP

#include "../../misc/misc.hpp"
#include "../../wrappers/wrapper_mpi.hpp"
#include "../interfaces/virtual_partition.hpp"

namespace htool {

template <typename CoefficientPrecision>
void local_to_global(const VirtualPartition<CoefficientPrecision> &partition, const CoefficientPrecision *in, CoefficientPrecision *out, int mu, MPI_Comm comm) {

    // Allgather
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = (partition.get_size_of_partition(i)) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
}
} // namespace htool

#endif
