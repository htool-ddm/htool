#ifndef HTOOL_SOLVERS_UTILITY_HPP
#define HTOOL_SOLVERS_UTILITY_HPP

#include "../distributed_operator/distributed_operator.hpp"
#include "../local_operators/local_hmatrix.hpp"
#include "ddm.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
DDM<CoefficientPrecision> build_ddm_solver(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix) {

    int rankWorld;
    MPI_Comm_rank(distributed_operator.get_comm(), &rankWorld);

    // Local dense operator
    int local_size = distributed_operator.get_target_partition().get_size_of_partition(rankWorld);

    std::unique_ptr<Matrix<CoefficientPrecision>> local_dense_operator = std::make_unique<Matrix<CoefficientPrecision>>(local_size, local_size);
    std::cout << "local size " << local_size << "\n";
    copy_to_dense(*block_diagonal_hmatrix, local_dense_operator->data());

    // ddm solver
    return DDM<CoefficientPrecision>(&distributed_operator, std::move(local_dense_operator));
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
DDM<CoefficientPrecision> build_ddm_solver(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix, const VirtualGeneratorWithPermutation<CoefficientPrecision> &generator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) {

    int rankWorld;
    MPI_Comm_rank(distributed_operator.get_comm(), &rankWorld);

    // Local dense operator
    int local_size_w_overlap = ovr_subdomain_to_global.size();

    std::unique_ptr<Matrix<CoefficientPrecision>> local_dense_operator(new Matrix<CoefficientPrecision>(local_size_w_overlap, local_size_w_overlap));
    copy_to_dense(*block_diagonal_hmatrix, local_dense_operator->data());

    // ddm solver
    return DDM<CoefficientPrecision>(&distributed_operator, *local_dense_operator, generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
}

} // namespace htool
#endif
