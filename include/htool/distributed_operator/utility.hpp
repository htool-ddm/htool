#ifndef HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP

#include "../hmatrix/hmatrix_distributed_output.hpp"
#include "../local_operators/local_hmatrix.hpp"
#include "distributed_operator.hpp"
#include "implementations/partition_from_cluster.hpp"
namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
DistributedOperator<CoefficientPrecision> build_default_hierarchical_approximation(const VirtualGenerator<CoefficientPrecision> &generator, std::shared_ptr<const Cluster<CoordinatePrecision>> target_cluster, std::shared_ptr<const Cluster<CoordinatePrecision>> source_cluster, double epsilon, double eta, char symmetry, char UPLO, MPI_Comm communicator, const HMatrix<CoefficientPrecision, CoordinatePrecision> **block_diagonal_hmatrix = nullptr) {

    //
    int rankWorld;
    MPI_Comm_rank(communicator, &rankWorld);

    // HMatrix compression
    auto local_hmatrix = std::make_shared<LocalHMatrix<CoefficientPrecision, CoordinatePrecision>>(generator, target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, false, false, rankWorld);
    if (block_diagonal_hmatrix != nullptr) {
        *block_diagonal_hmatrix = local_hmatrix->get_hmatrix().get_diagonal_hmatrix();
    }
    print_distributed_hmatrix_information(local_hmatrix->get_hmatrix(), std::cout, communicator);

    // Distributed operator
    std::shared_ptr<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>> target_partition = std::make_shared<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>>(target_cluster);
    std::shared_ptr<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>> source_partition = std::make_shared<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>>(source_cluster);
    DistributedOperator<CoefficientPrecision> distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator);
    distributed_operator.add_local_operator(local_hmatrix);

    return distributed_operator;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
DistributedOperator<CoefficientPrecision> build_default_local_hierarchical_approximation(const VirtualGenerator<CoefficientPrecision> &generator, std::shared_ptr<const Cluster<CoordinatePrecision>> target_cluster, std::shared_ptr<const Cluster<CoordinatePrecision>> source_cluster, double epsilon, double eta, char symmetry, char UPLO, MPI_Comm communicator, const HMatrix<CoefficientPrecision, CoordinatePrecision> **block_diagonal_hmatrix = nullptr) {

    //
    int rankWorld;
    MPI_Comm_rank(communicator, &rankWorld);

    // HMatrix compression
    std::shared_ptr<const Cluster<CoordinatePrecision>> local_target_root_cluster                         = std::make_shared<const Cluster<CoordinatePrecision>>(clone_cluster_tree_from_partition(*target_cluster, rankWorld));
    std::shared_ptr<const Cluster<htool::underlying_type<CoordinatePrecision>>> local_source_root_cluster = std::make_shared<const Cluster<CoordinatePrecision>>(clone_cluster_tree_from_partition(*source_cluster, rankWorld));

    auto local_hmatrix = std::make_shared<LocalHMatrix<CoefficientPrecision, CoordinatePrecision>>(generator, local_target_root_cluster, local_source_root_cluster, epsilon, eta, symmetry, UPLO);
    if (block_diagonal_hmatrix != nullptr) {
        *block_diagonal_hmatrix = local_hmatrix->get_hmatrix().get_diagonal_hmatrix();
    }

    print_distributed_hmatrix_information(local_hmatrix->get_hmatrix(), std::cout, communicator);

    //
    std::shared_ptr<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>> target_partition = std::make_shared<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>>(target_cluster);
    std::shared_ptr<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>> source_partition = std::make_shared<PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>>(source_cluster);
    DistributedOperator<CoefficientPrecision> distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator);
    distributed_operator.add_local_operator(local_hmatrix);

    return distributed_operator;
}

} // namespace htool
#endif
