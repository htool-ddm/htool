#ifndef HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP

#include "../hmatrix/hmatrix_distributed_output.hpp"
#include "../local_operators/local_hmatrix.hpp"
#include "distributed_operator.hpp"
#include "implementations/partition_from_cluster.hpp"
namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultApproximationBuilder {
  private:
    const PartitionFromCluster<CoefficientPrecision, CoordinatePrecision> target_partition, source_partition;
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };

  public:
    const HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;

  private:
    const LocalHMatrix<CoefficientPrecision, CoordinatePrecision> local_hmatrix;

  public:
    DistributedOperator<CoefficientPrecision> distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

    DefaultApproximationBuilder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : target_partition(target_cluster), source_partition(source_cluster), hmatrix(HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, -1, get_rankWorld(communicator), get_rankWorld(communicator)).build(generator)), local_hmatrix(hmatrix, target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster, symmetry, UPLO, false, false), distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator) {
        distributed_operator.add_local_operator(&local_hmatrix);
        block_diagonal_hmatrix = hmatrix.get_sub_hmatrix(target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)));
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultLocalApproximationBuilder {
  private:
    const PartitionFromCluster<CoefficientPrecision, CoordinatePrecision> target_partition, source_partition;
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };

  public:
    const HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;

  private:
    const LocalHMatrix<CoefficientPrecision, CoordinatePrecision> local_hmatrix;

  public:
    DistributedOperator<CoefficientPrecision> distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

  public:
    DefaultLocalApproximationBuilder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : target_partition(target_cluster), source_partition(source_cluster), hmatrix(HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)), epsilon, eta, symmetry, UPLO, -1, -1, -1).build(generator)), local_hmatrix(hmatrix, target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)), symmetry, UPLO, false, false), distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator) {
        distributed_operator.add_local_operator(&local_hmatrix);
        block_diagonal_hmatrix = hmatrix.get_sub_hmatrix(target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)));
    }
};

} // namespace htool
#endif
