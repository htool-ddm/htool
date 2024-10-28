#ifndef HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP

#include "../clustering/cluster_node.hpp"                // for Cluster
#include "../hmatrix/hmatrix.hpp"                        // for HMatrix
#include "../hmatrix/interfaces/virtual_generator.hpp"   // for GeneratorW...
#include "../hmatrix/tree_builder/tree_builder.hpp"      // for HMatrixTre...
#include "../local_operators/local_hmatrix.hpp"          // for LocalHMatrix
#include "../local_operators/virtual_local_operator.hpp" // for VirtualLocalOperator
#include "../misc/misc.hpp"                              // for underlying...
#include "distributed_operator.hpp"                      // for Distribute...
#include "implementations/partition_from_cluster.hpp"    // for PartitionF...
#include <functional>                                    // for function
#include <mpi.h>                                         // for MPI_Comm

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class CustomApproximationBuilder {
    const PartitionFromCluster<CoefficientPrecision, CoordinatePrecision> target_partition, source_partition;

  public:
    DistributedOperator<CoefficientPrecision> distributed_operator;

    explicit CustomApproximationBuilder(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, char symmetry, char UPLO, MPI_Comm communicator, const VirtualLocalOperator<CoefficientPrecision> &local_operator) : target_partition(target_cluster), source_partition(source_cluster), distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator) {
        distributed_operator.add_local_operator(&local_operator);
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class DistributedOperatorFromHMatrix {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };

  public:
    HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;

  private:
    const LocalHMatrix<CoefficientPrecision, CoordinatePrecision> local_hmatrix;
    CustomApproximationBuilder<CoefficientPrecision> distributed_operator_holder;

  public:
    DistributedOperator<CoefficientPrecision> &distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

    DistributedOperatorFromHMatrix(const VirtualInternalGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> &hmatrix_builder, MPI_Comm communicator) : hmatrix(hmatrix_builder.build(generator, target_cluster, source_cluster, get_rankWorld(communicator))), local_hmatrix(hmatrix, target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster, hmatrix_builder.get_symmetry(), hmatrix_builder.get_UPLO(), false, false), distributed_operator_holder(target_cluster, source_cluster, hmatrix_builder.get_symmetry(), hmatrix_builder.get_UPLO(), communicator, local_hmatrix), distributed_operator(distributed_operator_holder.distributed_operator) {
        block_diagonal_hmatrix = hmatrix.get_sub_hmatrix(target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)));
    }

    DistributedOperatorFromHMatrix(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> &hmatrix_builder, MPI_Comm communicator) : DistributedOperatorFromHMatrix(InternalGeneratorWithPermutation<CoefficientPrecision>(generator, target_cluster.get_permutation().data(), source_cluster.get_permutation().data()), target_cluster, source_cluster, hmatrix_builder, communicator) {
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class DefaultApproximationBuilder {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };
    DistributedOperatorFromHMatrix<CoefficientPrecision, CoordinatePrecision> distributed_operator_builder;

  public:
    HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix;

  public:
    DistributedOperator<CoefficientPrecision> &distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

    DefaultApproximationBuilder(const VirtualInternalGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : distributed_operator_builder(generator, target_cluster, source_cluster, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(epsilon, eta, symmetry, UPLO), communicator), hmatrix(distributed_operator_builder.hmatrix), distributed_operator(distributed_operator_builder.distributed_operator), block_diagonal_hmatrix(distributed_operator_builder.block_diagonal_hmatrix) {}

    DefaultApproximationBuilder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : DefaultApproximationBuilder(InternalGeneratorWithPermutation<CoefficientPrecision>(generator, target_cluster.get_permutation().data(), source_cluster.get_permutation().data()), target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, communicator) {}
};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class DefaultLocalApproximationBuilder {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld); 
    return rankWorld; };

  public:
    HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;

  private:
    const LocalHMatrix<CoefficientPrecision, CoordinatePrecision> local_hmatrix;
    CustomApproximationBuilder<CoefficientPrecision> distributed_operator_holder;

  public:
    DistributedOperator<CoefficientPrecision> &distributed_operator;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

  public:
    DefaultLocalApproximationBuilder(const VirtualInternalGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : hmatrix(HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(epsilon, eta, symmetry, UPLO).build(generator, target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)))), local_hmatrix(hmatrix, target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)), symmetry, UPLO, false, false), distributed_operator_holder(target_cluster, source_cluster, symmetry, UPLO, communicator, local_hmatrix), distributed_operator(distributed_operator_holder.distributed_operator) {
        block_diagonal_hmatrix = hmatrix.get_sub_hmatrix(target_cluster.get_cluster_on_partition(get_rankWorld(communicator)), source_cluster.get_cluster_on_partition(get_rankWorld(communicator)));
    }

    DefaultLocalApproximationBuilder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) : DefaultLocalApproximationBuilder(InternalGeneratorWithPermutation<CoefficientPrecision>(generator, target_cluster.get_permutation().data(), source_cluster.get_permutation().data()), target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, communicator) {}
};

} // namespace htool
#endif
