#ifndef HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_UTILITY_HPP

#include "../hmatrix/hmatrix_distributed_output.hpp"
#include "../local_operators/local_hmatrix.hpp"
#include "distributed_operator.hpp"
#include "implementations/partition_from_cluster.hpp"
namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision>
auto build_default_hierarchical_approximation(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) {
    struct Holder {

      public:
        Cluster<CoordinatePrecision> local_target_root_cluster;
        PartitionFromCluster<CoefficientPrecision, CoordinatePrecision> target_partition, source_partition;
        HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;
        std::unique_ptr<const LocalHMatrix<CoefficientPrecision, CoordinatePrecision>> local_hmatrix;
        DistributedOperator<CoefficientPrecision> distributed_operator;
        const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

        Holder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, int rankWorld, MPI_Comm communicator) : local_target_root_cluster(clone_cluster_tree_from_partition(target_cluster, rankWorld)), target_partition(target_cluster), source_partition(source_cluster), hmatrix(HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, -1, rankWorld).build(generator)), local_hmatrix(std::make_unique<LocalHMatrix<CoefficientPrecision, CoordinatePrecision>>(hmatrix, local_target_root_cluster, source_cluster, symmetry, UPLO, false, false)), distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator) {
            distributed_operator.add_local_operator(local_hmatrix.get());
            block_diagonal_hmatrix = hmatrix.get_diagonal_hmatrix();
        }

        Holder(const Holder &)                      = delete;
        Holder &operator=(const Holder &)           = delete;
        Holder(Holder &&holder) noexcept            = default;
        Holder &operator=(Holder &&holder) noexcept = default;
        virtual ~Holder()                           = default;
    };
    //
    int rankWorld;
    MPI_Comm_rank(communicator, &rankWorld);

    return Holder(generator, target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, rankWorld, communicator);
    ;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
auto build_default_local_hierarchical_approximation(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, MPI_Comm communicator) {

    struct Holder {

      public:
        Cluster<CoordinatePrecision> local_target_root_cluster, local_source_root_cluster;
        PartitionFromCluster<CoefficientPrecision, CoordinatePrecision> target_partition, source_partition;
        HMatrix<CoefficientPrecision, CoordinatePrecision> hmatrix;
        const LocalHMatrix<CoefficientPrecision, CoordinatePrecision> local_hmatrix;
        DistributedOperator<CoefficientPrecision> distributed_operator;
        const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix{nullptr};

        Holder(const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, htool::underlying_type<CoefficientPrecision> epsilon, htool::underlying_type<CoefficientPrecision> eta, char symmetry, char UPLO, int rankWorld, MPI_Comm communicator) : local_target_root_cluster(clone_cluster_tree_from_partition(target_cluster, rankWorld)), local_source_root_cluster(clone_cluster_tree_from_partition(source_cluster, rankWorld)), target_partition(target_cluster), source_partition(source_cluster), hmatrix(HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(local_target_root_cluster, local_source_root_cluster, epsilon, eta, symmetry, UPLO, -1, -1).build(generator)), local_hmatrix(hmatrix, local_target_root_cluster, local_source_root_cluster, symmetry, UPLO, false, false), distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator) {
            distributed_operator.add_local_operator(&local_hmatrix);
            block_diagonal_hmatrix = hmatrix.get_diagonal_hmatrix();
        }

        Holder(const Holder &)                      = delete;
        Holder &operator=(const Holder &)           = delete;
        Holder(Holder &&holder) noexcept            = default;
        Holder &operator=(Holder &&holder) noexcept = default;
        virtual ~Holder()                           = default;
    };
    //
    int rankWorld;
    MPI_Comm_rank(communicator, &rankWorld);
    // if (block_diagonal_hmatrix != nullptr) {
    //     *block_diagonal_hmatrix = local_hmatrix->get_hmatrix().get_diagonal_hmatrix();
    // }
    // DistributedOperator<CoefficientPrecision> distributed_operator(target_partition, source_partition, symmetry, UPLO, communicator);
    // distributed_operator.add_local_operator(local_hmatrix);

    return Holder(generator, target_cluster, source_cluster, epsilon, eta, symmetry, UPLO, rankWorld, communicator);
    ;
}

} // namespace htool
#endif
