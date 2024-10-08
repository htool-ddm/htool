#ifndef HTOOL_SOLVERS_UTILITY_HPP
#define HTOOL_SOLVERS_UTILITY_HPP

#include "../clustering/cluster_node.hpp"                   // for Cluster
#include "../clustering/tree_builder/recursive_build.hpp"   // for ClusterTre...
#include "../distributed_operator/distributed_operator.hpp" // for DistributedOperator
#include "../hmatrix/hmatrix.hpp"                           // for HMatrix
#include "../hmatrix/interfaces/virtual_generator.hpp"      // for VirtualGen...
#include "../hmatrix/tree_builder/tree_builder.hpp"         // for HMatrixTre...
#include "../matrix/matrix.hpp"                             // for Matrix
#include "../misc/misc.hpp"                                 // for underlying...
#include "../wrappers/wrapper_hpddm.hpp"                    // for HPDDMCusto...
#include "ddm.hpp"                                          // for make_DDM_s...
#include <algorithm>                                        // for copy, copy_n
#include <array>                                            // for array
#include <functional>                                       // for function
#include <memory>                                           // for make_unique
#include <vector>                                           // for vector

namespace htool {

class LocalNumberingBuilder {
  public:
    std::vector<int> local_to_global_numbering;
    std::vector<std::vector<int>> intersections;
    LocalNumberingBuilder() {}
    LocalNumberingBuilder(const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<std::vector<int>> &input_intersections) : local_to_global_numbering(ovr_subdomain_to_global.size()), intersections(input_intersections.size()) {

        // Renumbering for the overlap
        int local_size_with_overlap = ovr_subdomain_to_global.size();
        std::vector<int> renum(local_size_with_overlap, -1);

        for (int i = 0; i < cluster_to_ovr_subdomain.size(); i++) {
            renum[cluster_to_ovr_subdomain[i]] = i;
            local_to_global_numbering[i]       = ovr_subdomain_to_global[cluster_to_ovr_subdomain[i]];
        }
        int count = cluster_to_ovr_subdomain.size();
        for (int i = 0; i < local_size_with_overlap; i++) {
            if (renum[i] == -1) {
                renum[i]                           = count;
                local_to_global_numbering[count++] = ovr_subdomain_to_global[i];
            }
        }

        for (int i = 0; i < input_intersections.size(); i++) {
            intersections[i].resize(input_intersections[i].size());
            for (int j = 0; j < intersections[i].size(); j++) {
                intersections[i][j] = renum[input_intersections[i][j]];
            }
        }
    }
};

// template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
// class BlockJacobiWithDenseLocalSolver {
//   private:
//     std::function<Matrix<CoefficientPrecision>(const HMatrix<CoefficientPrecision, CoordinatePrecision> &)> initialize_diagonal_block = [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix_ptr) {
//         Matrix<CoefficientPrecision> block(block_diagonal_hmatrix_ptr.get_target_cluster().get_size(), block_diagonal_hmatrix_ptr.get_source_cluster().get_size());
//         copy_to_dense(block_diagonal_hmatrix_ptr, block.data());
//         return block;
//     };
//     Matrix<CoefficientPrecision> m_block_diagonal_dense_matrix;
//     std::vector<int> m_neighbors;
//     std::vector<std::vector<int>> m_intersections;

//   public:
//     DDM<CoefficientPrecision, HPDDM::LapackTRSub> solver;
//     BlockJacobiWithDenseLocalSolver(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix) : m_block_diagonal_dense_matrix(initialize_diagonal_block(block_diagonal_hmatrix)), solver(make_DDM_solver(distributed_operator, m_block_diagonal_dense_matrix, m_neighbors, m_intersections)) {}
// };

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class DDMSolverWithDenseLocalSolver {
  private:
    std::vector<int> m_neighbors;
    std::vector<std::vector<int>> m_intersections;
    LocalNumberingBuilder m_local_numbering;

  public:
    const std::vector<int> &local_to_global_numbering;

  private:
    std::function<Matrix<CoefficientPrecision>(const HMatrix<CoefficientPrecision, CoordinatePrecision> &)> initialize_diagonal_block_without_overlap = [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix_ptr) {
        Matrix<CoefficientPrecision> block(block_diagonal_hmatrix_ptr.get_target_cluster().get_size(), block_diagonal_hmatrix_ptr.get_source_cluster().get_size());
        copy_to_dense(block_diagonal_hmatrix_ptr, block.data());
        return block;
    };

    std::function<Matrix<CoefficientPrecision>(DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> &, const VirtualGenerator<CoefficientPrecision> &)> initialize_diagonal_block_adding_overlap = [this](DistributedOperator<CoefficientPrecision> &distributed_operator0, const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix0, const VirtualGenerator<CoefficientPrecision> &generator0) {
        int local_size_with_overlap    = local_to_global_numbering.size();
        int local_size_without_overlap = block_diagonal_hmatrix0.get_target_cluster().get_size();
        Matrix<CoefficientPrecision> block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap), block_diagonal_dense_matrix_without_overlap(local_size_without_overlap, local_size_without_overlap);

        // Diagonal block without overlap
        copy_to_dense(block_diagonal_hmatrix0, block_diagonal_dense_matrix_without_overlap.data());

        // Assemble block diagonal dense matrix with overlap
        for (int j = 0; j < local_size_without_overlap; j++) {
            std::copy_n(block_diagonal_dense_matrix_without_overlap.data() + j * local_size_without_overlap, local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + j * local_size_with_overlap);
        }

        // Overlap
        std::vector<CoefficientPrecision> horizontal_block((local_size_with_overlap - local_size_without_overlap) * local_size_without_overlap), diagonal_block((local_size_with_overlap - local_size_without_overlap) * (local_size_with_overlap - local_size_without_overlap));

        std::vector<int> overlap_num(local_to_global_numbering.begin() + local_size_without_overlap, local_to_global_numbering.end());
        std::vector<int> inside_num(local_to_global_numbering.begin(), local_to_global_numbering.begin() + local_size_without_overlap);

        generator0.copy_submatrix(local_size_with_overlap - local_size_without_overlap, local_size_without_overlap, overlap_num.data(), inside_num.data(), horizontal_block.data());
        for (int j = 0; j < local_size_without_overlap; j++) {
            std::copy_n(horizontal_block.begin() + j * (local_size_with_overlap - local_size_without_overlap), local_size_with_overlap - local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + local_size_without_overlap + j * local_size_with_overlap);
        }

        generator0.copy_submatrix(local_size_with_overlap - local_size_without_overlap, local_size_with_overlap - local_size_without_overlap, overlap_num.data(), overlap_num.data(), diagonal_block.data());
        for (int j = 0; j < local_size_with_overlap - local_size_without_overlap; j++) {
            std::copy_n(diagonal_block.begin() + j * (local_size_with_overlap - local_size_without_overlap), local_size_with_overlap - local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + local_size_without_overlap + (j + local_size_without_overlap) * local_size_with_overlap);
        }

        bool sym = (distributed_operator0.get_symmetry_type() == 'S' || (distributed_operator0.get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) ? true : false;
        if (!sym) {
            std::vector<CoefficientPrecision> vertical_block(local_size_without_overlap * (local_size_with_overlap - local_size_without_overlap));
            generator0.copy_submatrix(local_size_without_overlap, local_size_with_overlap - local_size_without_overlap, inside_num.data(), overlap_num.data(), vertical_block.data());
            for (int j = local_size_without_overlap; j < local_size_with_overlap; j++) {
                std::copy_n(vertical_block.begin() + (j - local_size_without_overlap) * local_size_without_overlap, local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + j * local_size_with_overlap);
            }
        }
        return block_diagonal_dense_matrix_with_overlap;
    };

    std::function<Matrix<CoefficientPrecision>(const HMatrix<CoefficientPrecision, CoordinatePrecision> &)> initialize_diagonal_block_with_overlap = [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix0) {
        int local_size_with_overlap = block_diagonal_hmatrix0.get_target_cluster().get_size();
        // Matrix<CoefficientPrecision> block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap);
        Matrix<CoefficientPrecision> permuted_block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap);
        copy_to_dense_in_user_numbering(block_diagonal_hmatrix0, permuted_block_diagonal_dense_matrix_with_overlap.data());

        return permuted_block_diagonal_dense_matrix_with_overlap;
    };

    std::function<Cluster<CoordinatePrecision>(int, const CoordinatePrecision *)> initialize_local_cluster = [this](int spatial_dimension0, const CoordinatePrecision *global_geometry0) {
        // Local geometry
        int local_size = local_to_global_numbering.size();
        std::vector<double> local_geometry(spatial_dimension0 * local_size);
        for (int i = 0; i < local_to_global_numbering.size(); i++) {
            for (int dimension = 0; dimension < spatial_dimension0; dimension++) {
                local_geometry[spatial_dimension0 * i + dimension] = global_geometry0[spatial_dimension0 * local_to_global_numbering[i] + dimension];
            }
        }

        // Local cluster
        ClusterTreeBuilder<double> recursive_build_strategy;
        return recursive_build_strategy.create_cluster_tree(local_to_global_numbering.size(), spatial_dimension0, local_geometry.data(), 2, 2, nullptr);
    };

    std::unique_ptr<Cluster<CoordinatePrecision>> local_cluster;

    std::function<HMatrix<CoefficientPrecision, CoordinatePrecision>(const VirtualGenerator<CoefficientPrecision> &, underlying_type<CoefficientPrecision>, CoordinatePrecision, char)> initialize_local_hmatrix = [this](const VirtualGenerator<CoefficientPrecision> &generator0, underlying_type<CoefficientPrecision> epsilon0, CoordinatePrecision eta0, char symmetry0) {
        struct LocalGeneratorInUserNumbering : public VirtualGenerator<CoefficientPrecision> {
            const std::vector<int> &m_target_local_to_global_numbering;
            const std::vector<int> &m_source_local_to_global_numbering;
            const VirtualGenerator<CoefficientPrecision> &m_generator;

            LocalGeneratorInUserNumbering(const VirtualGenerator<CoefficientPrecision> &generator0, const std::vector<int> &target_local_to_global_numbering, const std::vector<int> &source_local_to_global_numbering) : m_target_local_to_global_numbering(target_local_to_global_numbering), m_source_local_to_global_numbering(source_local_to_global_numbering), m_generator(generator0) {}

            void copy_submatrix(int M, int N, const int *const rows, const int *const cols, CoefficientPrecision *ptr) const override {
                std::vector<int> new_rows(M), new_cols(N);
                for (int i = 0; i < M; i++) {
                    new_rows[i] = m_target_local_to_global_numbering[rows[i]];
                }
                for (int j = 0; j < N; j++) {
                    new_cols[j] = m_source_local_to_global_numbering[cols[j]];
                }
                m_generator.copy_submatrix(M, N, new_rows.data(), new_cols.data(), ptr);
            }
        };

        // Local Generator
        LocalGeneratorInUserNumbering local_generator(generator0, local_to_global_numbering, local_to_global_numbering);

        // Local HMatrix
        HMatrixTreeBuilder<CoefficientPrecision> local_hmatrix_builder(*local_cluster, *local_cluster, epsilon0, eta0, symmetry0, symmetry0 != 'N' ? 'L' : 'N', -1, -1, -1);

        return local_hmatrix_builder.build(local_generator);
    };

  public:
    std::unique_ptr<HMatrix<CoefficientPrecision, CoordinatePrecision>> local_hmatrix;
    Matrix<CoefficientPrecision> block_diagonal_dense_matrix;
    DDM<CoefficientPrecision, HPDDM::LapackTRSub> solver;

    // Block Jacobi
    DDMSolverWithDenseLocalSolver(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix) : local_to_global_numbering(block_diagonal_hmatrix.get_target_cluster().get_permutation()), block_diagonal_dense_matrix(initialize_diagonal_block_without_overlap(block_diagonal_hmatrix)), solver(make_DDM_solver(distributed_operator, block_diagonal_dense_matrix, m_neighbors, m_intersections)) {}

    // DDM adding overlap to local hmatrix without overlap
    DDMSolverWithDenseLocalSolver(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix, const VirtualGenerator<CoefficientPrecision> &generator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) : m_local_numbering(ovr_subdomain_to_global, cluster_to_ovr_subdomain, intersections), local_to_global_numbering(m_local_numbering.local_to_global_numbering), block_diagonal_dense_matrix(initialize_diagonal_block_adding_overlap(distributed_operator, block_diagonal_hmatrix, generator)), solver(make_DDM_solver(distributed_operator, block_diagonal_dense_matrix, neighbors, m_local_numbering.intersections)) {}

    // DDM building local hmatrix with overlap
    DDMSolverWithDenseLocalSolver(DistributedOperator<CoefficientPrecision> &distributed_operator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections, const VirtualGenerator<CoefficientPrecision> &generator, int spatial_dimension, const CoordinatePrecision *global_geometry, underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta) : m_local_numbering(ovr_subdomain_to_global, cluster_to_ovr_subdomain, intersections), local_to_global_numbering(m_local_numbering.local_to_global_numbering), local_cluster(std::make_unique<Cluster<CoordinatePrecision>>(initialize_local_cluster(spatial_dimension, global_geometry))), local_hmatrix(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(initialize_local_hmatrix(generator, epsilon, eta, distributed_operator.get_symmetry_type()))), block_diagonal_dense_matrix(initialize_diagonal_block_with_overlap(*local_hmatrix)), solver(make_DDM_solver(distributed_operator, block_diagonal_dense_matrix, neighbors, m_local_numbering.intersections)) {}
};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class DDMSolverBuilder {
  private:
    std::vector<int> m_neighbors;
    std::vector<std::vector<int>> m_intersections;
    LocalNumberingBuilder m_local_numbering;

  public:
    const std::vector<int> &local_to_global_numbering;

  private:
    std::function<std::array<Matrix<CoefficientPrecision>, 3>(DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> &, const VirtualGenerator<CoefficientPrecision> &)> initialize_blocks_in_overlap = [this](DistributedOperator<CoefficientPrecision> &distributed_operator0, const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix0, const VirtualGenerator<CoefficientPrecision> &generator0) {
        std::array<Matrix<CoefficientPrecision>, 3> blocks_in_overlap0;
        Matrix<CoefficientPrecision> &B = blocks_in_overlap0[0];
        Matrix<CoefficientPrecision> &C = blocks_in_overlap0[1];
        Matrix<CoefficientPrecision> &D = blocks_in_overlap0[2];
        int local_size_with_overlap     = local_to_global_numbering.size();
        int local_size_without_overlap  = block_diagonal_hmatrix0.get_target_cluster().get_size();

        std::vector<int> overlap_num(local_to_global_numbering.begin() + local_size_without_overlap, local_to_global_numbering.end());
        std::vector<int> inside_num(local_to_global_numbering.begin(), local_to_global_numbering.begin() + local_size_without_overlap);

        // Overlap
        D.resize(local_size_with_overlap - local_size_without_overlap, local_size_with_overlap - local_size_without_overlap);

        generator0.copy_submatrix(local_size_with_overlap - local_size_without_overlap, local_size_with_overlap - local_size_without_overlap, overlap_num.data(), overlap_num.data(), D.data());

        bool sym = (distributed_operator0.get_symmetry_type() == 'S' || (distributed_operator0.get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) ? true : false;

        if (!sym or block_diagonal_hmatrix0.get_UPLO() == 'U') {
            B.resize(local_size_without_overlap, local_size_with_overlap - local_size_without_overlap);
            generator0.copy_submatrix(local_size_without_overlap, local_size_with_overlap - local_size_without_overlap, inside_num.data(), overlap_num.data(), B.data());
        } else if (!sym or block_diagonal_hmatrix0.get_UPLO() == 'L') {
            C.resize(local_size_with_overlap - local_size_without_overlap, local_size_without_overlap);

            generator0.copy_submatrix(local_size_with_overlap - local_size_without_overlap, local_size_without_overlap, overlap_num.data(), inside_num.data(), C.data());
        }
        return blocks_in_overlap0;
    };

    std::function<Cluster<CoordinatePrecision>(int, const CoordinatePrecision *)> initialize_local_cluster = [this](int spatial_dimension0, const CoordinatePrecision *global_geometry0) {
        // Local geometry
        int local_size = local_to_global_numbering.size();
        std::vector<double> local_geometry(spatial_dimension0 * local_size);
        for (int i = 0; i < local_to_global_numbering.size(); i++) {
            for (int dimension = 0; dimension < spatial_dimension0; dimension++) {
                local_geometry[spatial_dimension0 * i + dimension] = global_geometry0[spatial_dimension0 * local_to_global_numbering[i] + dimension];
            }
        }

        // Local cluster
        ClusterTreeBuilder<double> recursive_build_strategy;
        return recursive_build_strategy.create_cluster_tree(local_to_global_numbering.size(), spatial_dimension0, local_geometry.data(), 2, 2);
    };

    std::unique_ptr<Cluster<CoordinatePrecision>> local_cluster;

    std::function<HMatrix<CoefficientPrecision, CoordinatePrecision>(const VirtualGenerator<CoefficientPrecision> &, underlying_type<CoefficientPrecision>, CoordinatePrecision, char, char)> initialize_local_hmatrix = [this](const VirtualGenerator<CoefficientPrecision> &generator0, underlying_type<CoefficientPrecision> epsilon0, CoordinatePrecision eta0, char symmetry0, char UPLO0) {
        struct LocalGeneratorInUserNumbering : public VirtualGenerator<CoefficientPrecision> {
            const std::vector<int> &m_target_local_to_global_numbering;
            const std::vector<int> &m_source_local_to_global_numbering;
            const VirtualGenerator<CoefficientPrecision> &m_generator;

            LocalGeneratorInUserNumbering(const VirtualGenerator<CoefficientPrecision> &generator, const std::vector<int> &target_local_to_global_numbering, const std::vector<int> &source_local_to_global_numbering) : m_target_local_to_global_numbering(target_local_to_global_numbering), m_source_local_to_global_numbering(source_local_to_global_numbering), m_generator(generator) {}

            void copy_submatrix(int M, int N, const int *const rows, const int *const cols, CoefficientPrecision *ptr) const override {
                std::vector<int> new_rows(M), new_cols(N);
                for (int i = 0; i < M; i++) {
                    new_rows[i] = m_target_local_to_global_numbering[rows[i]];
                }
                for (int j = 0; j < N; j++) {
                    new_cols[j] = m_source_local_to_global_numbering[cols[j]];
                }
                m_generator.copy_submatrix(M, N, new_rows.data(), new_cols.data(), ptr);
            }
        };

        // Local Generator
        LocalGeneratorInUserNumbering local_generator(generator0, local_to_global_numbering, local_to_global_numbering);

        // Local HMatrix
        HMatrixTreeBuilder<CoefficientPrecision> local_hmatrix_builder(*local_cluster, *local_cluster, epsilon0, eta0, symmetry0, UPLO0, -1, -1, -1);

        return local_hmatrix_builder.build(local_generator);
    };

  public:
    std::unique_ptr<HMatrix<CoefficientPrecision, CoordinatePrecision>> local_hmatrix; // A
    std::array<Matrix<CoefficientPrecision>, 3> blocks_in_overlap;                     // B,C,D
    DDM<CoefficientPrecision, HPDDMCustomLocalSolver> solver;

    // Block Jacobi
    DDMSolverBuilder(DistributedOperator<CoefficientPrecision> &distributed_operator, HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix) : local_to_global_numbering(block_diagonal_hmatrix.get_target_cluster().get_permutation()), solver(make_DDM_solver_w_custom_local_solver(distributed_operator, block_diagonal_hmatrix, m_neighbors, m_intersections, false)) {}

    // DDM building local hmatrix adding overlap
    DDMSolverBuilder(DistributedOperator<CoefficientPrecision> &distributed_operator, HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix, const VirtualGenerator<CoefficientPrecision> &generator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) : m_local_numbering(ovr_subdomain_to_global, cluster_to_ovr_subdomain, intersections), local_to_global_numbering(m_local_numbering.local_to_global_numbering), blocks_in_overlap(initialize_blocks_in_overlap(distributed_operator, block_diagonal_hmatrix, generator)), solver(make_DDM_solver_w_custom_local_solver(distributed_operator, block_diagonal_hmatrix, blocks_in_overlap[0], blocks_in_overlap[1], blocks_in_overlap[2], neighbors, m_local_numbering.intersections)) {}

    // DDM building local hmatrix with overlap
    DDMSolverBuilder(DistributedOperator<CoefficientPrecision> &distributed_operator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections, const VirtualGenerator<CoefficientPrecision> &generator, int spatial_dimension, const CoordinatePrecision *global_geometry, underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta) : m_local_numbering(ovr_subdomain_to_global, cluster_to_ovr_subdomain, intersections), local_to_global_numbering(m_local_numbering.local_to_global_numbering), local_cluster(std::make_unique<Cluster<CoordinatePrecision>>(initialize_local_cluster(spatial_dimension, global_geometry))), local_hmatrix(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(initialize_local_hmatrix(generator, epsilon, eta, distributed_operator.get_symmetry_type(), distributed_operator.get_storage_type()))), solver(make_DDM_solver_w_custom_local_solver(distributed_operator, *local_hmatrix, neighbors, m_local_numbering.intersections, true)) {}
};

} // namespace htool
#endif
