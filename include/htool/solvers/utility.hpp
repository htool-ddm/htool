#ifndef HTOOL_SOLVERS_UTILITY_HPP
#define HTOOL_SOLVERS_UTILITY_HPP

#include "../distributed_operator/distributed_operator.hpp"
#include "../hmatrix/hmatrix.hpp"
#include "ddm.hpp"

namespace htool {

class LocalNumberingBuilder {
  public:
    std::vector<int> local_to_global_numbering;
    std::vector<std::vector<int>> intersections;
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

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultSolverBuilder {
  private:
    std::function<Matrix<CoefficientPrecision>(const HMatrix<CoefficientPrecision, CoordinatePrecision> *)> initialize_diagonal_block = [](const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix_ptr) {
        Matrix<CoefficientPrecision> block(block_diagonal_hmatrix_ptr->get_target_cluster().get_size(), block_diagonal_hmatrix_ptr->get_source_cluster().get_size());
        copy_to_dense(*block_diagonal_hmatrix_ptr, block.data());
        return block;
    };
    Matrix<CoefficientPrecision> m_block_diagonal_dense_matrix;
    std::vector<int> m_neighbors;
    std::vector<std::vector<int>> m_intersections;

  public:
    DDM<CoefficientPrecision> solver;
    DefaultSolverBuilder(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix) : m_block_diagonal_dense_matrix(initialize_diagonal_block(block_diagonal_hmatrix)), solver(distributed_operator, m_block_diagonal_dense_matrix, m_neighbors, m_intersections) {}
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultDDMSolverBuilderAddingOverlap {
  private:
    LocalNumberingBuilder m_local_numbering;

  public:
    const std::vector<int> &local_to_global_numbering;

  private:
    std::function<Matrix<CoefficientPrecision>(DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> *, const VirtualGeneratorWithPermutation<CoefficientPrecision> &)> initialize_diagonal_block = [this](DistributedOperator<CoefficientPrecision> &distributed_operator0, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix0, const VirtualGeneratorWithPermutation<CoefficientPrecision> &generator0) {
        int local_size_with_overlap    = local_to_global_numbering.size();
        int local_size_without_overlap = block_diagonal_hmatrix0->get_target_cluster().get_size();
        Matrix<CoefficientPrecision> block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap), block_diagonal_dense_matrix_without_overlap(local_size_without_overlap, local_size_without_overlap);

        // Diagonal block without overlap
        copy_to_dense(*block_diagonal_hmatrix0, block_diagonal_dense_matrix_without_overlap.data());

        // Assemble block diagonal dense matrix with overlap
        for (int j = 0; j < local_size_without_overlap; j++) {
            std::copy_n(block_diagonal_dense_matrix_without_overlap.data() + j * local_size_without_overlap, local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + j * local_size_with_overlap);
        }

        // Overlap
        std::vector<CoefficientPrecision> horizontal_block((local_size_with_overlap - local_size_without_overlap) * local_size_without_overlap), diagonal_block((local_size_with_overlap - local_size_without_overlap) * (local_size_with_overlap - local_size_without_overlap));

        std::vector<int> overlap_num(local_to_global_numbering.begin() + local_size_without_overlap, local_to_global_numbering.end());
        std::vector<int> inside_num(local_to_global_numbering.begin(), local_to_global_numbering.begin() + local_size_without_overlap);

        generator0.copy_submatrix_from_user_numbering(local_size_with_overlap - local_size_without_overlap, local_size_without_overlap, overlap_num.data(), inside_num.data(), horizontal_block.data());
        for (int j = 0; j < local_size_without_overlap; j++) {
            std::copy_n(horizontal_block.begin() + j * (local_size_with_overlap - local_size_without_overlap), local_size_with_overlap - local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + local_size_without_overlap + j * local_size_with_overlap);
        }

        generator0.copy_submatrix_from_user_numbering(local_size_with_overlap - local_size_without_overlap, local_size_with_overlap - local_size_without_overlap, overlap_num.data(), overlap_num.data(), diagonal_block.data());
        for (int j = 0; j < local_size_with_overlap - local_size_without_overlap; j++) {
            std::copy_n(diagonal_block.begin() + j * (local_size_with_overlap - local_size_without_overlap), local_size_with_overlap - local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + local_size_without_overlap + (j + local_size_without_overlap) * local_size_with_overlap);
        }

        bool sym = (distributed_operator0.get_symmetry_type() == 'S' || (distributed_operator0.get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) ? true : false;
        if (!sym) {
            std::vector<CoefficientPrecision> vertical_block(local_size_without_overlap * (local_size_with_overlap - local_size_without_overlap));
            generator0.copy_submatrix_from_user_numbering(local_size_without_overlap, local_size_with_overlap - local_size_without_overlap, inside_num.data(), overlap_num.data(), vertical_block.data());
            for (int j = local_size_without_overlap; j < local_size_with_overlap; j++) {
                std::copy_n(vertical_block.begin() + (j - local_size_without_overlap) * local_size_without_overlap, local_size_without_overlap, block_diagonal_dense_matrix_with_overlap.data() + j * local_size_with_overlap);
            }
        }
        return block_diagonal_dense_matrix_with_overlap;
    };
    std::vector<std::vector<int>> m_intersections;

  public:
    Matrix<CoefficientPrecision> block_diagonal_dense_matrix;
    DDM<CoefficientPrecision> solver;

    DefaultDDMSolverBuilderAddingOverlap(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix, const VirtualGeneratorWithPermutation<CoefficientPrecision> &generator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) : m_local_numbering(ovr_subdomain_to_global, cluster_to_ovr_subdomain, intersections), local_to_global_numbering(m_local_numbering.local_to_global_numbering), block_diagonal_dense_matrix(initialize_diagonal_block(distributed_operator, block_diagonal_hmatrix, generator)), solver(distributed_operator, block_diagonal_dense_matrix, neighbors, m_local_numbering.intersections) {}
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class DefaultDDMSolverBuilder {
  private:
    std::function<Matrix<CoefficientPrecision>(const HMatrix<CoefficientPrecision, CoordinatePrecision> &)> initialize_diagonal_block = [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix0) {
        int local_size_with_overlap = block_diagonal_hmatrix0.get_target_cluster().get_size();
        Matrix<CoefficientPrecision> block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap);
        Matrix<CoefficientPrecision> permuted_block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap);
        // Diagonal block without overlap
        copy_to_dense(block_diagonal_hmatrix0, block_diagonal_dense_matrix_with_overlap.data());

        int nr                   = block_diagonal_hmatrix0.get_target_cluster().get_size();
        int nc                   = block_diagonal_hmatrix0.get_source_cluster().get_size();
        auto &target_permutation = block_diagonal_hmatrix0.get_target_cluster().get_permutation();

        if (block_diagonal_hmatrix0.get_symmetry() == 'N') {
            for (int i = 0; i < nr; i++) {
                for (int j = 0; j < nc; j++) {

                    permuted_block_diagonal_dense_matrix_with_overlap(target_permutation[i], target_permutation[j]) = block_diagonal_dense_matrix_with_overlap(i, j);
                }
            }
        } else {
            int index_i, index_j;
            for (int i = 0; i < nr; i++) {
                for (int j = 0; j <= i; j++) {
                    if (target_permutation[i] < target_permutation[j]) {
                        index_i = target_permutation[j];
                        index_j = target_permutation[i];
                    } else {
                        index_i = target_permutation[i];
                        index_j = target_permutation[j];
                    }
                    permuted_block_diagonal_dense_matrix_with_overlap(index_i, index_j) = block_diagonal_dense_matrix_with_overlap(i, j);
                }
            }
        }

        return permuted_block_diagonal_dense_matrix_with_overlap;
    };

  public:
    Matrix<CoefficientPrecision> block_diagonal_dense_matrix;
    DDM<CoefficientPrecision> solver;

    DefaultDDMSolverBuilder(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> &block_diagonal_hmatrix, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) : block_diagonal_dense_matrix(initialize_diagonal_block(block_diagonal_hmatrix)), solver(distributed_operator, block_diagonal_dense_matrix, neighbors, intersections) {}
};

} // namespace htool
#endif
