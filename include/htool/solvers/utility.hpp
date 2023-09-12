#ifndef HTOOL_SOLVERS_UTILITY_HPP
#define HTOOL_SOLVERS_UTILITY_HPP

#include "../distributed_operator/distributed_operator.hpp"
#include "../hmatrix/hmatrix.hpp"
#include "ddm.hpp"

namespace htool {

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
class DefaultDDMSolverBuilder {
  private:
    std::function<Matrix<CoefficientPrecision>(DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> *, const VirtualGeneratorWithPermutation<CoefficientPrecision> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &)> initialize_diagonal_block = [this](DistributedOperator<CoefficientPrecision> &distributed_operator0, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix0, const VirtualGeneratorWithPermutation<CoefficientPrecision> &generator0, const std::vector<int> &ovr_subdomain_to_global0, const std::vector<int> &cluster_to_ovr_subdomain0, const std::vector<int> &neighbors0, const std::vector<std::vector<int>> &input_intersections0) {
        int local_size_with_overlap    = ovr_subdomain_to_global0.size();
        int local_size_without_overlap = block_diagonal_hmatrix0->get_target_cluster().get_size();
        Matrix<CoefficientPrecision> block_diagonal_dense_matrix_with_overlap(local_size_with_overlap, local_size_with_overlap), block_diagonal_dense_matrix_without_overlap(local_size_without_overlap, local_size_without_overlap);

        // Diagonal block without overlap
        copy_to_dense(*block_diagonal_hmatrix0, block_diagonal_dense_matrix_without_overlap.data());

        // Renumbering for the overlap
        std::vector<int> renum(local_size_with_overlap, -1);
        local_to_global_numbering.resize(local_size_with_overlap);

        for (int i = 0; i < cluster_to_ovr_subdomain0.size(); i++) {
            renum[cluster_to_ovr_subdomain0[i]] = i;
            local_to_global_numbering[i]        = ovr_subdomain_to_global0[cluster_to_ovr_subdomain0[i]];
        }
        int count = cluster_to_ovr_subdomain0.size();
        // std::cout << count << std::endl;
        for (int i = 0; i < local_size_with_overlap; i++) {
            if (renum[i] == -1) {
                renum[i]                           = count;
                local_to_global_numbering[count++] = ovr_subdomain_to_global0[i];
            }
        }

        m_intersections.resize(neighbors0.size());
        for (int i = 0; i < neighbors0.size(); i++) {
            m_intersections[i].resize(input_intersections0[i].size());
            for (int j = 0; j < m_intersections[i].size(); j++) {
                m_intersections[i][j] = renum[input_intersections0[i][j]];
            }
        }

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
    Matrix<CoefficientPrecision> m_block_diagonal_dense_matrix;

  public:
    DDM<CoefficientPrecision> solver;
    std::vector<int> local_to_global_numbering;

    DefaultDDMSolverBuilder(DistributedOperator<CoefficientPrecision> &distributed_operator, const HMatrix<CoefficientPrecision, CoordinatePrecision> *block_diagonal_hmatrix, const VirtualGeneratorWithPermutation<CoefficientPrecision> &generator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) : m_intersections(intersections), m_block_diagonal_dense_matrix(initialize_diagonal_block(distributed_operator, block_diagonal_hmatrix, generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections)), solver(distributed_operator, m_block_diagonal_dense_matrix, neighbors, m_intersections) {}
};

} // namespace htool
#endif
