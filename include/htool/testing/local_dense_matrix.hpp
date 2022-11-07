
#ifndef HTOOL_TESTING_LOCAL_DENSE_MATRIX_HPP
#define HTOOL_TESTING_LOCAL_DENSE_MATRIX_HPP

#include "../basic_types/matrix.hpp"
#include "../interfaces/hmatrix/virtual_generator.hpp"
#include "../interfaces/virtual_local_operator.hpp"

namespace htool {

template <typename T>
class LocalDenseMatrix : public VirtualLocalOperator<T> {
  private:
    std::shared_ptr<VirtualCluster> m_cluster_tree_target, m_cluster_tree_source;
    Matrix<T> m_data;
    char m_symmetry{'N'};
    char m_UPLO{'N'};
    bool m_target_use_permutation_to_build{true}; // Permutation used when building m_data
    bool m_source_use_permutation_to_build{true}; // Permutation used when building m_data

    bool m_target_use_permutation_to_mvprod{false}; // Permutation used when add_mvprod, useful for offdiag
    bool m_source_use_permutation_to_mvprod{false}; // Permutation used when add_mvprod, useful for offdiag

    int m_source_offset{0};
    int m_target_offset{0};

  public:
    LocalDenseMatrix(const VirtualGenerator<T> &mat, std::shared_ptr<VirtualCluster> cluster_tree_target, std::shared_ptr<VirtualCluster> cluster_tree_source, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_build = true, bool source_use_permutation_to_build = true, bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : m_cluster_tree_target(cluster_tree_target), m_cluster_tree_source(cluster_tree_source), m_data(cluster_tree_target->get_size(), cluster_tree_source->get_size()), m_symmetry(symmetry), m_UPLO(UPLO), m_target_use_permutation_to_build(target_use_permutation_to_build), m_source_use_permutation_to_build(source_use_permutation_to_build), m_target_use_permutation_to_mvprod(target_use_permutation_to_mvprod), m_source_use_permutation_to_mvprod(source_use_permutation_to_mvprod) {

        std::vector<int> source_iota(!source_use_permutation_to_build ? cluster_tree_source->get_size() : 0);
        std::vector<int> target_iota(!target_use_permutation_to_build ? cluster_tree_target->get_size() : 0);
        std::iota(source_iota.begin(), source_iota.end(), int(cluster_tree_source->get_offset()));
        std::iota(target_iota.begin(), target_iota.end(), int(cluster_tree_target->get_offset()));
        int *col = (source_use_permutation_to_build ? cluster_tree_source->get_perm_data() : source_iota.data());
        int *row = (target_use_permutation_to_build ? cluster_tree_target->get_perm_data() : target_iota.data());
        int rankWorld;
        MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
        if (m_symmetry == 'N') {
            mat.copy_submatrix(m_data.nb_rows(), m_data.nb_cols(), row, col, m_data.data());
        } else if ((m_symmetry == 'S' || m_symmetry == 'H') && m_UPLO == 'L') {
            for (int i = 0; i < m_data.nb_rows(); i++) {
                for (int j = 0; j < i + 1; j++) {
                    mat.copy_submatrix(1, 1, &(row[i]), &(col[j]), m_data.data() + i + j * m_data.nb_rows());
                }
            }
        } else if ((m_symmetry == 'S' || m_symmetry == 'H') && m_UPLO == 'U') {
            for (int j = 0; j < m_data.nb_cols(); j++) {
                for (int i = 0; i < j + 1; i++) {
                    mat.copy_submatrix(1, 1, &(row[i]), &(col[j]), m_data.data() + i + j * m_data.nb_rows());
                }
            }
        }
    };

    // -- Operations --
    void add_vector_product_global_to_local(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out) const override {

        // Permutation
        std::vector<T> buffer_in(m_source_use_permutation_to_mvprod ? m_cluster_tree_source->get_size() : 0);
        std::vector<T> buffer_out(m_target_use_permutation_to_mvprod ? m_cluster_tree_target->get_size() : 0);
        if (m_source_use_permutation_to_mvprod) {
            global_to_cluster(m_cluster_tree_source.get(), in + m_cluster_tree_source->get_offset(), buffer_in.data());
        }

        const T *input = m_source_use_permutation_to_mvprod ? buffer_in.data() : in + m_cluster_tree_source->get_offset();
        T *output      = m_target_use_permutation_to_mvprod ? buffer_out.data() : out;

        // Local to local product
        if (m_symmetry == 'N') {
            m_data.add_vector_product('N', alpha, input, beta, output);
        } else if (m_symmetry == 'S' || m_symmetry == 'H') {
            m_data.add_vector_product_symmetric(alpha, input, beta, output, m_UPLO, m_symmetry);
        } else {
            throw std::invalid_argument("[Htool error] Invalid arguments for LocalDenseMatrix");
        }

        // Permutation
        if (m_target_use_permutation_to_mvprod) {
            cluster_to_global(m_cluster_tree_target.get(), output, out);
        }
    };
    void add_matrix_product_global_to_local(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out, int mu) const override {

        // std::vector<T> in_perm(std::max(m_cluster_tree_source->get_size(), m_cluster_tree_target->get_size()) * mu * 2);
        // std::vector<T> out_perm(m_cluster_tree_target->get_size() * mu);
        // std::vector<T> buffer_in(m_cluster_tree_source->get_size());

        // for (int i = 0; i < mu; i++) {
        //     if (m_source_use_permutation_to_mvprod) {
        //         global_to_cluster(m_cluster_tree_source.get(), in + m_cluster_tree_source->get_offset() + size_in * i, buffer_in.data());
        //         // Transpose
        //         for (int j = 0; j < m_cluster_tree_source->get_size(); j++) {
        //             in_perm[i + j * mu] = buffer_in[j];
        //         }
        //     } else {
        //         for (int j = 0; j < m_cluster_tree_source->get_size(); j++) {
        //             in_perm[i + j * mu] = buffer_in[j + m_cluster_tree_source->get_offset() + size_in * i];
        //         }
        //     }
        // }

        // if (m_symmetry == 'H') {
        //     conj_if_complex(in_perm.data(), m_cluster_tree_source->get_size() * mu);
        // }

        // // Local to local product
        // if (m_symmetry == 'N') {
        //     m_data.add_matrix_product(alpha, in_perm.data(), beta, in_perm.data() + m_cluster_tree_source->get_size() * mu, mu);
        // } else if (m_symmetry == 'S' || m_symmetry == 'H') {
        //     m_data.add_matrix_product_symmetric(alpha, in_perm.data(), beta, in_perm.data() + m_cluster_tree_source->get_size() * mu, mu, m_UPLO, m_symmetry, 'R');
        // } else {
        //     throw std::invalid_argument("[Htool error] Invalid arguments for LocalDenseMatrix");
        // }
    }

    virtual void add_vector_product_transp_local_to_global(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out) const override {
        // Permutation
        // std::vector<T> buffer_in(m_source_use_permutation_to_mvprod ? m_cluster_tree_target->get_size() : 0);
        std::vector<T> buffer_out(m_source_use_permutation_to_mvprod ? m_cluster_tree_source->get_size() : 0);
        // m_data.print(std::cout, ",");
        // const T *input = m_source_use_permutation_to_mvprod ? buffer_in.data() : in + m_cluster_tree_source->get_offset();
        T *output = m_source_use_permutation_to_mvprod ? buffer_out.data() : out + m_cluster_tree_source->get_offset();

        // Local to local product
        if (m_symmetry == 'N') {
            m_data.add_vector_product('T', alpha, in, beta, output);
        } else if (m_symmetry == 'S' || m_symmetry == 'H') {
            // m_data.add_vector_product('T', alpha, in, beta, output);
            m_data.add_vector_product_symmetric(alpha, in, beta, output, m_UPLO, m_symmetry);
        } else {
            throw std::invalid_argument("[Htool error] Invalid arguments for LocalDenseMatrix");
        }

        // for (int i = 0; i < m_cluster_tree_source->get_size(); i++) {
        //     std::cout << output[i] << "\n";
        // }
        // std::cout << "\n";

        // Permutation
        if (m_source_use_permutation_to_mvprod) {
            global_to_cluster(m_cluster_tree_source.get(), output, out + m_cluster_tree_source->get_offset());
        }
    }

    virtual void add_matrix_product_transp_local_to_global(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out, int mu) const override {}

    void print() {
        m_data.print(std::cout, ",");
    }
};
} // namespace htool
#endif
