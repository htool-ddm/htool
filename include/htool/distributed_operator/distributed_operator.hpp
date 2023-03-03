#ifndef HTOOL_DISTRIBUTED_OPERATOR_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_HPP

#include "../clustering/cluster_node.hpp"
#include "../local_operators/virtual_local_operator.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "virtual_distributed_operator.hpp"

namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class DistributedOperator {

  private:
    //
    std::shared_ptr<const Cluster<CoordinatePrecision>> m_global_target_root_cluster;
    std::shared_ptr<const Cluster<CoordinatePrecision>> m_global_source_root_cluster;
    const Cluster<CoordinatePrecision> *m_local_cluster_target;
    const Cluster<CoordinatePrecision> *m_local_cluster_source;

    // Local operators
    std::vector<std::shared_ptr<VirtualLocalOperator<CoefficientPrecision>>> m_local_operators = {};

    // Properties
    bool m_use_permutation = true;
    char m_symmetry        = 'N';
    char m_UPLO            = 'N';
    MPI_Comm comm          = MPI_COMM_WORLD;

    //
    mutable std::map<std::string, std::string> infos;

  public:
    // Constructor
    DistributedOperator(std::shared_ptr<const Cluster<CoordinatePrecision>> global_cluster_tree_target, std::shared_ptr<const Cluster<CoordinatePrecision>> global_cluster_tree_source, char symmetry = 'N', char UPLO = 'N') : m_global_target_root_cluster(global_cluster_tree_target), m_global_source_root_cluster(global_cluster_tree_source), m_symmetry(symmetry), m_UPLO(UPLO) {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);
        m_local_cluster_target = m_global_target_root_cluster->get_clusters_on_partition()[rankWorld];
        m_local_cluster_source = m_global_source_root_cluster->get_clusters_on_partition()[rankWorld];
    };

    //
    void add_local_operator(std::shared_ptr<VirtualLocalOperator<CoefficientPrecision>> local_operator) {
        m_local_operators.push_back(local_operator);
    }

    // Operations using user numbering and column major input/output
    void vector_product_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void matrix_product_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const;
    void vector_product_transp_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void matrix_product_transp_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const;

    void vector_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work = nullptr) const;
    void matrix_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work = nullptr) const;
    void vector_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work = nullptr) const;
    void matrix_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work = nullptr) const;

    // Operations using internal numbering and row major input/output
    void internal_vector_product_global_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void internal_matrix_product_global_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const;
    void internal_vector_product_transp_local_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void internal_matrix_product_transp_local_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const;

    void internal_vector_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work = nullptr) const;
    void internal_matrix_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work = nullptr) const;
    void internal_vector_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work = nullptr) const;
    void internal_matrix_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work = nullptr) const;

    // // Special matrix-vector product for building coarse space
    // virtual void mvprod_subrhs(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu, const int &offset, const int &size, const int &margin) const = 0;

    // Permutations
    void source_to_cluster_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void target_to_cluster_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void cluster_to_target_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void cluster_to_source_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void local_target_to_local_cluster(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void local_source_to_local_cluster(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void local_cluster_to_local_target(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;
    void local_cluster_to_local_source(const CoefficientPrecision *const in, CoefficientPrecision *const out) const;

    // local to global
    void local_to_global_source(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const;
    void local_to_global_target(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const;

    // Getters/setters
    bool &use_permutation() { return m_use_permutation; }
    const bool &use_permutation() const { return m_use_permutation; }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::vector_product_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    double time = MPI_Wtime();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    int nr         = m_global_target_root_cluster->get_size();
    int nc         = m_global_source_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    std::vector<CoefficientPrecision> out_perm(local_size);
    std::vector<CoefficientPrecision> input_buffer(m_use_permutation ? nc : 0);
    std::vector<CoefficientPrecision> output_buffer(m_use_permutation ? nr : 0);

    // Permutation
    if (m_use_permutation) {
        this->source_to_cluster_permutation(in, input_buffer.data());
    }
    const CoefficientPrecision *input = m_use_permutation ? input_buffer.data() : in;

    // Product
    this->internal_vector_product_global_to_local(input, out_perm.data());
    // if (rankWorld == 0)
    //     std::cout << rankWorld << " " << out_perm << "\n";
    // if (rankWorld == 1)
    //     std::cout << rankWorld << " " << out_perm << "\n";
    // Allgather
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = m_global_target_root_cluster->get_clusters_on_partition()[i]->get_size();
        // if (rankWorld == 0)
        //     std::cout << recvcounts[i] << "\n";
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    if (!m_use_permutation) {
        MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
    } else if (m_use_permutation) {
        MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), output_buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
        this->cluster_to_target_permutation(output_buffer.data(), out);
    }

    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::matrix_product_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    double time = MPI_Wtime();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    int nr         = m_global_target_root_cluster->get_size();
    int nc         = m_global_source_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    std::vector<CoefficientPrecision> out_perm(2 * local_size * mu, 0);
    std::vector<CoefficientPrecision> input_buffer_transpose(m_use_permutation ? nc : 0);
    std::vector<CoefficientPrecision> input_buffer(nc * mu);
    std::vector<CoefficientPrecision> output_buffer(m_use_permutation ? 2 * nr * mu : nr * mu, 0);

    // Permutation and column major to row major
    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            this->source_to_cluster_permutation(in + i * nc, input_buffer_transpose.data());
            for (int j = 0; j < nc; j++) {
                input_buffer[i + j * mu] = input_buffer_transpose[j];
            }

        } else {
            // Transpose
            for (int j = 0; j < nc; j++) {
                input_buffer[i + j * mu] = in[j + i * nc];
            }
        }
    }

    // Product
    this->internal_matrix_product_global_to_local(input_buffer.data(), out_perm.data() + local_size * mu, mu);

    // Transpose
    for (int i = 0; i < mu; i++) {
        for (int j = 0; j < local_size; j++) {
            out_perm[i * local_size + j] = out_perm[local_size * mu + i + j * mu];
        }
    }

    // Allgather
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = m_global_target_root_cluster->get_clusters_on_partition()[i]->get_size() * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), output_buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            for (int j = 0; j < sizeWorld; j++) {
                std::copy_n(output_buffer.data() + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, output_buffer.data() + nr * mu + i * nr + displs[j] / mu);
            }

            // Permutation
            this->cluster_to_target_permutation(output_buffer.data() + nr * mu + i * nr, out + i * nr);
        } else {
            for (int j = 0; j < sizeWorld; j++) {
                std::copy_n(output_buffer.data() + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, out + i * nr + displs[j] / mu);
            }
        }
    }
    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::vector_product_transp_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int nr         = m_global_target_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    int nc         = m_global_source_root_cluster->get_size();

    if (m_symmetry == 'S') {
        this->vector_product_global_to_global(in, out);
        return;
    } else if (m_symmetry == 'H') {
        std::vector<CoefficientPrecision> in_conj(in, in + nr);
        conj_if_complex<CoefficientPrecision>(in_conj.data(), nr);
        this->vector_product_global_to_global(in_conj.data(), out);
        conj_if_complex<CoefficientPrecision>(out, nc);
        return;
    }

    double time = MPI_Wtime();
    std::vector<CoefficientPrecision> in_perm(m_use_permutation ? nr : 0);
    std::vector<CoefficientPrecision> out_perm(m_use_permutation ? nc : 0);
    if (m_use_permutation) {
        this->target_to_cluster_permutation(in, in_perm.data());
    }

    const CoefficientPrecision *input = m_use_permutation ? in_perm.data() : in;
    CoefficientPrecision *output      = m_use_permutation ? out_perm.data() : out;

    // Product
    this->internal_vector_product_transp_local_to_global(input + m_local_cluster_target->get_offset(), output);

    if (m_use_permutation) {
        this->cluster_to_source_permutation(output, out);
    }

    // Timing
    infos["nb_mat_vec_prod_transp"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod_transp"]));
    infos["total_time_mat_vec_prod_transp"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod_transp"]));
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::matrix_product_transp_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    int nr           = m_global_target_root_cluster->get_size();
    int local_size   = m_local_cluster_target->get_size();
    int local_offset = m_local_cluster_target->get_offset();
    int nc           = m_global_source_root_cluster->get_size();

    if (m_symmetry == 'S') {
        this->matrix_product_global_to_global(in, out, mu);
        return;
    } else if (m_symmetry == 'H') {
        std::vector<CoefficientPrecision> in_conj(in, in + nr * mu);
        conj_if_complex(in_conj.data(), nr * mu);
        this->matrix_product_global_to_global(in_conj.data(), out, mu);
        conj_if_complex(out, mu * nc);
        return;
    }

    std::vector<CoefficientPrecision> out_perm(mu * nc);
    std::vector<CoefficientPrecision> in_perm(local_size * mu + mu * nc);
    std::vector<CoefficientPrecision> buffer(nr);

    for (int i = 0; i < mu; i++) {
        // Permutation
        if (m_use_permutation) {
            this->target_to_cluster_permutation(in + i * nr, buffer.data());
            // Transpose
            for (int j = local_offset; j < local_offset + local_size; j++) {
                in_perm[i + (j - local_offset) * mu] = buffer[j];
            }
        } else {
            // Transpose
            for (int j = local_offset; j < local_offset + local_size; j++) {
                in_perm[i + (j - local_offset) * mu] = in[j + i * nr];
            }
        }
    }

    internal_matrix_product_transp_local_to_global(in_perm.data(), in_perm.data() + local_size * mu, mu);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            // Transpose
            for (int j = 0; j < nc; j++) {
                out_perm[i * nc + j] = in_perm[i + j * mu + local_size * mu];
            }
            cluster_to_source_permutation(out_perm.data() + i * nc, out + i * nc);
        } else {
            for (int j = 0; j < nc; j++) {
                out[i * nc + j] = in_perm[i + j * mu + local_size * mu];
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::vector_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {

    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    int local_size_source = m_local_cluster_source->get_size();
    int local_size        = m_local_cluster_target->get_size();
    int nc                = m_global_source_root_cluster->get_size();
    bool need_delete      = false;
    if (work == nullptr) {
        work        = new CoefficientPrecision[nc];
        need_delete = true;
    }
    if (!(m_global_source_root_cluster->is_permutation_local()) || !(m_global_target_root_cluster->is_permutation_local())) {
        throw std::logic_error("[Htool error] Permutation is not local, vector_product_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    std::vector<CoefficientPrecision> in_perm(m_use_permutation ? local_size_source : 0);
    std::vector<CoefficientPrecision> out_perm(m_use_permutation ? local_size : 0);
    const CoefficientPrecision *input = m_use_permutation ? in_perm.data() : in;
    CoefficientPrecision *output      = m_use_permutation ? out_perm.data() : out;

    if (m_use_permutation) {
        this->local_source_to_local_cluster(in, in_perm.data());
    }

    this->internal_vector_product_local_to_local(input, output, work);

    if (m_use_permutation) {
        this->local_cluster_to_local_target(output, out);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::matrix_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {
    bool need_delete = false;
    if (work == nullptr) {
        work        = new CoefficientPrecision[this->m_global_source_root_cluster->get_size() * mu];
        need_delete = true;
    }

    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    int local_size_source = m_local_cluster_source->get_size();
    int local_size        = m_local_cluster_target->get_size();

    if (!(m_global_source_root_cluster->is_permutation_local()) || !(m_global_target_root_cluster->is_permutation_local())) {
        throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    std::vector<CoefficientPrecision> input_buffer(local_size_source * mu);
    std::vector<CoefficientPrecision> input_buffer_transpose(m_use_permutation ? local_size_source : 0);
    std::vector<CoefficientPrecision> output_buffer_transpose(m_use_permutation ? local_size : 0);
    std::vector<CoefficientPrecision> output_perm(local_size * mu);

    // Permutation
    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            this->local_source_to_local_cluster(in + i * local_size_source, input_buffer_transpose.data());
            for (int j = 0; j < local_size_source; j++) {
                input_buffer[i + j * mu] = input_buffer_transpose[j];
            }
        } else {
            // Transpose
            for (int j = 0; j < local_size_source; j++) {
                input_buffer[i + j * mu] = in[j + i * local_size_source];
            }
        }
    }

    this->internal_matrix_product_local_to_local(input_buffer.data(), output_perm.data(), mu, work);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            // Transpose
            for (int j = 0; j < local_size; j++) {
                output_buffer_transpose[j] = output_perm[i + j * mu];
            }
            this->local_cluster_to_local_target(output_buffer_transpose.data(), out + i * local_size);
        } else {
            // Transpose
            for (int j = 0; j < local_size; j++) {
                out[j + i * local_size] = output_perm[i + j * mu];
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::vector_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {
    int local_size_source = m_local_cluster_source->get_size();
    int local_size        = m_local_cluster_target->get_size();
    int nc                = m_global_source_root_cluster->get_size();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);

    if (m_symmetry == 'S') {
        this->vector_product_local_to_local(in, out, work);
        return;
    } else if (m_symmetry == 'H') {
        std::vector<CoefficientPrecision> in_conj(in, in + local_size);
        conj_if_complex<CoefficientPrecision>(in_conj.data(), local_size);
        this->vector_product_local_to_local(in_conj.data(), out, work);
        conj_if_complex<CoefficientPrecision>(out, local_size_source);
        return;
    }

    bool need_delete = false;
    if (work == nullptr) {
        work        = new CoefficientPrecision[(nc + sizeWorld * m_local_cluster_source->get_size())];
        need_delete = true;
    }

    if (!(m_global_source_root_cluster->is_permutation_local()) || !(m_global_target_root_cluster->is_permutation_local())) {
        throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    std::vector<CoefficientPrecision> in_perm(m_use_permutation ? local_size : 0);
    std::vector<CoefficientPrecision> out_perm(m_use_permutation ? local_size_source : 0);
    const CoefficientPrecision *input = m_use_permutation ? in_perm.data() : in;
    CoefficientPrecision *output      = m_use_permutation ? out_perm.data() : out;

    // local permutation
    if (m_use_permutation) {
        this->local_target_to_local_cluster(in, in_perm.data());
    }
    // prod
    internal_vector_product_transp_local_to_local(input, output, work);

    // permutation
    if (m_use_permutation) {
        this->local_cluster_to_local_source(output, out);
    }
}
template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::matrix_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {
    int local_size_source = m_local_cluster_source->get_size();
    int local_size        = m_local_cluster_target->get_size();
    int nc                = m_global_source_root_cluster->get_size();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);

    if (m_symmetry == 'S') {
        this->matrix_product_local_to_local(in, out, mu, work);
        return;
    } else if (m_symmetry == 'H') {
        std::vector<CoefficientPrecision> in_conj(in, in + local_size * mu);
        conj_if_complex<CoefficientPrecision>(in_conj.data(), local_size * mu);
        this->matrix_product_local_to_local(in_conj.data(), out, mu, work);
        conj_if_complex<CoefficientPrecision>(out, local_size_source * mu);
        return;
    }

    bool need_delete = false;
    if (work == nullptr) {
        work        = new CoefficientPrecision[(nc + sizeWorld * m_local_cluster_source->get_size()) * mu];
        need_delete = true;
    }

    if (!(m_global_source_root_cluster->is_permutation_local()) || !(m_global_target_root_cluster->is_permutation_local())) {
        throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }
    std::vector<CoefficientPrecision> in_perm(local_size * mu);
    std::vector<CoefficientPrecision> out_perm(local_size_source * mu);
    std::vector<CoefficientPrecision> buffer_in(m_use_permutation ? local_size : 0);
    std::vector<CoefficientPrecision> buffer_out(m_use_permutation ? local_size_source : 0);

    for (int i = 0; i < mu; i++) {
        // local permutation
        if (m_use_permutation) {
            this->local_target_to_local_cluster(in + i * local_size, buffer_in.data());

            // Transpose
            for (int j = 0; j < local_size; j++) {
                in_perm[i + j * mu] = buffer_in[j];
            }
        } else {
            // Transpose
            for (int j = 0; j < local_size; j++) {
                in_perm[i + j * mu] = in[j + i * local_size];
            }
        }
    }
    // It should never happen since we use mvprod_global_to_global in this case
    // if (symmetry == 'H') {
    //     conj_if_complex(in_perm.data(), local_size_source * mu);
    // }

    internal_matrix_product_transp_local_to_local(in_perm.data(), out_perm.data(), mu, work);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            // Tranpose
            for (int j = 0; j < local_size_source; j++) {
                buffer_out[j] = out_perm[i + j * mu];
            }

            // local permutation
            this->local_cluster_to_local_source(buffer_out.data(), out + i * local_size_source);
        } else {
            // Tranpose
            for (int j = 0; j < local_size_source; j++) {
                out[j + i * local_size_source] = out_perm[i + j * mu];
            }
        }
    }
}

// Operations using internal numbering
template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_vector_product_global_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int nc         = m_global_source_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    for (auto &local_operator : m_local_operators) {
        local_operator->add_vector_product_global_to_local(1, in, 1, out);
    }
}
template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_matrix_product_global_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    int nc         = m_global_source_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    for (auto &local_operator : m_local_operators) {
        local_operator->add_matrix_product_global_to_local(1, in, 1, out, mu);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_vector_product_transp_local_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int nc         = m_global_source_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    std::fill(out, out + nc, 0);
    for (auto &local_operator : m_local_operators) {
        local_operator->add_vector_product_transp_local_to_global(1, in, 1, out);
    }

    MPI_Allreduce(MPI_IN_PLACE, out, nc, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_matrix_product_transp_local_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    int nc         = m_global_source_root_cluster->get_size();
    int local_size = m_local_cluster_target->get_size();
    std::fill(out, out + nc * mu, 0);
    for (auto &local_operator : m_local_operators) {
        local_operator->add_matrix_product_transp_local_to_global(1, in, 1, out, mu);
    }
    MPI_Allreduce(MPI_IN_PLACE, out, nc * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_vector_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {
    bool need_delete = false;
    if (work == nullptr) {
        work        = new CoefficientPrecision[m_global_source_root_cluster->get_size()];
        need_delete = true;
    }

    this->local_to_global_source(in, work, 1);
    this->internal_vector_product_global_to_local(work, out);

    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_matrix_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {
    bool need_delete = false;
    if (work == nullptr) {
        work        = new CoefficientPrecision[m_global_source_root_cluster->get_size() * mu];
        need_delete = true;
    }

    this->local_to_global_source(in, work, mu);
    this->internal_matrix_product_global_to_local(work, out, mu);

    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_vector_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {
    if (m_symmetry == 'S' || m_symmetry == 'H') {
        this->internal_vector_product_local_to_local(in, out, work);
        return;
    }

    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    bool need_delete      = false;
    int local_size_source = m_local_cluster_source->get_size();
    int nc                = m_global_source_root_cluster->get_size();
    int local_size        = m_local_cluster_target->get_size();

    if (work == nullptr) {
        work        = new CoefficientPrecision[(nc + local_size_source * sizeWorld)];
        need_delete = true;
    }
    std::fill(out, out + local_size_source, 0);
    std::fill(work, work + nc, 0);
    CoefficientPrecision *rbuf = work + nc;

    for (auto &local_operator : m_local_operators) {
        local_operator->add_vector_product_transp_local_to_global(1, in, 1, work);
    }

    std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
    std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        scounts[i] = (m_global_source_root_cluster->get_clusters_on_partition()[i]->get_size());
        rcounts[i] = (local_size_source);
        if (i > 0) {
            sdispls[i] = sdispls[i - 1] + scounts[i - 1];
            rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        }
    }

    MPI_Alltoallv(work, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

    for (int i = 0; i < sizeWorld; i++)
        std::transform(out, out + local_size_source, rbuf + rdispls[i], out, std::plus<CoefficientPrecision>());

    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::internal_matrix_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {
    if (m_symmetry == 'S' || m_symmetry == 'H') {
        this->internal_matrix_product_local_to_local(in, out, mu, work);
        return;
    }

    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    bool need_delete      = false;
    int local_size_source = m_local_cluster_source->get_size();
    int nc                = m_global_source_root_cluster->get_size();
    int local_size        = m_local_cluster_target->get_size();

    if (work == nullptr) {
        work        = new CoefficientPrecision[(nc + local_size_source * sizeWorld) * mu];
        need_delete = true;
    }
    std::fill(work, work + nc * mu, 0);
    CoefficientPrecision *rbuf = work + nc * mu;

    for (auto &local_operator : m_local_operators) {
        local_operator->add_matrix_product_transp_local_to_global(1, in, 1, work, mu);
    }

    std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
    std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        scounts[i] = (m_global_source_root_cluster->get_clusters_on_partition()[i]->get_size()) * mu;
        rcounts[i] = (local_size_source)*mu;
        if (i > 0) {
            sdispls[i] = sdispls[i - 1] + scounts[i - 1];
            rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        }
    }

    MPI_Alltoallv(work, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

    for (int i = 0; i < sizeWorld; i++)
        std::transform(out, out + local_size_source * mu, rbuf + rdispls[i], out, std::plus<CoefficientPrecision>());
    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
}

// Permutations
template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::source_to_cluster_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    global_to_root_cluster(*m_global_source_root_cluster, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::target_to_cluster_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    global_to_root_cluster(*m_global_target_root_cluster, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::cluster_to_target_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    root_cluster_to_global(*m_global_target_root_cluster, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::cluster_to_source_permutation(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    root_cluster_to_global(*m_global_source_root_cluster, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::local_target_to_local_cluster(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    local_to_local_cluster(*m_global_target_root_cluster, rankWorld, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::local_source_to_local_cluster(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    local_to_local_cluster(*m_global_source_root_cluster, rankWorld, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::local_cluster_to_local_target(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    local_cluster_to_local(*m_global_target_root_cluster, rankWorld, in, out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::local_cluster_to_local_source(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    local_cluster_to_local(*m_global_source_root_cluster, rankWorld, in, out);
}

// Local to global
template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::local_to_global_target(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
    // Allgather
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = (m_global_target_root_cluster->partition()[i].second) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void DistributedOperator<CoefficientPrecision, CoordinatePrecision>::local_to_global_source(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
    // Allgather
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = (m_global_source_root_cluster->get_clusters_on_partition()[i]->get_size()) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
}

} // namespace htool

#endif
