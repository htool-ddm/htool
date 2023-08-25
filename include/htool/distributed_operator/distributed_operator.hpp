#ifndef HTOOL_DISTRIBUTED_OPERATOR_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_HPP

#include "../local_operators/local_hmatrix.hpp"
#include "../local_operators/virtual_local_operator.hpp"
#include "../misc/logger.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "interfaces/partition.hpp"
#include <map>

namespace htool {
template <typename CoefficientPrecision>
class DistributedOperator {

  private:
    //
    const IPartition<CoefficientPrecision> &m_target_partition;
    const IPartition<CoefficientPrecision> &m_source_partition;

    // Local operators
    std::vector<const VirtualLocalOperator<CoefficientPrecision> *> m_local_operators = {};

    // Properties
    bool m_use_permutation = true;
    char m_symmetry        = 'N';
    char m_UPLO            = 'N';
    MPI_Comm m_comm        = MPI_COMM_WORLD;

    //
    mutable std::map<std::string, std::string> infos;

  public:
    // no copy
    DistributedOperator(const DistributedOperator &)                       = delete;
    DistributedOperator &operator=(const DistributedOperator &)            = delete;
    DistributedOperator(DistributedOperator &&cluster) noexcept            = default;
    DistributedOperator &operator=(DistributedOperator &&cluster) noexcept = default;
    virtual ~DistributedOperator()                                         = default;

    // Constructor
    explicit DistributedOperator(const IPartition<CoefficientPrecision> &target_partition, const IPartition<CoefficientPrecision> &source_partition, char symmetry, char UPLO, MPI_Comm comm) : m_target_partition(target_partition), m_source_partition(source_partition), m_symmetry(symmetry), m_UPLO(UPLO), m_comm(comm) {}

    void add_local_operator(const VirtualLocalOperator<CoefficientPrecision> *local_operator) {
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

    // Special matrix-vector product for building coarse space
    void internal_sub_matrix_product_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) const;

    // local to global
    void local_to_global_source(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const;
    void local_to_global_target(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const;

    // Getters
    bool &use_permutation() { return m_use_permutation; }
    const bool &use_permutation() const { return m_use_permutation; }
    char get_symmetry_type() const { return m_symmetry; }
    char get_storage_type() const { return m_UPLO; }
    MPI_Comm get_comm() const { return m_comm; }
    const IPartition<CoefficientPrecision> &get_target_partition() const { return m_target_partition; }
    const IPartition<CoefficientPrecision> &get_source_partition() const { return m_source_partition; }
};

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::vector_product_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    double time = MPI_Wtime();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    // int nr = m_global_target_root_cluster->get_size();
    // int nc = m_global_source_root_cluster->get_size();
    // int local_size = m_local_cluster_target->get_size();
    int nr         = m_target_partition.get_global_size();
    int nc         = m_source_partition.get_global_size();
    int local_size = m_target_partition.get_size_of_partition(rankWorld);
    std::vector<CoefficientPrecision> out_perm(local_size);
    std::vector<CoefficientPrecision> input_buffer(m_use_permutation ? nc : 0);
    std::vector<CoefficientPrecision> output_buffer(m_use_permutation ? nr : 0);

    // Permutation
    if (m_use_permutation) {
        m_source_partition.global_to_partition_numbering(in, input_buffer.data());
        // this->source_to_cluster_permutation(in, input_buffer.data());
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
        // recvcounts[i] = m_global_target_root_cluster->get_clusters_on_partition()[i]->get_size();
        recvcounts[i] = m_target_partition.get_size_of_partition(i);
        // if (rankWorld == 0)
        //     std::cout << recvcounts[i] << "\n";
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    if (!m_use_permutation) {
        MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);
    } else if (m_use_permutation) {
        MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), output_buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);
        m_target_partition.partition_to_global_numbering(output_buffer.data(), out);
        // this->cluster_to_target_permutation(output_buffer.data(), out);
    }

    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::matrix_product_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    double time = MPI_Wtime();
    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    // int nr = m_global_target_root_cluster->get_size();
    // int nc = m_global_source_root_cluster->get_size();
    // int local_size = m_local_cluster_target->get_size();
    int nr         = m_target_partition.get_global_size();
    int nc         = m_source_partition.get_global_size();
    int local_size = m_target_partition.get_size_of_partition(rankWorld);
    std::vector<CoefficientPrecision> out_perm(2 * local_size * mu, 0);
    std::vector<CoefficientPrecision> input_buffer_transpose(m_use_permutation ? nc : 0);
    std::vector<CoefficientPrecision> input_buffer(nc * mu);
    std::vector<CoefficientPrecision> output_buffer(m_use_permutation ? 2 * nr * mu : nr * mu, 0);

    // Permutation and column major to row major
    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            m_source_partition.global_to_partition_numbering(in + i * nc, input_buffer_transpose.data());
            // this->source_to_cluster_permutation(in + i * nc, input_buffer_transpose.data());
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
        // recvcounts[i] = m_global_target_root_cluster->get_clusters_on_partition()[i]->get_size() * mu;
        recvcounts[i] = m_target_partition.get_size_of_partition(i) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), output_buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            for (int j = 0; j < sizeWorld; j++) {
                std::copy_n(output_buffer.data() + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, output_buffer.data() + nr * mu + i * nr + displs[j] / mu);
            }

            // Permutation
            // m_target_partition.this->cluster_to_target_permutation(output_buffer.data() + nr * mu + i * nr, out + i * nr);
            m_target_partition.partition_to_global_numbering(output_buffer.data() + nr * mu + i * nr, out + i * nr);
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

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::vector_product_transp_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    // int nr         = m_global_target_root_cluster->get_size();
    // int local_size = m_local_cluster_target->get_size();
    // int nc         = m_global_source_root_cluster->get_size();
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int nr = m_target_partition.get_global_size();
    int nc = m_source_partition.get_global_size();

    int local_offset = m_target_partition.get_offset_of_partition(rankWorld);

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
        m_target_partition.global_to_partition_numbering(in, in_perm.data());
        // this->target_to_cluster_permutation(in, in_perm.data());
    }

    const CoefficientPrecision *input = m_use_permutation ? in_perm.data() : in;
    CoefficientPrecision *output      = m_use_permutation ? out_perm.data() : out;

    // Product
    this->internal_vector_product_transp_local_to_global(input + local_offset, output);

    if (m_use_permutation) {
        m_source_partition.partition_to_global_numbering(output, out);
        // this->cluster_to_source_permutation(output, out);
    }

    // Timing
    infos["nb_mat_vec_prod_transp"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod_transp"]));
    infos["total_time_mat_vec_prod_transp"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod_transp"]));
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::matrix_product_transp_global_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int nr           = m_target_partition.get_global_size();
    int nc           = m_source_partition.get_global_size();
    int local_size   = m_target_partition.get_size_of_partition(rankWorld);
    int local_offset = m_target_partition.get_offset_of_partition(rankWorld);

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
            m_target_partition.global_to_partition_numbering(in + i * nr, buffer.data());
            // this->target_to_cluster_permutation(in + i * nr, buffer.data());
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
            m_source_partition.partition_to_global_numbering(out_perm.data() + i * nc, out + i * nc);
            // cluster_to_source_permutation(out_perm.data() + i * nc, out + i * nc);
        } else {
            for (int j = 0; j < nc; j++) {
                out[i * nc + j] = in_perm[i + j * mu + local_size * mu];
            }
        }
    }
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::vector_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int nc                = m_source_partition.get_global_size();
    int local_size        = m_target_partition.get_size_of_partition(rankWorld);
    int local_size_source = m_source_partition.get_size_of_partition(rankWorld);

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? nc : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    if ((!(m_source_partition.is_renumbering_local()) || !(m_target_partition.is_renumbering_local())) && m_use_permutation) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation is not local, vector_product_local_to_local cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Permutation is not local, vector_product_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    std::vector<CoefficientPrecision> in_perm(m_use_permutation ? local_size_source : 0);
    std::vector<CoefficientPrecision> out_perm(m_use_permutation ? local_size : 0);
    const CoefficientPrecision *input = m_use_permutation ? in_perm.data() : in;
    CoefficientPrecision *output      = m_use_permutation ? out_perm.data() : out;

    if (m_use_permutation) {
        // this->local_source_to_local_cluster(in, in_perm.data());
        m_source_partition.local_to_local_partition_numbering(rankWorld, in, in_perm.data());
    }

    this->internal_vector_product_local_to_local(input, output, buffer_ptr);

    if (m_use_permutation) {
        // this->local_cluster_to_local_target(output, out);
        m_target_partition.local_partition_to_local_numbering(rankWorld, output, out);
    }
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::matrix_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? m_source_partition.get_global_size() * mu : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int local_size        = m_target_partition.get_size_of_partition(rankWorld);
    int local_size_source = m_source_partition.get_size_of_partition(rankWorld);

    if ((!(m_source_partition.is_renumbering_local()) || !(m_target_partition.is_renumbering_local())) && m_use_permutation) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    std::vector<CoefficientPrecision> input_buffer(local_size_source * mu);
    std::vector<CoefficientPrecision> input_buffer_transpose(m_use_permutation ? local_size_source : 0);
    std::vector<CoefficientPrecision> output_buffer_transpose(m_use_permutation ? local_size : 0);
    std::vector<CoefficientPrecision> output_perm(local_size * mu);

    // Permutation
    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            m_source_partition.local_to_local_partition_numbering(rankWorld, in + i * local_size_source, input_buffer_transpose.data());
            // this->local_source_to_local_cluster(in + i * local_size_source, input_buffer_transpose.data());
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

    this->internal_matrix_product_local_to_local(input_buffer.data(), output_perm.data(), mu, buffer_ptr);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            // Transpose
            for (int j = 0; j < local_size; j++) {
                output_buffer_transpose[j] = output_perm[i + j * mu];
            }
            // this->local_cluster_to_local_target(output_buffer_transpose.data(), out + i * local_size);
            m_target_partition.local_partition_to_local_numbering(rankWorld, output_buffer_transpose.data(), out + i * local_size);
        } else {
            // Transpose
            for (int j = 0; j < local_size; j++) {
                out[j + i * local_size] = output_perm[i + j * mu];
            }
        }
    }
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::vector_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {
    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    int local_size        = m_target_partition.get_size_of_partition(rankWorld);
    int local_size_source = m_source_partition.get_size_of_partition(rankWorld);
    int nc                = m_source_partition.get_global_size();

    // int local_size_source = m_local_cluster_source->get_size();
    // int local_size        = m_local_cluster_target->get_size();
    // int nc                = m_global_source_root_cluster->get_size();
    // int sizeWorld, rankWorld;
    // MPI_Comm_rank(m_comm, &rankWorld);
    // MPI_Comm_size(m_comm, &sizeWorld);

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? nc + sizeWorld * local_size_source : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    if (m_symmetry == 'S') {
        this->vector_product_local_to_local(in, out, buffer_ptr);
        return;
    } else if (m_symmetry == 'H') {
        std::vector<CoefficientPrecision> in_conj(in, in + local_size);
        conj_if_complex<CoefficientPrecision>(in_conj.data(), local_size);
        this->vector_product_local_to_local(in_conj.data(), out, buffer_ptr);
        conj_if_complex<CoefficientPrecision>(out, local_size_source);
        return;
    }

    if ((!(m_source_partition.is_renumbering_local()) || !(m_target_partition.is_renumbering_local())) && m_use_permutation) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    std::vector<CoefficientPrecision> in_perm(m_use_permutation ? local_size : 0);
    std::vector<CoefficientPrecision> out_perm(m_use_permutation ? local_size_source : 0);
    const CoefficientPrecision *input = m_use_permutation ? in_perm.data() : in;
    CoefficientPrecision *output      = m_use_permutation ? out_perm.data() : out;

    // local permutation
    if (m_use_permutation) {
        m_target_partition.local_to_local_partition_numbering(rankWorld, in, in_perm.data());
        // this->local_target_to_local_cluster(in, in_perm.data());
    }
    // prod
    internal_vector_product_transp_local_to_local(input, output, buffer_ptr);

    // permutation
    if (m_use_permutation) {
        m_source_partition.local_partition_to_local_numbering(rankWorld, output, out);
        // this->local_cluster_to_local_source(output, out);
    }
}
template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::matrix_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {
    // int local_size_source = m_local_cluster_source->get_size();
    // int local_size        = m_local_cluster_target->get_size();
    // int nc                = m_global_source_root_cluster->get_size();
    // int sizeWorld, rankWorld;
    // MPI_Comm_rank(m_comm, &rankWorld);
    // MPI_Comm_size(m_comm, &sizeWorld);

    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    int local_size        = m_target_partition.get_size_of_partition(rankWorld);
    int local_size_source = m_source_partition.get_size_of_partition(rankWorld);
    int nc                = m_source_partition.get_global_size();

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? (nc + sizeWorld * local_size_source) * mu : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    if (m_symmetry == 'S') {
        this->matrix_product_local_to_local(in, out, mu, buffer_ptr);
        return;
    } else if (m_symmetry == 'H') {
        std::vector<CoefficientPrecision> in_conj(in, in + local_size * mu);
        conj_if_complex<CoefficientPrecision>(in_conj.data(), local_size * mu);
        this->matrix_product_local_to_local(in_conj.data(), out, mu, buffer_ptr);
        conj_if_complex<CoefficientPrecision>(out, local_size_source * mu);
        return;
    }

    if ((!(m_source_partition.is_renumbering_local()) || !(m_target_partition.is_renumbering_local())) && m_use_permutation) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }
    std::vector<CoefficientPrecision> in_perm(local_size * mu);
    std::vector<CoefficientPrecision> out_perm(local_size_source * mu);
    std::vector<CoefficientPrecision> buffer_in(m_use_permutation ? local_size : 0);
    std::vector<CoefficientPrecision> buffer_out(m_use_permutation ? local_size_source : 0);

    for (int i = 0; i < mu; i++) {
        // local permutation
        if (m_use_permutation) {
            m_target_partition.local_to_local_partition_numbering(rankWorld, in + i * local_size, buffer_in.data());
            // this->local_target_to_local_cluster(in + i * local_size, buffer_in.data());

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

    internal_matrix_product_transp_local_to_local(in_perm.data(), out_perm.data(), mu, buffer_ptr);

    for (int i = 0; i < mu; i++) {
        if (m_use_permutation) {
            // Tranpose
            for (int j = 0; j < local_size_source; j++) {
                buffer_out[j] = out_perm[i + j * mu];
            }

            // local permutation
            // this->local_cluster_to_local_source(buffer_out.data(), out + i * local_size_source);
            m_source_partition.local_partition_to_local_numbering(rankWorld, buffer_out.data(), out + i * local_size_source);
        } else {
            // Tranpose
            for (int j = 0; j < local_size_source; j++) {
                out[j + i * local_size_source] = out_perm[i + j * mu];
            }
        }
    }
}

// Operations using internal numbering
template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_vector_product_global_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    // int nc         = m_global_source_root_cluster->get_size();
    // int local_size = m_local_cluster_target->get_size();
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int local_size = m_target_partition.get_size_of_partition(rankWorld);
    std::fill_n(out, local_size, CoefficientPrecision(0));
    for (auto &local_operator : m_local_operators) {
        local_operator->add_vector_product_global_to_local(1, in, 1, out);
    }
}
template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_matrix_product_global_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    // int nc         = m_global_source_root_cluster->get_size();
    // int local_size = m_local_cluster_target->get_size();
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int local_size = m_target_partition.get_size_of_partition(rankWorld);
    std::fill_n(out, local_size * mu, CoefficientPrecision(0));
    for (auto &local_operator : m_local_operators) {
        local_operator->add_matrix_product_global_to_local(1, in, 1, out, mu);
    }
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_vector_product_transp_local_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int nc = m_source_partition.get_global_size();

    std::fill(out, out + nc, 0);
    for (auto &local_operator : m_local_operators) {
        local_operator->add_vector_product_transp_local_to_global(1, in, 1, out);
    }

    MPI_Allreduce(MPI_IN_PLACE, out, nc, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, m_comm);
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_matrix_product_transp_local_to_global(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu) const {
    int rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    int nc = m_source_partition.get_global_size();

    std::fill(out, out + nc * mu, 0);
    for (auto &local_operator : m_local_operators) {
        local_operator->add_matrix_product_transp_local_to_global(1, in, 1, out, mu);
    }
    MPI_Allreduce(MPI_IN_PLACE, out, nc * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, m_comm);
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_vector_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? m_source_partition.get_global_size() : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    this->local_to_global_source(in, buffer_ptr, 1);
    this->internal_vector_product_global_to_local(buffer_ptr, out);
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_matrix_product_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? m_source_partition.get_global_size() * mu : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    this->local_to_global_source(in, buffer_ptr, mu);
    this->internal_matrix_product_global_to_local(buffer_ptr, out, mu);
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_vector_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, CoefficientPrecision *work) const {
    if (m_symmetry == 'S' || m_symmetry == 'H') {
        this->internal_vector_product_local_to_local(in, out, work);
        return;
    }

    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);

    int local_size_source = m_source_partition.get_size_of_partition(rankWorld);
    int nc                = m_source_partition.get_global_size();

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? nc + local_size_source * sizeWorld : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    std::fill(out, out + local_size_source, 0);
    std::fill(buffer_ptr, buffer_ptr + nc, 0);
    CoefficientPrecision *rbuf = buffer_ptr + nc;

    for (auto &local_operator : m_local_operators) {
        local_operator->add_vector_product_transp_local_to_global(1, in, 1, buffer_ptr);
    }

    std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
    std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        // scounts[i] = (m_global_source_root_cluster->get_clusters_on_partition()[i]->get_size());
        scounts[i] = m_source_partition.get_size_of_partition(i);
        rcounts[i] = (local_size_source);
        if (i > 0) {
            sdispls[i] = sdispls[i - 1] + scounts[i - 1];
            rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        }
    }

    MPI_Alltoallv(buffer_ptr, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);

    for (int i = 0; i < sizeWorld; i++)
        std::transform(out, out + local_size_source, rbuf + rdispls[i], out, std::plus<CoefficientPrecision>());
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_matrix_product_transp_local_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, CoefficientPrecision *work) const {
    if (m_symmetry == 'S' || m_symmetry == 'H') {
        this->internal_matrix_product_local_to_local(in, out, mu, work);
        return;
    }

    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    int local_size_source = m_source_partition.get_size_of_partition(rankWorld);
    int nc                = m_source_partition.get_global_size();

    std::vector<CoefficientPrecision> buffer((work == nullptr) ? (nc + local_size_source * sizeWorld) * mu : 0);
    CoefficientPrecision *buffer_ptr = (work == nullptr) ? buffer.data() : work;

    std::fill(buffer_ptr, buffer_ptr + nc * mu, 0);
    CoefficientPrecision *rbuf = buffer_ptr + nc * mu;

    for (auto &local_operator : m_local_operators) {
        local_operator->add_matrix_product_transp_local_to_global(1, in, 1, buffer_ptr, mu);
    }

    std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
    std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        // scounts[i] = (m_global_source_root_cluster->get_clusters_on_partition()[i]->get_size()) * mu;
        scounts[i] = m_source_partition.get_size_of_partition(i) * mu;
        rcounts[i] = (local_size_source)*mu;
        if (i > 0) {
            sdispls[i] = sdispls[i - 1] + scounts[i - 1];
            rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        }
    }

    MPI_Alltoallv(buffer_ptr, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);

    for (int i = 0; i < sizeWorld; i++)
        std::transform(out, out + local_size_source * mu, rbuf + rdispls[i], out, std::plus<CoefficientPrecision>());
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::internal_sub_matrix_product_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) const {
    for (auto &local_operator : m_local_operators) {
        local_operator->sub_matrix_product_to_local(in, out, mu, offset, size);
    }
}

// Local to global
template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::local_to_global_target(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
    // Allgather
    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        // recvcounts[i] = (m_global_target_root_cluster->partition()[i].second) * mu;
        recvcounts[i] = (m_target_partition.get_size_of_partition(i)) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);
}

template <typename CoefficientPrecision>
void DistributedOperator<CoefficientPrecision>::local_to_global_source(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
    // Allgather
    int sizeWorld, rankWorld;
    MPI_Comm_rank(m_comm, &rankWorld);
    MPI_Comm_size(m_comm, &sizeWorld);
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        // recvcounts[i] = (m_global_source_root_cluster->get_clusters_on_partition()[i]->get_size()) * mu;
        recvcounts[i] = (m_source_partition.get_size_of_partition(i)) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), m_comm);
}

} // namespace htool

#endif
