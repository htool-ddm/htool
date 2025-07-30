#include <algorithm>                                                                                                // for copy_n
#include <cmath>                                                                                                    // for sqrt
#include <htool/basic_types/vector.hpp>                                                                             // for norm2
#include <htool/clustering/cluster_node.hpp>                                                                        // for Cluster...
#include <htool/clustering/tree_builder/tree_builder.hpp>                                                           // for Cluster...
#include <htool/distributed_operator/implementations/global_to_local_operators/dense_matrix.hpp>                    // for RestrictedGlobalToLocalDe...
#include <htool/distributed_operator/implementations/global_to_local_operators/hmatrix.hpp>                         // for RestrictedGlobalToLocalHM...
#include <htool/distributed_operator/implementations/global_to_local_operators/restricted_operator.hpp>             // for RestrictedGlobalToLocalOp...
#include <htool/distributed_operator/implementations/local_to_local_operators/hmatrix.hpp>                          // for LocalToLocalHM...
#include <htool/distributed_operator/linalg/add_distributed_operator_matrix_product_global_to_global.hpp>           // for add_dist..
#include <htool/distributed_operator/linalg/add_distributed_operator_matrix_product_local_to_local.hpp>             // for add_dist..
#include <htool/distributed_operator/linalg/add_distributed_operator_matrix_product_row_major_global_to_global.hpp> // for add_dist..
#include <htool/distributed_operator/linalg/add_distributed_operator_matrix_product_row_major_local_to_local.hpp>   // for add_dist..
#include <htool/distributed_operator/linalg/add_distributed_operator_vector_product_global_to_global.hpp>           // for add_dist..
#include <htool/distributed_operator/linalg/add_distributed_operator_vector_product_local_to_local.hpp>             // for add_dist..
#include <htool/distributed_operator/utility.hpp>                                                                   // for Default...
#include <htool/hmatrix/hmatrix.hpp>                                                                                // for HMatrix
#include <htool/hmatrix/tree_builder/tree_builder.hpp>                                                              // for HMatrix...
#include <htool/matrix/linalg/transpose.hpp>                                                                        // for transpo...
#include <htool/matrix/matrix_view.hpp>
#include <htool/misc/misc.hpp>               // for underly...
#include <htool/testing/generator_input.hpp> // for generat...
#include <htool/testing/geometry.hpp>        // for create_...
#include <htool/wrappers/wrapper_mpi.hpp>    // for wrapper...
#include <iostream>                          // for basic_o...
#include <memory>                            // for make_un...
#include <mpi.h>                             // for MPI_COM...
#include <stdlib.h>                          // for srand
#include <vector>                            // for vector
namespace htool {
template <typename CoefficientPrecision>
class DistributedOperator;
}
namespace htool {
template <typename T>
class Matrix;
}

using namespace std;
using namespace htool;

enum class DataType { Matrix,
                      HMatrix,
                      DefaultHMatrix };

template <typename T, typename GeneratorTestType>
int test_vector_product(GeneratorTestType generator, const DistributedOperator<T> &distributed_operator, const Cluster<htool::underlying_type<T>> &target_root_cluster, const std::vector<int> &MasterOffset_target, const Cluster<htool::underlying_type<T>> &source_root_cluster, const std::vector<int> &MasterOffset_source, int mu, char op, bool use_buffer, htool::underlying_type<T> epsilon) {

    int is_error = 0;
    int nr       = target_root_cluster.get_size();
    int nc       = source_root_cluster.get_size();
    // Get the rankWorld of the process
    int rankWorld, sizeWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    int local_nr = target_root_cluster.get_cluster_on_partition(rankWorld).get_size();
    int local_nc = source_root_cluster.get_cluster_on_partition(rankWorld).get_size();

    int ni = (op == 'T' || op == 'C') ? nr : nc;
    int no = (op == 'T' || op == 'C') ? nc : nr;

    auto &input_cluster  = (op == 'T' || op == 'C') ? target_root_cluster : source_root_cluster;
    auto &output_cluster = (op == 'T' || op == 'C') ? source_root_cluster : target_root_cluster;

    std::vector<int> MasterOffset_input  = (op == 'T' || op == 'C') ? MasterOffset_target : MasterOffset_source;
    std::vector<int> MasterOffset_output = (op == 'T' || op == 'C') ? MasterOffset_source : MasterOffset_target;

    std::vector<T> buffer;
    T *buffer_ptr = nullptr;

    // Random input vector
    Matrix<T> B(ni, mu), C(no, mu), Y(C);
    T alpha, beta;
    if (rankWorld == 0) {
        generate_random_array(B.data(), B.nb_cols() * B.nb_rows());
        generate_random_array(Y.data(), Y.nb_cols() * Y.nb_rows());
        generate_random_scalar(alpha);
        generate_random_scalar(beta);
    }
    int T_size = 1;
    MPI_Bcast(B.data(), B.nb_rows() * B.nb_cols(), wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(Y.data(), Y.nb_rows() * Y.nb_cols(), wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, T_size, wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, T_size, wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);

    Matrix<T> Yt(mu, no), Bt(mu, ni);
    transpose(B, Bt);
    transpose(Y, Yt);

    // reference in user numbering
    Matrix<T> A_dense(nr, nc), matrix_result_w_matrix_sum(Y), transposed_matrix_result_w_matrix_sum(mu, no), matrix_result_wo_matrix_sum(Y), transposed_matrix_result_wo_matrix_sum(mu, no);
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nc; j++) {
            A_dense(i, j) = generator(i, j);
        }
    }
    add_matrix_matrix_product(op, 'N', alpha, A_dense, B, beta, matrix_result_w_matrix_sum);
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);

    add_matrix_matrix_product(op, 'N', alpha, A_dense, B, T(0), matrix_result_wo_matrix_sum);
    transpose(matrix_result_wo_matrix_sum, transposed_matrix_result_wo_matrix_sum);

    Matrix<T> Y_perm(Y), matrix_result_w_matrix_sum_perm(matrix_result_w_matrix_sum), B_perm(B), matrix_result_wo_matrix_sum_perm(matrix_result_wo_matrix_sum);
    for (int p = 0; p < mu; p++) {
        global_to_root_cluster(output_cluster, Y.data() + no * p, Y_perm.data() + no * p);
        global_to_root_cluster(output_cluster, matrix_result_w_matrix_sum.data() + no * p, matrix_result_w_matrix_sum_perm.data() + no * p);
        global_to_root_cluster(input_cluster, B.data() + ni * p, B_perm.data() + ni * p);
        global_to_root_cluster(output_cluster, matrix_result_wo_matrix_sum.data() + no * p, matrix_result_wo_matrix_sum_perm.data() + no * p);
    }

    Matrix<T> Yt_perm(Yt), Bt_perm(Bt), transposed_matrix_result_w_matrix_sum_perm(transposed_matrix_result_w_matrix_sum), transposed_matrix_result_wo_matrix_sum_perm(transposed_matrix_result_wo_matrix_sum);
    transpose(Y_perm, Yt_perm);
    transpose(B_perm, Bt_perm);
    transpose(matrix_result_w_matrix_sum_perm, transposed_matrix_result_w_matrix_sum_perm);
    transpose(matrix_result_wo_matrix_sum_perm, transposed_matrix_result_wo_matrix_sum_perm);

    // Global product
    double global_error, norm_ref(normFrob(matrix_result_w_matrix_sum));
    if (mu == 1) {
        C = Y;
        if (use_buffer) {
            if (op == 'N') {
                buffer.resize(nr + nc + local_nr);
            } else {
                buffer.resize(nr + nc + nc);
            }
            buffer_ptr = buffer.data();
        }
        add_distributed_operator_vector_product_global_to_global(op, alpha, distributed_operator, B.data(), beta, C.data(), buffer_ptr);
        global_error = normFrob(matrix_result_w_matrix_sum - C) / norm_ref;
        is_error     = is_error || !(global_error < epsilon);
        cout << "> Errors on a global to global distributed operator vector product with sum: " << global_error << endl;

        if (use_buffer) {
            if (op != 'N') {
                buffer.resize(nr + nc);
            }
            buffer_ptr = buffer.data();
        }
        C = Y;
        add_distributed_operator_vector_product_global_to_global(op, alpha, distributed_operator, B.data(), T(0), C.data(), buffer_ptr);
        global_error = normFrob(matrix_result_wo_matrix_sum - C) / norm_ref;
        is_error     = is_error || !(global_error < epsilon);
        cout << "> Errors on a global to global distributed operator vector product without sum: " << global_error << endl;

        C = Y_perm;
        if (use_buffer) {
            if (op == 'N') {
                buffer.resize(local_nr);
            } else {
                buffer.resize(nc);
            }
            buffer_ptr = buffer.data();
        }
        internal_add_distributed_operator_vector_product_global_to_global(op, alpha, distributed_operator, B_perm.data(), beta, C.data(), buffer_ptr);
        global_error = normFrob(matrix_result_w_matrix_sum_perm - C) / norm_ref;
        is_error     = is_error || !(global_error < epsilon);
        cout << "> Errors on a global to global internal distributed operator vector product with sum: " << global_error << endl;

        C = Y_perm;
        if (use_buffer) {
            if (op != 'N') {
                buffer.resize(0);
            }
            buffer_ptr = buffer.data();
        }
        internal_add_distributed_operator_vector_product_global_to_global(op, alpha, distributed_operator, B_perm.data(), T(0), C.data(), buffer_ptr);
        global_error = normFrob(matrix_result_wo_matrix_sum_perm - C) / norm_ref;
        is_error     = is_error || !(global_error < epsilon);
        cout << "> Errors on a global to global internal distributed operator vector product without sum: " << global_error << endl;
    }

    C = Y;
    if (use_buffer) {
        if (op == 'N') {
            buffer.resize((nr + nc) * (mu + 1) + local_nr * mu);
        } else {
            buffer.resize(nr + local_nr * mu + nc * (2 * mu + 1));
        }
        buffer_ptr = buffer.data();
    }
    add_distributed_operator_matrix_product_global_to_global(op, alpha, distributed_operator, B, beta, C, buffer_ptr);
    global_error = normFrob(matrix_result_w_matrix_sum - C) / norm_ref;
    is_error     = is_error || !(global_error < epsilon);
    cout << "> Errors on a global to global distributed operator matrix product with sum: " << global_error << endl;

    if (use_buffer) {
        if (op == 'N') {
            buffer.resize(nc * (mu + 1) + nr * mu + local_nr * mu);
        } else {
            buffer.resize(nr + local_nr * mu + 2 * nc * mu);
        }
        buffer_ptr = buffer.data();
    }
    add_distributed_operator_matrix_product_global_to_global(op, alpha, distributed_operator, B, T(0), C, buffer_ptr);
    global_error = normFrob(matrix_result_wo_matrix_sum - C) / norm_ref;
    is_error     = is_error || !(global_error < epsilon);
    cout << "> Errors on a global to global distributed operator matrix product without sum: " << global_error << endl;

    C = Yt_perm;
    if (use_buffer) {
        if (op == 'N') {
            buffer.resize(local_nr * mu);
        } else {
            buffer.resize(nc * mu);
        }
        buffer_ptr = buffer.data();
    }
    internal_add_distributed_operator_matrix_product_row_major_global_to_global<T>(op, alpha, distributed_operator, Bt_perm, beta, C, buffer_ptr);
    global_error = normFrob(transposed_matrix_result_w_matrix_sum_perm - C) / norm_ref;
    is_error     = is_error || !(global_error < epsilon);
    cout << "> Errors on a global to global internal distributed operator matrix product row major with sum: " << global_error << endl;

    if (use_buffer) {
        if (op != 'N') {
            buffer.resize(0);
        }
        buffer_ptr = buffer.data();
    }
    internal_add_distributed_operator_matrix_product_row_major_global_to_global(op, alpha, distributed_operator, Bt_perm, T(0), C, buffer_ptr);
    global_error = normFrob(transposed_matrix_result_wo_matrix_sum_perm - C) / norm_ref;
    is_error     = is_error || !(global_error < epsilon);
    cout << "> Errors on a global to global internal distributed operator matrix product row major without sum: " << global_error << endl;

    C = Y_perm;
    if (use_buffer) {
        if (op == 'N') {
            buffer.resize(local_nr * mu + nr * mu + nc * mu);
        } else {
            buffer.resize(nc * mu + nr * mu + nc * mu);
        }
        buffer_ptr = buffer.data();
    }
    internal_add_distributed_operator_matrix_product_global_to_global(op, alpha, distributed_operator, B_perm, beta, C, buffer_ptr);
    global_error = normFrob(matrix_result_w_matrix_sum_perm - C) / norm_ref;
    is_error     = is_error || !(global_error < epsilon);
    cout << "> Errors on a global to global internal distributed operator matrix product with sum: " << global_error << endl;

    if (use_buffer) {
        if (op != 'N') {
            buffer.resize(nr * mu + nc * mu);
        }
        buffer_ptr = buffer.data();
    }
    internal_add_distributed_operator_matrix_product_global_to_global(op, alpha, distributed_operator, B_perm, T(0), C, buffer_ptr);
    global_error = normFrob(matrix_result_wo_matrix_sum_perm - C) / norm_ref;
    is_error     = is_error || !(global_error < epsilon);
    cout << "> Errors on a global to global internal distributed operator matrix product without sum: " << global_error << endl;

    // Local vectors
    Matrix<T> B_local(MasterOffset_input[2 * rankWorld + 1], mu), Y_local(MasterOffset_output[2 * rankWorld + 1], mu), B_local_perm(MasterOffset_input[2 * rankWorld + 1], mu), Y_local_perm(MasterOffset_output[2 * rankWorld + 1], mu), Yt_local_perm(mu, MasterOffset_output[2 * rankWorld + 1]), Bt_local_perm(mu, MasterOffset_input[2 * rankWorld + 1]);
    for (int i = 0; i < mu; i++) {
        std::copy_n(B.data() + MasterOffset_input[2 * rankWorld] + ni * i,
                    MasterOffset_input[2 * rankWorld + 1],
                    B_local.data() + MasterOffset_input[2 * rankWorld + 1] * i);

        std::copy_n(Y.data() + MasterOffset_output[2 * rankWorld] + no * i,
                    MasterOffset_output[2 * rankWorld + 1],
                    Y_local.data() + MasterOffset_output[2 * rankWorld + 1] * i);
        local_to_local_cluster(input_cluster, rankWorld, B_local.data() + B_local.nb_rows() * i, B_local_perm.data() + B_local.nb_rows() * i);
        local_to_local_cluster(output_cluster, rankWorld, Y_local.data() + Y_local.nb_rows() * i, Y_local_perm.data() + Y_local.nb_rows() * i);
    }
    transpose(Y_local_perm, Yt_local_perm);
    transpose(B_local_perm, Bt_local_perm);

    Matrix<T> ref_local_w_sum(MasterOffset_output[2 * rankWorld + 1], mu), ref_local_wo_sum(MasterOffset_output[2 * rankWorld + 1], mu), ref_local_perm_w_sum(ref_local_w_sum), ref_local_perm_wo_sum(ref_local_wo_sum), ref_t_local_perm_w_sum(mu, MasterOffset_output[2 * rankWorld + 1]), ref_t_local_perm_wo_sum(mu, MasterOffset_output[2 * rankWorld + 1]);
    for (int j = 0; j < mu; j++) {
        for (int i = 0; i < MasterOffset_output[2 * rankWorld + 1]; i++) {
            ref_local_w_sum(i, j)  = matrix_result_w_matrix_sum(i + MasterOffset_output[2 * rankWorld], j);
            ref_local_wo_sum(i, j) = matrix_result_wo_matrix_sum(i + MasterOffset_output[2 * rankWorld], j);
        }
        local_to_local_cluster(output_cluster, rankWorld, ref_local_w_sum.data() + ref_local_w_sum.nb_rows() * j, ref_local_perm_w_sum.data() + ref_local_w_sum.nb_rows() * j);
        local_to_local_cluster(output_cluster, rankWorld, ref_local_wo_sum.data() + ref_local_wo_sum.nb_rows() * j, ref_local_perm_wo_sum.data() + ref_local_wo_sum.nb_rows() * j);
    }
    transpose(ref_local_perm_w_sum, ref_t_local_perm_w_sum);
    transpose(ref_local_perm_wo_sum, ref_t_local_perm_wo_sum);

    Matrix<T> C_local;

    // Local product
    double local_error, local_norm_ref(normFrob(ref_local_w_sum));
    if (mu == 1) {
        if (use_buffer) {
            if (op == 'N') {
                buffer.resize(local_nr + local_nc + nc);
            } else {
                buffer.resize(local_nr + local_nc + nc + local_nc * sizeWorld);
            }
            buffer_ptr = buffer.data();
        }
        C_local = Y_local;
        add_distributed_operator_vector_product_local_to_local(op, alpha, distributed_operator, B_local.data(), beta, C_local.data(), buffer_ptr);
        local_error = normFrob(ref_local_w_sum - C_local) / norm_ref;
        is_error    = is_error || !(local_error < epsilon);
        cout << "> Errors on a local to local distributed operator vector product with sum: " << local_error << endl;

        add_distributed_operator_vector_product_local_to_local(op, alpha, distributed_operator, B_local.data(), T(0), C_local.data(), buffer_ptr);
        local_error = normFrob(ref_local_wo_sum - C_local) / norm_ref;
        is_error    = is_error || !(local_error < epsilon);
        cout << "> Errors on a local to local distributed operator vector product without sum: " << local_error << endl;

        if (use_buffer) {
            if (op == 'N') {
                buffer.resize(nc);
            } else {
                buffer.resize(nc + local_nc * sizeWorld);
            }
            buffer_ptr = buffer.data();
        }
        C_local = Y_local_perm;
        internal_add_distributed_operator_vector_product_local_to_local(op, alpha, distributed_operator, B_local_perm.data(), beta, C_local.data(), buffer_ptr);
        local_error = normFrob(ref_t_local_perm_w_sum - C_local) / local_norm_ref;
        is_error    = is_error || !(local_error < epsilon);
        cout << "> Errors on a local to local internal distributed operator vector product with sum: " << local_error << endl;

        internal_add_distributed_operator_vector_product_local_to_local(op, alpha, distributed_operator, B_local_perm.data(), T(0), C_local.data(), buffer_ptr);
        local_error = normFrob(ref_t_local_perm_wo_sum - C_local) / local_norm_ref;
        is_error    = is_error || !(local_error < epsilon);
        cout << "> Errors on a local to local internal distributed operator vector product without sum: " << local_error << endl;
    }

    if (use_buffer) {
        if (op == 'N') {
            buffer.resize(nc * mu);
        } else {
            buffer.resize((nc + local_nc * sizeWorld) * mu);
        }
        buffer_ptr = buffer.data();
    }
    C_local = Yt_local_perm;
    internal_add_distributed_operator_matrix_product_row_major_local_to_local(op, alpha, distributed_operator, Bt_local_perm, beta, C_local, buffer_ptr);
    local_error = normFrob(ref_t_local_perm_w_sum - C_local) / local_norm_ref;
    is_error    = is_error || !(local_error < epsilon);
    cout << "> Errors on a local to local internal distributed operator matrix product row major with sum: " << local_error << endl;

    C_local = Yt_local_perm;
    internal_add_distributed_operator_matrix_product_row_major_local_to_local(op, alpha, distributed_operator, Bt_local_perm, T(0), C_local, buffer_ptr);
    local_error = normFrob(ref_t_local_perm_wo_sum - C_local) / local_norm_ref;
    is_error    = is_error || !(local_error < epsilon);
    cout << "> Errors on a local to local internal distributed operator matrix product row major without sum: " << local_error << endl;

    if (use_buffer) {
        if (op == 'N') {
            buffer.resize((local_nr + local_nc + nc) * mu);
        } else {
            buffer.resize((local_nr + local_nc + nc + local_nc * sizeWorld) * mu);
        }
        buffer_ptr = buffer.data();
    }

    C_local = Y_local_perm;
    internal_add_distributed_operator_matrix_product_local_to_local(op, alpha, distributed_operator, B_local_perm, beta, C_local, buffer_ptr);
    local_error = normFrob(ref_local_perm_w_sum - C_local) / local_norm_ref;
    is_error    = is_error || !(local_error < epsilon);
    cout << "> Errors on a local to local internal distributed operator matrix product with sum: " << local_error << endl;

    C_local = Y_local_perm;
    internal_add_distributed_operator_matrix_product_local_to_local(op, alpha, distributed_operator, B_local_perm, T(0), C_local, buffer_ptr);
    local_error = normFrob(ref_local_perm_wo_sum - C_local) / local_norm_ref;
    is_error    = is_error || !(local_error < epsilon);
    cout << "> Errors on a local to local internal distributed operator matrix product without sum: " << local_error << endl;

    if (use_buffer) {
        if (op == 'N') {
            buffer.resize((local_nr + local_nc) * (mu + 1) + nc * mu);
        } else {
            buffer.resize((local_nr + local_nc) * (mu + 1) + (nc + local_nc * sizeWorld) * mu);
        }
        buffer_ptr = buffer.data();
    }
    C_local = Y_local;
    add_distributed_operator_matrix_product_local_to_local(op, alpha, distributed_operator, B_local.data(), beta, C_local.data(), mu, buffer_ptr);
    local_error = normFrob(ref_local_w_sum - C_local) / local_norm_ref;
    is_error    = is_error || !(local_error < epsilon);
    cout << "> Errors on a local to local distributed operator matrix product with sum: " << local_error << endl;

    C_local = Y_local;
    add_distributed_operator_matrix_product_local_to_local(op, alpha, distributed_operator, B_local.data(), T(0), C_local.data(), mu, buffer_ptr);
    local_error = normFrob(ref_local_wo_sum - C_local) / local_norm_ref;
    is_error    = is_error || !(local_error < epsilon);
    cout << "> Errors on a local to local distributed operator matrix product without sum: " << local_error << endl;

    return is_error;
}

template <typename T, typename GeneratorTestType>
auto add_off_diagonal_operator(ClusterTreeBuilder<htool::underlying_type<T>> &recursive_build, DistributedOperator<T> &distributed_operator, const Cluster<htool::underlying_type<T>> &target_root_cluster, const std::vector<double> p1, const Cluster<htool::underlying_type<T>> &source_root_cluster, const std::vector<double> p2, DataType data_type, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {

    struct Holder {
        std::unique_ptr<const Cluster<htool::underlying_type<T>>> off_diagonal_cluster;
        std::unique_ptr<GeneratorTestType> generator_off_diagonal;
        std::unique_ptr<Matrix<T>> off_diagonal_matrix_1, off_diagonal_matrix_2;
        std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> off_diagonal_hmatrix_1, off_diagonal_hmatrix_2;
        std::unique_ptr<VirtualGlobalToLocalOperator<T>> local_off_diagonal_operator_1, local_off_diagonal_operator_2;

        Holder(ClusterTreeBuilder<htool::underlying_type<T>> &recursive_build, const Cluster<htool::underlying_type<T>> &target_root_cluster, const std::vector<double> &p1, const Cluster<htool::underlying_type<T>> &source_root_cluster, const std::vector<double> &p2, DataType data_type, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {
            // Local clusters
            int rankWorld;
            MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
            const Cluster<htool::underlying_type<T>> &local_target_root_cluster = target_root_cluster.get_cluster_on_partition(rankWorld);
            const Cluster<htool::underlying_type<T>> &local_source_root_cluster = source_root_cluster.get_cluster_on_partition(rankWorld);

            // Sizes
            int nc = source_root_cluster.get_size();
            // int nr       = target_root_cluster.get_size();
            int nc_local = local_source_root_cluster.get_size();
            int off_diagonal_nc_1{local_source_root_cluster.get_offset()};
            int off_diagonal_nc_2{source_root_cluster.get_size() - local_source_root_cluster.get_size() - local_source_root_cluster.get_offset()};

            // Local off diagonal cluster
            std::vector<double> p2_permuted;
            p2_permuted.resize(p2.size());
            const auto &source_permutation = source_root_cluster.get_permutation();
            for (int i = 0; i < source_permutation.size(); i++) {
                p2_permuted[i * 3 + 0] = p2[source_permutation[i] * 3 + 0];
                p2_permuted[i * 3 + 1] = p2[source_permutation[i] * 3 + 1];
                p2_permuted[i * 3 + 2] = p2[source_permutation[i] * 3 + 2];
            }
            std::vector<int> off_diagonal_partition;
            off_diagonal_partition.push_back(0);
            off_diagonal_partition.push_back(off_diagonal_nc_1);
            // off_diagonal_partition.push_back(off_diagonal_nc_1);
            // off_diagonal_partition.push_back(nr);
            off_diagonal_partition.push_back(off_diagonal_nc_1 + nc_local);
            off_diagonal_partition.push_back(off_diagonal_nc_2);

            off_diagonal_cluster   = make_unique<const Cluster<htool::underlying_type<T>>>(recursive_build.create_cluster_tree(nc, 3, p2_permuted.data(), 2, 2, off_diagonal_partition.data()));
            generator_off_diagonal = std::make_unique<GeneratorTestType>(3, p1, p2_permuted);

            // // recursive_build.set_partition(2, off_diagonal_partition.data());
            // off_diagonal_cluster = make_unique<const Cluster<htool::underlying_type<T>>>(recursive_build.create_cluster_tree(nc, 3, p2_permuted.data(), 2, 2, off_diagonal_partition.data()));

            // // Generators
            // if (use_permutation) {
            //     generator_off_diagonal = std::make_unique<GeneratorTestType>(3, nr, nc, p1, p2_permuted, local_target_root_cluster, *off_diagonal_cluster, true, true);
            // } else {
            //     generator_off_diagonal = std::make_unique<GeneratorTestType>(3, nr, nc, p1_permuted, p2_permuted, local_target_root_cluster, *off_diagonal_cluster, true, true);
            //     generator_off_diagonal->set_use_target_permutation(false);
            //     generator_off_diagonal->set_use_source_permutation(true);
            // }

            // Off diagonal LocalDenseMatrix
            const Cluster<htool::underlying_type<T>> *local_off_diagonal_cluster_tree_1 = &off_diagonal_cluster->get_cluster_on_partition(0);
            const Cluster<htool::underlying_type<T>> *local_off_diagonal_cluster_tree_2 = &off_diagonal_cluster->get_cluster_on_partition(1);

            if (data_type == DataType::Matrix) {
                GeneratorTestType generator(3, p1, p2);

                off_diagonal_matrix_1 = std::make_unique<Matrix<T>>(local_target_root_cluster.get_size(), local_source_root_cluster.get_offset());
                off_diagonal_matrix_2 = std::make_unique<Matrix<T>>(local_target_root_cluster.get_size(), source_root_cluster.get_size() - local_source_root_cluster.get_size() - local_source_root_cluster.get_offset());

                generator.copy_submatrix(off_diagonal_matrix_1->nb_rows(), off_diagonal_matrix_1->nb_cols(), local_target_root_cluster.get_permutation().data() + local_target_root_cluster.get_offset(), source_root_cluster.get_permutation().data(), off_diagonal_matrix_1->data());

                generator.copy_submatrix(off_diagonal_matrix_2->nb_rows(), off_diagonal_matrix_2->nb_cols(), local_target_root_cluster.get_permutation().data() + local_target_root_cluster.get_offset(), source_root_cluster.get_permutation().data() + local_source_root_cluster.get_offset() + local_source_root_cluster.get_size(), off_diagonal_matrix_2->data());

                if (local_source_root_cluster.get_offset() > 0)
                    local_off_diagonal_operator_1 = make_unique<RestrictedGlobalToLocalDenseMatrix<T>>(*off_diagonal_matrix_1, local_target_root_cluster, LocalRenumbering(0, local_source_root_cluster.get_offset(), source_root_cluster.get_permutation().size(), source_root_cluster.get_permutation().data()), 'N', 'N', false, false);
                if (source_root_cluster.get_size() - local_source_root_cluster.get_size() - local_source_root_cluster.get_offset() > 0)
                    local_off_diagonal_operator_2 = make_unique<RestrictedGlobalToLocalDenseMatrix<T>>(*off_diagonal_matrix_2, local_target_root_cluster, LocalRenumbering(local_source_root_cluster.get_size() + local_source_root_cluster.get_offset(), source_root_cluster.get_size() - local_source_root_cluster.get_size() - local_source_root_cluster.get_offset(), source_root_cluster.get_permutation().size(), source_root_cluster.get_permutation().data()), 'N', 'N', false, false);

            } else if (data_type == DataType::HMatrix || data_type == DataType::DefaultHMatrix) {
                HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_builder_1(epsilon, eta, 'N', 'N');
                HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_builder_2(epsilon, eta, 'N', 'N');

                off_diagonal_hmatrix_1 = make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_builder_1.build(*generator_off_diagonal, local_target_root_cluster, *local_off_diagonal_cluster_tree_1));
                off_diagonal_hmatrix_2 = make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_builder_2.build(*generator_off_diagonal, local_target_root_cluster, *local_off_diagonal_cluster_tree_2));

                local_off_diagonal_operator_1 = make_unique<RestrictedGlobalToLocalHMatrix<T, htool::underlying_type<T>>>(*off_diagonal_hmatrix_1, local_target_root_cluster, *local_off_diagonal_cluster_tree_1, false, true);
                local_off_diagonal_operator_2 = make_unique<RestrictedGlobalToLocalHMatrix<T, htool::underlying_type<T>>>(*off_diagonal_hmatrix_2, local_target_root_cluster, *local_off_diagonal_cluster_tree_2, false, true);
            }
        }
    };

    Holder holder(recursive_build, target_root_cluster, p1, source_root_cluster, p2, data_type, epsilon, eta);
    if (holder.off_diagonal_cluster->get_cluster_on_partition(0).get_size() > 0)
        distributed_operator.add_global_to_local_operator(holder.local_off_diagonal_operator_1.get());
    if (holder.off_diagonal_cluster->get_cluster_on_partition(1).get_size() > 0)
        distributed_operator.add_global_to_local_operator(holder.local_off_diagonal_operator_2.get());
    return holder;
}

template <typename T, typename GeneratorTestType>
bool test_default_distributed_operator(int nr, int nc, int mu, bool use_buffer, char Symmetry, char UPLO, char op, bool off_diagonal_approximation, DataType data_type, htool::underlying_type<T> epsilon = 1e-14) {

    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool test                     = 0;
    htool::underlying_type<T> eta = 10;

    // Geometry
    double z1 = 1;
    vector<double> p1(3 * nr), p1_permuted, off_diagonal_p1;
    vector<double> p2(Symmetry == 'N' ? 3 * nc : 1), p2_permuted, off_diagonal_p2;
    create_disk(3, z1, nr, p1.data());
    int size_numbering = nr / (sizeWorld);
    int count_size     = 0;
    std::vector<int> MasterOffset_target, MasterOffset_source;
    for (int p = 0; p < sizeWorld - 1; p++) {
        MasterOffset_target.push_back(count_size);
        MasterOffset_target.push_back(size_numbering);

        count_size += size_numbering;
    }
    MasterOffset_target.push_back(count_size);
    MasterOffset_target.push_back(nr - count_size);

    size_numbering = nc / sizeWorld;
    count_size     = 0;

    ClusterTreeBuilder<htool::underlying_type<T>> recursive_build;
    std::shared_ptr<const Cluster<htool::underlying_type<T>>> source_root_cluster;
    std::shared_ptr<const Cluster<htool::underlying_type<T>>> target_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(recursive_build.create_cluster_tree(nr, 3, p1.data(), 2, sizeWorld, MasterOffset_target.data()));

    if (Symmetry == 'N') {
        double z2 = 1 + 0.1;
        create_disk(3, z2, nc, p2.data());

        for (int p = 0; p < sizeWorld - 1; p++) {
            MasterOffset_source.push_back(count_size);
            MasterOffset_source.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_source.push_back(count_size);
        MasterOffset_source.push_back(nc - count_size);

        // recursive_build.set_partition(sizeWorld, MasterOffset_source.data());

        source_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(recursive_build.create_cluster_tree(nc, 3, p2.data(), 2, sizeWorld, MasterOffset_source.data()));

    } else {

        MasterOffset_source = MasterOffset_target;
        source_root_cluster = target_root_cluster;
        p2                  = p1;
    }

    // Generator
    GeneratorTestType generator_in_user_numbering(3, p1, p2);
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(epsilon, eta, Symmetry, UPLO);

    if (off_diagonal_approximation) {
        DefaultLocalApproximationBuilder<T, htool::underlying_type<T>> distributed_operator_holder(generator_in_user_numbering, *target_root_cluster, *source_root_cluster, hmatrix_tree_builder, MPI_COMM_WORLD);

        DistributedOperator<T> &distributed_operator = distributed_operator_holder.distributed_operator;
        auto dependencies                            = add_off_diagonal_operator<T, GeneratorTestType>(recursive_build, distributed_operator, *target_root_cluster, p1, *source_root_cluster, p2, data_type, epsilon, eta);
        test                                         = test_vector_product(generator_in_user_numbering, distributed_operator, *target_root_cluster, MasterOffset_target, *source_root_cluster, MasterOffset_source, mu, op, use_buffer, epsilon);
    } else {
        DefaultApproximationBuilder<T, htool::underlying_type<T>> distributed_operator_holder(generator_in_user_numbering, *target_root_cluster, *source_root_cluster, hmatrix_tree_builder, MPI_COMM_WORLD);

        DistributedOperator<T> &distributed_operator = distributed_operator_holder.distributed_operator;
        test                                         = test_vector_product(generator_in_user_numbering, distributed_operator, *target_root_cluster, MasterOffset_target, *source_root_cluster, MasterOffset_source, mu, op, use_buffer, epsilon);
    }

    return test;
}

template <typename T, typename GeneratorTestType>
bool test_distributed_operator(int nr, int nc, int mu, bool use_buffer, char Symmetry, char UPLO, char op, bool off_diagonal_approximation, DataType data_type, htool::underlying_type<T> epsilon) {
    return test_default_distributed_operator<T, GeneratorTestType>(nr, nc, mu, use_buffer, Symmetry, UPLO, op, off_diagonal_approximation, data_type, epsilon);
}
