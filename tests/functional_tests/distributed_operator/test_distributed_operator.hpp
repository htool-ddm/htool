
#include <htool/clustering/clustering.hpp>
#include <htool/distributed_operator/distributed_operator.hpp>
#include <htool/distributed_operator/implementations/partition_from_cluster.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <htool/local_operators/local_dense_matrix.hpp>
#include <htool/local_operators/local_hmatrix.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <random>

using namespace std;
using namespace htool;

enum class DataType { Matrix,
                      HMatrix,
                      DefaultHMatrix };

template <typename T, typename GeneratorTestType>
int test_vector_product(GeneratorTestType generator, const DistributedOperator<T> &distributed_operator, const Cluster<htool::underlying_type<T>> &target_root_cluster, const std::vector<int> &MasterOffset_target, const Cluster<htool::underlying_type<T>> &source_root_cluster, const std::vector<int> &MasterOffset_source, int mu, char op, bool use_permutation, htool::underlying_type<T> epsilon) {

    int test = 0;
    int nr   = target_root_cluster.get_size();
    int nc   = source_root_cluster.get_size();
    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    int ni = (op == 'T' || op == 'C') ? nr : nc;
    int no = (op == 'T' || op == 'C') ? nc : nr;

    std::vector<int> MasterOffset_input  = (op == 'T' || op == 'C') ? MasterOffset_target : MasterOffset_source;
    std::vector<int> MasterOffset_output = (op == 'T' || op == 'C') ? MasterOffset_source : MasterOffset_target;

    // Random input vector
    std::vector<T> x_vec(ni * mu, 1), y_vec(no * mu, 1), ref(no * mu, 0), out(no * mu, 0);
    T alpha, beta;
    if (rankWorld == 0) {
        generate_random_vector(x_vec);
        // std::iota(x_vec.begin(), x_vec.end(), T(0));
        generate_random_vector(y_vec);
        generate_random_scalar(alpha);
        generate_random_scalar(beta);
    }
    alpha      = 1;
    beta       = 0;
    int T_size = 1;
    MPI_Bcast(x_vec.data(), x_vec.size(), wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(y_vec.data(), y_vec.size(), wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, T_size, wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, T_size, wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);

    // reference in user numbering
    if (op == 'N') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    ref.at(i + p * no) += alpha * generator(i, j) * x_vec.at(j + p * ni);
                }
                ref.at(i + p * no) += beta * y_vec.at(i + p * no);
            }
        }
    } else if (op == 'T') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    ref.at(i + p * no) += alpha * generator(j, i) * x_vec.at(j + p * ni);
                }
                ref.at(i + p * no) += beta * y_vec.at(i + p * no);
            }
        }
    } else if (op == 'C') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    ref.at(i + p * no) += alpha * conj_if_complex(generator(j, i)) * x_vec.at(j + p * ni);
                }
                ref.at(i + p * no) += beta * y_vec.at(i + p * no);
            }
        }
    }

    // Permutation
    if (!use_permutation) {
        std::vector<T> temp_1(x_vec), temp_2(y_vec);
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                global_to_root_cluster(target_root_cluster, x_vec.data() + ni * j, temp_1.data() + ni * j);
                global_to_root_cluster(source_root_cluster, y_vec.data() + no * j, temp_2.data() + no * j);
            } else {
                global_to_root_cluster(source_root_cluster, x_vec.data() + ni * j, temp_1.data() + ni * j);
                global_to_root_cluster(target_root_cluster, y_vec.data() + no * j, temp_2.data() + no * j);
            }
        }
        x_vec = temp_1;
        y_vec = temp_2;
    }

    // if (rankWorld == 0) {
    //     std::cout << x_vec << "\n";
    // }

    // Global product
    if (op == 'T') {
        if (mu == 1) {
            distributed_operator.vector_product_transp_global_to_global(x_vec.data(), y_vec.data());
        }
        distributed_operator.matrix_product_transp_global_to_global(x_vec.data(), y_vec.data(), mu);
    } else if (op == 'C') {
    } else {
        if (mu == 1) {
            distributed_operator.vector_product_global_to_global(x_vec.data(), y_vec.data());
        }
        distributed_operator.matrix_product_global_to_global(x_vec.data(), y_vec.data(), mu);
    }
    // if (rankWorld == 0) {
    //     std::cout << ref << "\n";
    //     std::cout << y_vec << "\n";
    // }
    // Error on global product
    double global_error, norm_ref(norm2(ref));
    if (use_permutation) {
        global_error = norm2(y_vec - ref) / norm_ref;
    } else {
        std::vector<T> temp(y_vec);
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                root_cluster_to_global(source_root_cluster, y_vec.data() + no * j, temp.data() + no * j);
            } else {
                root_cluster_to_global(target_root_cluster, y_vec.data() + no * j, temp.data() + no * j);
            }
        }
        global_error = norm2(temp - ref) / norm_ref;
    }
    if (rankWorld == 0)
        cout << "error with global product: " << global_error << endl;
    test = test || !(global_error < epsilon);

    // Local vectors
    std::vector<T> x_local(MasterOffset_input[2 * rankWorld + 1] * mu), out_local(MasterOffset_output[2 * rankWorld + 1] * mu), out_local_permuted(MasterOffset_output[2 * rankWorld + 1] * mu);
    for (int i = 0; i < mu; i++) {
        std::copy_n(x_vec.data() + MasterOffset_input[2 * rankWorld] + ni * i, MasterOffset_input[2 * rankWorld + 1], x_local.data() + MasterOffset_input[2 * rankWorld + 1] * i);
    }

    // Local product
    if (op == 'T') {
        if (mu == 1) {
            distributed_operator.vector_product_transp_local_to_local(x_local.data(), out_local.data());
        }
        distributed_operator.matrix_product_transp_local_to_local(x_local.data(), out_local.data(), mu);
    } else if (op == 'C') {
        // HA->mvprod_conj_local_to_local(x_local.data(), out_local.data(), mu);
    } else {
        if (mu == 1) {
            distributed_operator.vector_product_local_to_local(x_local.data(), out_local.data());
        }
        distributed_operator.matrix_product_local_to_local(x_local.data(), out_local.data(), mu);
    }

    // Error
    double global_local_diff = 0;
    if (!use_permutation) {
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                local_cluster_to_local(source_root_cluster, rankWorld, out_local.data() + MasterOffset_output[2 * rankWorld + 1] * j, out_local_permuted.data() + MasterOffset_output[2 * rankWorld + 1] * j);
            } else {
                local_cluster_to_local(target_root_cluster, rankWorld, out_local.data() + MasterOffset_output[2 * rankWorld + 1] * j, out_local_permuted.data() + MasterOffset_output[2 * rankWorld + 1] * j);
            }
        }
    }
    const T *local_output = (use_permutation) ? out_local.data() : out_local_permuted.data();
    // const T *global_output = (use_permutation) ? y_vec.data() : out_global_permuted.data();
    for (int i = 0; i < MasterOffset_output[2 * rankWorld + 1]; i++) {
        for (int j = 0; j < mu; j++) {
            global_local_diff += std::abs(ref[i + MasterOffset_output[2 * rankWorld] + j * no] - local_output[i + j * MasterOffset_output[2 * rankWorld + 1]]) * std::abs(ref[i + MasterOffset_output[2 * rankWorld] + j * no] - local_output[i + j * MasterOffset_output[2 * rankWorld + 1]]);
        }
    }

    double global_local_err = std::sqrt(global_local_diff) / norm2(out_local);

    if (rankWorld == 0) {
        cout << "error with local product: " << global_local_err << endl;
    }
    test = test || !(global_local_err < epsilon);
    return test;
}

template <typename T, typename GeneratorTestType>
auto add_off_diagonal_operator(ClusterTreeBuilder<htool::underlying_type<T>> &recursive_build, DistributedOperator<T> &distributed_operator, const Cluster<htool::underlying_type<T>> &target_root_cluster, const std::vector<double> p1, const std::vector<double> p1_permuted, const Cluster<htool::underlying_type<T>> &source_root_cluster, const std::vector<double> p2_permuted, bool use_permutation, DataType data_type, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {
    struct Holder {
        std::unique_ptr<const Cluster<htool::underlying_type<T>>> off_diagonal_cluster;
        // std::unique_ptr<const Cluster<htool::underlying_type<T>>> local_off_diagonal_cluster_tree_1, local_off_diagonal_cluster_tree_2;
        std::unique_ptr<GeneratorTestType> generator_off_diagonal;
        std::unique_ptr<Matrix<T>> off_diagonal_matrix_1, off_diagonal_matrix_2;
        std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> off_diagonal_hmatrix_1, off_diagonal_hmatrix_2;
        std::unique_ptr<LocalOperator<T, htool::underlying_type<T>>> local_off_diagonal_operator_1, local_off_diagonal_operator_2;

        Holder(ClusterTreeBuilder<htool::underlying_type<T>> &recursive_build, const Cluster<htool::underlying_type<T>> &target_root_cluster, const std::vector<double> p1, const std::vector<double> p1_permuted, const Cluster<htool::underlying_type<T>> &source_root_cluster, const std::vector<double> p2_permuted, bool use_permutation, DataType data_type, htool::underlying_type<T> epsilon, htool::underlying_type<T> eta) {
            // Local clusters
            int rankWorld;
            MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
            const Cluster<htool::underlying_type<T>> &local_target_root_cluster = target_root_cluster.get_cluster_on_partition(rankWorld);
            const Cluster<htool::underlying_type<T>> &local_source_root_cluster = source_root_cluster.get_cluster_on_partition(rankWorld);

            // Sizes
            int nc       = source_root_cluster.get_size();
            int nr       = target_root_cluster.get_size();
            int nc_local = local_source_root_cluster.get_size();
            int off_diagonal_nc_1{local_source_root_cluster.get_offset()};
            int off_diagonal_nc_2{source_root_cluster.get_size() - local_source_root_cluster.get_size() - local_source_root_cluster.get_offset()};

            // Local off diagonal cluster
            std::vector<int> off_diagonal_partition;
            off_diagonal_partition.push_back(0);
            off_diagonal_partition.push_back(off_diagonal_nc_1);
            // off_diagonal_partition.push_back(off_diagonal_nc_1);
            // off_diagonal_partition.push_back(nr);
            off_diagonal_partition.push_back(off_diagonal_nc_1 + nc_local);
            off_diagonal_partition.push_back(off_diagonal_nc_2);

            // recursive_build.set_partition(2, off_diagonal_partition.data());
            off_diagonal_cluster = make_unique<const Cluster<htool::underlying_type<T>>>(recursive_build.create_cluster_tree(nc, 3, p2_permuted.data(), 2, 2, off_diagonal_partition.data()));

            // Generators
            if (use_permutation) {
                generator_off_diagonal = std::make_unique<GeneratorTestType>(3, nr, nc, p1, p2_permuted, local_target_root_cluster, *off_diagonal_cluster, true, true);
            } else {
                generator_off_diagonal = std::make_unique<GeneratorTestType>(3, nr, nc, p1_permuted, p2_permuted, local_target_root_cluster, *off_diagonal_cluster, true, true);
                generator_off_diagonal->set_use_target_permutation(false);
                generator_off_diagonal->set_use_source_permutation(true);
            }

            // Off diagonal LocalDenseMatrix
            const Cluster<htool::underlying_type<T>> *local_off_diagonal_cluster_tree_1 = &off_diagonal_cluster->get_cluster_on_partition(0);
            const Cluster<htool::underlying_type<T>> *local_off_diagonal_cluster_tree_2 = &off_diagonal_cluster->get_cluster_on_partition(1);

            if (data_type == DataType::Matrix) {
                off_diagonal_matrix_1 = std::make_unique<Matrix<T>>(local_target_root_cluster.get_size(), local_off_diagonal_cluster_tree_1->get_size());
                off_diagonal_matrix_2 = std::make_unique<Matrix<T>>(local_target_root_cluster.get_size(), local_off_diagonal_cluster_tree_2->get_size());
                generator_off_diagonal->copy_submatrix(local_target_root_cluster.get_size(), local_off_diagonal_cluster_tree_1->get_size(), local_target_root_cluster.get_offset(), local_off_diagonal_cluster_tree_1->get_offset(), off_diagonal_matrix_1->data());
                generator_off_diagonal->copy_submatrix(local_target_root_cluster.get_size(), local_off_diagonal_cluster_tree_2->get_size(), local_target_root_cluster.get_offset(), local_off_diagonal_cluster_tree_2->get_offset(), off_diagonal_matrix_2->data());
                local_off_diagonal_operator_1 = make_unique<LocalDenseMatrix<T, htool::underlying_type<T>>>(*off_diagonal_matrix_1, local_target_root_cluster, *local_off_diagonal_cluster_tree_1, 'N', 'N', false, true);
                local_off_diagonal_operator_2 = make_unique<LocalDenseMatrix<T, htool::underlying_type<T>>>(*off_diagonal_matrix_2, local_target_root_cluster, *local_off_diagonal_cluster_tree_2, 'N', 'N', false, true);

            } else if (data_type == DataType::HMatrix || data_type == DataType::DefaultHMatrix) {
                HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_builder_1(local_target_root_cluster, *local_off_diagonal_cluster_tree_1, epsilon, eta, 'N', 'N', -1, -1, -1);
                HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_builder_2(local_target_root_cluster, *local_off_diagonal_cluster_tree_2, epsilon, eta, 'N', 'N', -1, -1, -1);

                off_diagonal_hmatrix_1 = make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_builder_1.build(*generator_off_diagonal));
                off_diagonal_hmatrix_2 = make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_builder_2.build(*generator_off_diagonal));

                local_off_diagonal_operator_1 = make_unique<LocalHMatrix<T, htool::underlying_type<T>>>(*off_diagonal_hmatrix_1, local_target_root_cluster, *local_off_diagonal_cluster_tree_1, 'N', 'N', false, true);
                local_off_diagonal_operator_2 = make_unique<LocalHMatrix<T, htool::underlying_type<T>>>(*off_diagonal_hmatrix_2, local_target_root_cluster, *local_off_diagonal_cluster_tree_2, 'N', 'N', false, true);
            }
        }
    };

    Holder holder(recursive_build, target_root_cluster, p1, p1_permuted, source_root_cluster, p2_permuted, use_permutation, data_type, epsilon, eta);
    if (holder.off_diagonal_cluster->get_cluster_on_partition(0).get_size() > 0)
        distributed_operator.add_local_operator(holder.local_off_diagonal_operator_1.get());
    if (holder.off_diagonal_cluster->get_cluster_on_partition(1).get_size() > 0)
        distributed_operator.add_local_operator(holder.local_off_diagonal_operator_2.get());
    return holder;
}

template <typename T, typename GeneratorTestType>
bool test_custom_distributed_operator(int nr, int nc, int mu, bool use_permutation, char Symmetry, char UPLO, char op, bool off_diagonal_approximation, DataType data_type, htool::underlying_type<T> epsilon = 1e-14) {

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

    // Permutation on geometry
    p1_permuted.resize(3 * nr);
    const auto &target_permutation = target_root_cluster->get_permutation();
    for (int i = 0; i < target_permutation.size(); i++) {
        p1_permuted[i * 3 + 0] = p1[target_permutation[i] * 3 + 0];
        p1_permuted[i * 3 + 1] = p1[target_permutation[i] * 3 + 1];
        p1_permuted[i * 3 + 2] = p1[target_permutation[i] * 3 + 2];
    }
    p2_permuted.resize(3 * nc);
    if (Symmetry == 'N') {
        const auto &source_permutation = source_root_cluster->get_permutation();
        for (int i = 0; i < source_permutation.size(); i++) {
            p2_permuted[i * 3 + 0] = p2[source_permutation[i] * 3 + 0];
            p2_permuted[i * 3 + 1] = p2[source_permutation[i] * 3 + 1];
            p2_permuted[i * 3 + 2] = p2[source_permutation[i] * 3 + 2];
        }
    } else {
        p2_permuted = p1_permuted;
    }

    // Generator
    GeneratorTestType generator(3, nr, nc, p1, p2, *target_root_cluster, *source_root_cluster, true, true);
    GeneratorTestType generator_permuted(3, nr, nc, p1_permuted, p2_permuted, *target_root_cluster, *source_root_cluster, false, false);

    // Diagonal LocalDenseMatrix
    std::shared_ptr<LocalOperator<T, htool::underlying_type<T>>> local_operator;
    const Cluster<htool::underlying_type<T>> *actual_target_cluster = &target_root_cluster->get_cluster_on_partition(rankWorld);
    const Cluster<htool::underlying_type<T>> *actual_source_cluster = off_diagonal_approximation ? &source_root_cluster->get_cluster_on_partition(rankWorld) : source_root_cluster.get();

    char symmetry = (sizeWorld == 1 || off_diagonal_approximation) ? Symmetry : 'N';
    char uplo     = (sizeWorld == 1 || off_diagonal_approximation) ? UPLO : 'N';

    std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> hmatrix;
    std::unique_ptr<Matrix<T>> matrix;

    if (data_type == DataType::Matrix) {
        matrix = std::make_unique<Matrix<T>>(actual_target_cluster->get_size(), actual_source_cluster->get_size());
        if (symmetry == 'N') {
            generator_permuted.copy_submatrix(matrix->nb_rows(), matrix->nb_cols(), actual_target_cluster->get_offset(), actual_source_cluster->get_offset(), matrix->data());
        } else if ((symmetry == 'S' || symmetry == 'H') && uplo == 'L') {
            for (int i = 0; i < matrix->nb_rows(); i++) {
                for (int j = 0; j < i + 1; j++) {
                    generator_permuted.copy_submatrix(1, 1, i + actual_target_cluster->get_offset(), j + actual_source_cluster->get_offset(), matrix->data() + i + j * matrix->nb_rows());
                }
            }
        } else if ((symmetry == 'S' || symmetry == 'H') && uplo == 'U') {
            for (int j = 0; j < matrix->nb_cols(); j++) {
                for (int i = 0; i < j + 1; i++) {
                    generator_permuted.copy_submatrix(1, 1, i + actual_target_cluster->get_offset(), j + actual_source_cluster->get_offset(), matrix->data() + i + j * matrix->nb_rows());
                }
            }
        }
        local_operator = std::make_unique<LocalDenseMatrix<T, htool::underlying_type<T>>>(*matrix, *actual_target_cluster, *actual_source_cluster, symmetry, uplo);
    } else if (data_type == DataType::HMatrix) {
        HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_builder(*actual_target_cluster, *actual_source_cluster, epsilon, eta, symmetry, uplo, -1, -1, rankWorld);
        hmatrix        = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_builder.build(generator_permuted));
        local_operator = std::make_unique<LocalHMatrix<T, htool::underlying_type<T>>>(*hmatrix, *actual_target_cluster, *actual_source_cluster, symmetry, uplo);
    }

    // Distributed operator
    PartitionFromCluster<T, htool::underlying_type<T>> target_partition(*target_root_cluster);
    PartitionFromCluster<T, htool::underlying_type<T>> source_partition(*source_root_cluster);
    DistributedOperator<T> distributed_operator(target_partition, source_partition, Symmetry, UPLO, MPI_COMM_WORLD);
    distributed_operator.add_local_operator(local_operator.get());
    distributed_operator.use_permutation() = use_permutation;

    if (off_diagonal_approximation) {
        auto dependencies = add_off_diagonal_operator<T, GeneratorTestType>(recursive_build, distributed_operator, *target_root_cluster, p1, p1_permuted, *source_root_cluster, p2_permuted, use_permutation, data_type, epsilon, eta);

        test = test_vector_product(generator, distributed_operator, *target_root_cluster, MasterOffset_target, *source_root_cluster, MasterOffset_source, mu, op, use_permutation, epsilon);
    } else {
        test = test_vector_product(generator, distributed_operator, *target_root_cluster, MasterOffset_target, *source_root_cluster, MasterOffset_source, mu, op, use_permutation, epsilon);
    }

    return test;
}

template <typename T, typename GeneratorTestType>
bool test_default_distributed_operator(int nr, int nc, int mu, bool use_permutation, char Symmetry, char UPLO, char op, bool off_diagonal_approximation, DataType data_type, htool::underlying_type<T> epsilon = 1e-14) {

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

    // Permutation on geometry
    p1_permuted.resize(3 * nr);
    const auto &target_permutation = target_root_cluster->get_permutation();
    for (int i = 0; i < target_permutation.size(); i++) {
        p1_permuted[i * 3 + 0] = p1[target_permutation[i] * 3 + 0];
        p1_permuted[i * 3 + 1] = p1[target_permutation[i] * 3 + 1];
        p1_permuted[i * 3 + 2] = p1[target_permutation[i] * 3 + 2];
    }
    p2_permuted.resize(3 * nc);
    if (Symmetry == 'N') {
        const auto &source_permutation = source_root_cluster->get_permutation();
        for (int i = 0; i < source_permutation.size(); i++) {
            p2_permuted[i * 3 + 0] = p2[source_permutation[i] * 3 + 0];
            p2_permuted[i * 3 + 1] = p2[source_permutation[i] * 3 + 1];
            p2_permuted[i * 3 + 2] = p2[source_permutation[i] * 3 + 2];
        }
    } else {
        p2_permuted = p1_permuted;
    }

    // Generator
    GeneratorTestType generator(3, nr, nc, p1, p2, *target_root_cluster, *source_root_cluster, true, true);
    GeneratorTestType generator_permuted(3, nr, nc, p1_permuted, p2_permuted, *target_root_cluster, *source_root_cluster, false, false);

    if (off_diagonal_approximation) {
        DefaultLocalApproximationBuilder<T, htool::underlying_type<T>> distributed_operator_holder(generator_permuted, *target_root_cluster, *source_root_cluster, epsilon, eta, Symmetry, UPLO, MPI_COMM_WORLD);

        DistributedOperator<T> &distributed_operator = distributed_operator_holder.distributed_operator;
        distributed_operator.use_permutation()       = use_permutation;
        auto dependencies                            = add_off_diagonal_operator<T, GeneratorTestType>(recursive_build, distributed_operator, *target_root_cluster, p1, p1_permuted, *source_root_cluster, p2_permuted, use_permutation, data_type, epsilon, eta);

        test = test_vector_product(generator, distributed_operator, *target_root_cluster, MasterOffset_target, *source_root_cluster, MasterOffset_source, mu, op, use_permutation, epsilon);
    } else {
        DefaultApproximationBuilder<T, htool::underlying_type<T>> distributed_operator_holder(generator_permuted, *target_root_cluster, *source_root_cluster, epsilon, eta, Symmetry, UPLO, MPI_COMM_WORLD);

        DistributedOperator<T> &distributed_operator = distributed_operator_holder.distributed_operator;
        distributed_operator.use_permutation()       = use_permutation;
        test                                         = test_vector_product(generator, distributed_operator, *target_root_cluster, MasterOffset_target, *source_root_cluster, MasterOffset_source, mu, op, use_permutation, epsilon);
    }

    return test;
}

template <typename T, typename GeneratorTestType>
bool test_distributed_operator(int nr, int nc, int mu, bool use_permutation, char Symmetry, char UPLO, char op, bool off_diagonal_approximation, DataType data_type, htool::underlying_type<T> epsilon) {
    if (data_type == DataType::DefaultHMatrix)
        return test_default_distributed_operator<T, GeneratorTestType>(nr, nc, mu, use_permutation, Symmetry, UPLO, op, off_diagonal_approximation, data_type, epsilon);
    else
        return test_custom_distributed_operator<T, GeneratorTestType>(nr, nc, mu, use_permutation, Symmetry, UPLO, op, off_diagonal_approximation, data_type, epsilon);
}
