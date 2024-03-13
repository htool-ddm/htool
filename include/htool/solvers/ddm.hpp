#ifndef HTOOL_DDM_HPP
#define HTOOL_DDM_HPP

#include "../distributed_operator/distributed_operator.hpp"
#include "../matrix/matrix.hpp"
#include "../misc/logger.hpp"
#include "../misc/misc.hpp"
#include "../wrappers/wrapper_hpddm.hpp"
#include "../wrappers/wrapper_lapack.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "./geneo/coarse_operator_builder.hpp"
#include "./interfaces/virtual_coarse_operator_builder.hpp"
#include "./interfaces/virtual_coarse_space_builder.hpp"

namespace htool {

template <typename CoefficientPrecision>
class DDM {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    return rankWorld; };

    int n;
    int n_inside;
    std::unique_ptr<HPDDMDense<CoefficientPrecision>> hpddm_op;
    Matrix<CoefficientPrecision> &mat_loc;
    std::vector<double> D;
    int nevi;
    bool one_level;
    bool two_level;
    mutable std::map<std::string, std::string> infos;

    CoefficientPrecision **m_Z;

  public:
    // no copy
    DDM(const DDM &)                       = delete;
    DDM &operator=(const DDM &)            = delete;
    DDM(DDM &&cluster) noexcept            = default;
    DDM &operator=(DDM &&cluster) noexcept = default;
    virtual ~DDM()                         = default;
    void clean() {
        hpddm_op.reset();
    }

    DDM(const DistributedOperator<CoefficientPrecision> &distributed_operator, Matrix<CoefficientPrecision> &local_dense_matrix, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) : n(local_dense_matrix.nb_rows()), n_inside(distributed_operator.get_target_partition().get_size_of_partition(get_rankWorld(distributed_operator.get_comm()))), hpddm_op(std::make_unique<HPDDMDense<CoefficientPrecision>>(&distributed_operator)), mat_loc(local_dense_matrix), D(n), nevi(0), one_level(0), two_level(0) {
        // Timing
        double mytime, maxtime;
        double time = MPI_Wtime();

        // Symmetry and storage
        bool sym = false;
        if (distributed_operator.get_symmetry_type() == 'S' || (distributed_operator.get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) {
            sym = true;

            if (distributed_operator.get_storage_type() == 'U') {
                htool::Logger::get_instance().log(LogLevel::ERROR, "HPDDM takes lower symmetric/hermitian matrices or regular matrices"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] HPDDM takes lower symmetric/hermitian matrices or regular matrices");                  // LCOV_EXCL_LINE
            }
            if (distributed_operator.get_symmetry_type() == 'S' && is_complex<CoefficientPrecision>()) {
                htool::Logger::get_instance().log(LogLevel::WARNING, "A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver"); // LCOV_EXCL_LINE
                // std::cout << "[Htool warning] A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver." << std::endl;
            }
        }

        hpddm_op->initialize(n, sym, mat_loc.data(), neighbors, intersections);

        fill(D.begin(), D.begin() + n_inside, 1);
        fill(D.begin() + n_inside, D.end(), 0);

        hpddm_op->HPDDMDense<CoefficientPrecision>::super::super::initialize(D.data());
        mytime = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

        infos["DDM_setup_one_level_max"] = NbrToStr(maxtime);
    }

    void facto_one_level() {
        double time = MPI_Wtime();
        double mytime, maxtime;
        hpddm_op->callNumfact();
        mytime = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

        infos["DDM_facto_one_level_max"] = NbrToStr(maxtime);
        one_level                        = 1;
    }

    void build_coarse_space(VirtualCoarseSpaceBuilder<CoefficientPrecision> &coarse_space_builder, VirtualCoarseOperatorBuilder<CoefficientPrecision> &coarse_operator_builder) {

        // Timing
        std::vector<double> mytime(3), maxtime(3);

        // Coarse space build
        double time                    = MPI_Wtime();
        Matrix<CoefficientPrecision> Z = coarse_space_builder.build_coarse_space();
        mytime[0]                      = MPI_Wtime() - time;

        nevi = Z.nb_cols();

        HPDDM::Option &opt = *HPDDM::Option::get();
        opt["geneo_nu"]    = nevi;

        // int rankWorld;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
        // Z.csv_save("test_" + NbrToStr(rankWorld));
        CoefficientPrecision **Z_ptr_ptr = new CoefficientPrecision *[nevi];
        CoefficientPrecision *Z_ptr      = Z.release();
        for (int i = 0; i < nevi; i++) {
            Z_ptr_ptr[i] = Z_ptr + i * n;
        }
        hpddm_op->setVectors(Z_ptr_ptr);

        //
        // int rankWorld;
        // MPI_Comm_rank(hpddm_op->HA->get_comm(), &rankWorld);
        int sizeWorld;
        MPI_Comm_size(hpddm_op->HA->get_comm(), &sizeWorld);

        time                                         = MPI_Wtime();
        Matrix<CoefficientPrecision> coarse_operator = coarse_operator_builder.build_coarse_operator(Z.nb_rows(), Z.nb_cols(), Z_ptr_ptr);
        mytime[1]                                    = MPI_Wtime() - time;

        time = MPI_Wtime();
        hpddm_op->buildTwo(MPI_COMM_WORLD, coarse_operator.data());
        mytime[2] = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), mytime.size(), MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

        infos["DDM_geev_max"]       = NbrToStr(maxtime[0]);
        infos["DDM_setup_ZtAZ_max"] = NbrToStr(maxtime[1]);
        infos["DDM_facto_ZtAZ_max"] = NbrToStr(maxtime[2]);
        infos["GenEO_coarse_size"]  = NbrToStr(coarse_operator.nb_cols());
        two_level                   = 1;
    }

    // void build_coarse_space(Matrix<CoefficientPrecision> &Ki) {
    //     // Timing
    //     double mytime, maxtime;
    //     double time = MPI_Wtime();

    //     //
    //     int info;

    //     // Partition of unity
    //     Matrix<CoefficientPrecision> DAiD(n, n);
    //     for (int i = 0; i < n_inside; i++) {
    //         std::copy_n(mat_loc.data() + i * n, n_inside, &(DAiD(0, i)));
    //     }

    //     // Build local eigenvalue problem
    //     int ldvl = n, ldvr = n, lwork = -1;
    //     int lda = n, ldb = n;
    //     std::vector<CoefficientPrecision> work(n);
    //     std::vector<double> rwork;
    //     std::vector<int> index(n, 0);
    //     std::iota(index.begin(), index.end(), int(0));

    //     // int rankWorld;
    //     // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    //     // DAiD.csv_save("DAiD_" + NbrToStr(rankWorld));
    //     // Ki.csv_save("Ki_" + NbrToStr(rankWorld));

    //     if (hpddm_op->HA->get_symmetry_type() == 'S' || hpddm_op->HA->get_symmetry_type() == 'H') {
    //         char uplo = hpddm_op->HA->get_storage_type();
    //         int itype = 1;
    //         std::vector<underlying_type<CoefficientPrecision>> w(n);
    //         if (is_complex<CoefficientPrecision>()) {
    //             rwork.resize(3 * n - 2);
    //         }

    //         Lapack<CoefficientPrecision>::gv(&itype, "V", &uplo, &n, DAiD.data(), &lda, Ki.data(), &ldb, w.data(), work.data(), &lwork, rwork.data(), &info);
    //         lwork = (int)std::real(work[0]);
    //         work.resize(lwork);
    //         Lapack<CoefficientPrecision>::gv(&itype, "V", &uplo, &n, DAiD.data(), &lda, Ki.data(), &ldb, w.data(), work.data(), &lwork, rwork.data(), &info);

    //         std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
    //             return (std::abs(w[a]) > std::abs(w[b]));
    //         });

    //         // if (rankWorld == 0) {
    //         //     for (int i = 0; i < index.size(); i++) {
    //         //         std::cout << std::abs(w[index[i]]) << " ";
    //         //     }
    //         //     std::cout << "\n";
    //         //     // std::cout << vr << "\n";
    //         // }
    //         // MPI_Barrier(hpddm_op->HA->get_comm());
    //         // if (rankWorld == 1) {
    //         //     std::cout << "w : " << w << "\n";
    //         //     std::cout << "info: " << info << "\n";
    //         //     for (int i = 0; i < index.size(); i++) {
    //         //         std::cout << std::abs(w[index[i]]) << " ";
    //         //     }
    //         //     std::cout << "\n";
    //         //     // std::cout << vr << "\n";
    //         // }

    //         HPDDM::Option &opt = *HPDDM::Option::get();
    //         nevi               = 0;
    //         double threshold   = opt.val("geneo_threshold", -1.0);
    //         if (threshold > 0.0) {
    //             while (std::abs(w[index[nevi]]) > threshold && nevi < index.size()) {
    //                 nevi++;
    //             }

    //         } else {
    //             nevi = opt.val("geneo_nu", 2);
    //         }

    //         opt["geneo_nu"] = nevi;
    //         m_Z             = new CoefficientPrecision *[nevi];
    //         *m_Z            = new CoefficientPrecision[nevi * n];
    //         for (int i = 0; i < nevi; i++) {
    //             m_Z[i] = *m_Z + i * n;
    //             std::copy_n(DAiD.data() + index[i] * n, n_inside, m_Z[i]);
    //             for (int j = n_inside; j < n; j++) {

    //                 m_Z[i][j] = 0;
    //             }
    //         }
    //     } else {
    //         if (is_complex<CoefficientPrecision>()) {
    //             rwork.resize(8 * n);
    //         }
    //         std::vector<CoefficientPrecision> alphar(n), alphai((is_complex<CoefficientPrecision>() ? 0 : n)), beta(n);
    //         std::vector<CoefficientPrecision> vl(n * n), vr(n * n);

    //         Lapack<CoefficientPrecision>::ggev("N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alphar.data(), alphai.data(), beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
    //         lwork = (int)std::real(work[0]);
    //         work.resize(lwork);
    //         Lapack<CoefficientPrecision>::ggev("N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alphar.data(), alphai.data(), beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);

    //         std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
    //             return ((std::abs(beta[a]) < 1e-15 || (std::abs(alphar[a] / beta[a]) > std::abs(alphar[b] / beta[b]))) && !(std::abs(beta[b]) < 1e-15));
    //         });

    //         // if (rankWorld == 0) {
    //         //     // for (int i = 0; i < index.size(); i++) {
    //         //     //     std::cout << std::abs(alphar[index[i]] / beta[index[i]]) << " ";
    //         //     // }
    //         //     // std::cout << "\n";
    //         //     std::cout << vr << "\n";
    //         // }
    //         // MPI_Barrier(hpddm_op->HA->get_comm());
    //         // if (rankWorld == 1) {
    //         //     // std::cout << "alphar : " << alphar << "\n";
    //         //     // std::cout << "alphai : " << alphai << "\n";
    //         //     // std::cout << "beta : " << beta << "\n";
    //         //     // std::cout << "info: " << info << "\n";
    //         //     // for (int i = 0; i < index.size(); i++) {
    //         //     //     std::cout << std::abs(alphar[index[i]] / beta[index[i]]) << " ";
    //         //     // }
    //         //     // std::cout << "\n";
    //         //     std::cout << vr << "\n";
    //         // }

    //         HPDDM::Option &opt = *HPDDM::Option::get();
    //         nevi               = 0;
    //         double threshold   = opt.val("geneo_threshold", -1.0);
    //         if (threshold > 0.0) {
    //             while (std::abs(beta[index[nevi]]) < 1e-15 || (std::abs(alphar[index[nevi]] / beta[index[nevi]]) > threshold && nevi < index.size())) {
    //                 nevi++;
    //             }

    //         } else {
    //             nevi = opt.val("geneo_nu", 2);
    //         }

    //         opt["geneo_nu"] = nevi;
    //         m_Z             = new CoefficientPrecision *[nevi];
    //         *m_Z            = new CoefficientPrecision[nevi * n];
    //         for (int i = 0; i < nevi; i++) {
    //             m_Z[i] = *m_Z + i * n;
    //             std::copy_n(vr.data() + index[i] * n, n_inside, m_Z[i]);
    //             for (int j = n_inside; j < n; j++) {

    //                 m_Z[i][j] = 0;
    //             }
    //         }
    //     }

    //     // Matrix<CoefficientPrecision> Z_test(n, nevi);
    //     // Z_test.assign(n, nevi, m_Z[0], false);
    //     // // int rankWorld;
    //     // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    //     // Z_test.csv_save("test_2_" + NbrToStr(rankWorld));

    //     hpddm_op->setVectors(m_Z);

    //     // timing
    //     mytime = MPI_Wtime() - time;
    //     MPI_Barrier(hpddm_op->HA->get_comm());
    //     time = MPI_Wtime();
    //     MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());
    //     infos["DDM_geev_max"] = NbrToStr(maxtime);

    //     //
    //     build_E(m_Z);
    // }

    // void build_E(CoefficientPrecision *const *Z) {
    //     //
    //     int sizeWorld;
    //     MPI_Comm_size(hpddm_op->HA->get_comm(), &sizeWorld);

    //     // Timing
    //     std::vector<double> mytime(2), maxtime(2);
    //     double time = MPI_Wtime();

    //     // Allgather
    //     std::vector<int> recvcounts(sizeWorld);
    //     std::vector<int> displs(sizeWorld);
    //     MPI_Allgather(&nevi, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, hpddm_op->HA->get_comm());

    //     displs[0] = 0;
    //     for (int i = 1; i < sizeWorld; i++) {
    //         displs[i] = displs[i - 1] + recvcounts[i - 1];
    //     }

    //     Matrix<CoefficientPrecision> E;
    //     htool::build_geneo_coarse_operator(*hpddm_op->HA, nevi, n, Z, E);
    //     mytime[0] = MPI_Wtime() - time;

    //     time = MPI_Wtime();
    //     hpddm_op->buildTwo(MPI_COMM_WORLD, E.release());
    //     mytime[1] = MPI_Wtime() - time;

    //     // Timing
    //     MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

    //     infos["DDM_setup_ZtAZ_max"] = NbrToStr(maxtime[0]);
    //     infos["DDM_facto_ZtAZ_max"] = NbrToStr(maxtime[1]);
    //     infos["GenEO_coarse_size"]  = NbrToStr(E.nb_cols());
    //     two_level                   = 1;
    // }

    void solve(const CoefficientPrecision *const rhs, CoefficientPrecision *const x, const int &mu = 1) {
        // Check facto
        if (!one_level && two_level) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "Factorisation for one-level missing"); // LCOV_EXCL_LINE
            // throw std::logic_error("[Htool error] Factorisation for one-level missing"); // LCOV_EXCL_LINE
        }

        // Eventually change one-level type
        HPDDM::Option &opt = *HPDDM::Option::get();
        switch (opt.val("schwarz_method", 0)) {
        case HPDDM_SCHWARZ_METHOD_NONE:
            hpddm_op->setType(HPDDMDense<CoefficientPrecision>::Prcndtnr::NO);
            break;
        case HPDDM_SCHWARZ_METHOD_RAS:
            hpddm_op->setType(HPDDMDense<CoefficientPrecision>::Prcndtnr::GE);
            break;
        case HPDDM_SCHWARZ_METHOD_ASM:
            hpddm_op->setType(HPDDMDense<CoefficientPrecision>::Prcndtnr::SY);
            break;
            // case HPDDM_SCHWARZ_METHOD_OSM:
            // hpddm_op->setType(HPDDM::Schwarz::Prcndtnr::NO);
            // break;
            // case HPDDM_SCHWARZ_METHOD_ORAS:
            // hpddm_op->setType(HPDDM::Schwarz::Prcndtnr::NO);
            // break;
            // case HPDDM_SCHWARZ_METHOD_SORAS:
            // hpddm_op->setType(HPDDM::Schwarz::Prcndtnr::NO);
            // break;
        }

        //
        MPI_Comm comm = hpddm_op->HA->get_comm();
        int rankWorld;
        int sizeWorld;
        MPI_Comm_rank(comm, &rankWorld);
        MPI_Comm_size(comm, &sizeWorld);
        int offset = hpddm_op->HA->get_target_partition().get_offset_of_partition(rankWorld);
        // int size        = hpddm_op->HA->get_local_size();
        int nb_rows = hpddm_op->HA->get_target_partition().get_global_size();
        // int nb_vec_prod = StrToNbr<int>(hpddm_op->HA->get_infos("nb_mat_vec_prod"));
        double time = MPI_Wtime();

        //
        std::vector<CoefficientPrecision> rhs_perm(nb_rows);
        std::vector<CoefficientPrecision> x_local(n * mu, 0);
        std::vector<CoefficientPrecision> local_rhs(n * mu, 0);
        hpddm_op->in_global->resize(nb_rows * (mu == 1 ? 1 : 2 * mu));
        hpddm_op->buffer->resize(n_inside * (mu == 1 ? 1 : 2 * mu));

        // TODO: blocking ?
        for (int i = 0; i < mu; i++) {
            // Permutation
            hpddm_op->HA->get_target_partition().global_to_partition_numbering(rhs + i * nb_rows, rhs_perm.data());
            // global_to_root_cluster(hpddm_op->HA->get_root_target_cluster(), rhs + i * nb_rows, rhs_perm.data());
            // hpddm_op->HA->target_to_cluster_permutation(rhs + i * nb_rows, rhs_perm.data());

            std::copy_n(rhs_perm.begin() + offset, n_inside, local_rhs.begin() + i * n);
        }

        // TODO: avoid com here
        // for (int i=0;i<n-n_inside;i++){
        //   local_rhs[i]=rhs_perm[]
        // }
        hpddm_op->scaledexchange(local_rhs.data(), mu);

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*hpddm_op, local_rhs.data(), x_local.data(), mu, comm);

        // Delete the overlap (useful only when mu>1 and n!=n_inside)
        for (int i = 0; i < mu; i++) {
            std::copy_n(x_local.data() + i * n, n_inside, local_rhs.data() + i * n_inside);
        }

        // Local to global
        // hpddm_op->HA->local_to_global(x_local.data(),hpddm_op->in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            // recvcounts[i] = (hpddm_op->HA->get_root_target_cluster().get_clusters_on_partition()[i]->get_size()) * mu;
            recvcounts[i] = (hpddm_op->HA->get_target_partition().get_size_of_partition(i)) * mu;
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(local_rhs.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), hpddm_op->in_global->data() + (mu == 1 ? 0 : mu * nb_rows), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

        //

        for (int i = 0; i < mu; i++) {
            if (mu != 1) {
                for (int j = 0; j < sizeWorld; j++) {
                    std::copy_n(hpddm_op->in_global->data() + mu * nb_rows + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, hpddm_op->in_global->data() + i * nb_rows + displs[j] / mu);
                }
            }

            // Permutation
            hpddm_op->HA->get_target_partition().partition_to_global_numbering(hpddm_op->in_global->data() + i * nb_rows, x + i * nb_rows);
            // root_cluster_to_global(hpddm_op->HA->get_root_target_cluster(), hpddm_op->in_global->data() + i * nb_rows, x + i * nb_rows);
            // hpddm_op->HA->cluster_to_target_permutation(hpddm_op->in_global->data() + i * nb_rows, x + i * nb_rows);
        }

        // Infos
        time                   = MPI_Wtime() - time;
        infos["Solve"]         = NbrToStr(time);
        infos["Nb_it"]         = NbrToStr(nb_it);
        infos["Nb_subdomains"] = NbrToStr(sizeWorld);
        // infos["nb_mat_vec_prod"]        = NbrToStr(StrToNbr<int>(hpddm_op->HA->get_infos("nb_mat_vec_prod")) - nb_vec_prod);
        // infos["mean_time_mat_vec_prod"] = NbrToStr(StrToNbr<double>(hpddm_op->HA->get_infos("total_time_mat_vec_prod")) / StrToNbr<double>(hpddm_op->HA->get_infos("nb_mat_vec_prod")));
        switch (opt.val("schwarz_method", 0)) {
        case HPDDM_SCHWARZ_METHOD_NONE:
            infos["Precond"] = "None";
            break;
        case HPDDM_SCHWARZ_METHOD_RAS:
            infos["Precond"] = "RAS";
            break;
        case HPDDM_SCHWARZ_METHOD_ASM:
            infos["Precond"] = "ASM";
            break;
            // case HPDDM_SCHWARZ_METHOD_OSM:
            // infos["Precond"] = "OSM";
            // break;
            // case HPDDM_SCHWARZ_METHOD_ORAS:
            // infos["Precond"] = "ORAS";
            // break;
            // case HPDDM_SCHWARZ_METHOD_SORAS:
            // infos["Precond"] = "SORAS";
            // break;
        }

        switch (opt.val("krylov_method", 8)) {
        case HPDDM_KRYLOV_METHOD_GMRES:
            infos["krylov_method"] = "gmres";
            break;
        case HPDDM_KRYLOV_METHOD_BGMRES:
            infos["krylov_method"] = "bgmres";
            break;
        case HPDDM_KRYLOV_METHOD_CG:
            infos["krylov_method"] = "cg";
            break;
        case HPDDM_KRYLOV_METHOD_BCG:
            infos["krylov_method"] = "bcg";
            break;
        case HPDDM_KRYLOV_METHOD_GCRODR:
            infos["krylov_method"] = "gcrodr";
            break;
        case HPDDM_KRYLOV_METHOD_BGCRODR:
            infos["krylov_method"] = "bgcrodr";
            break;
        case HPDDM_KRYLOV_METHOD_BFBCG:
            infos["krylov_method"] = "bfbcg";
            break;
        case HPDDM_KRYLOV_METHOD_RICHARDSON:
            infos["krylov_method"] = "richardson";
            break;
        case HPDDM_KRYLOV_METHOD_NONE:
            infos["krylov_method"] = "none";
            break;
        }

        if (infos["Precond"] == "None") {
            infos["Coarse_correction"] = "None";
        } else {
            int nevi_mean = nevi;
            int nevi_max  = nevi;
            int nevi_min  = nevi;

            if (rankWorld == 0) {
                MPI_Reduce(MPI_IN_PLACE, &(nevi_mean), 1, MPI_INT, MPI_SUM, 0, hpddm_op->HA->get_comm());
                MPI_Reduce(MPI_IN_PLACE, &(nevi_max), 1, MPI_INT, MPI_MAX, 0, hpddm_op->HA->get_comm());
                MPI_Reduce(MPI_IN_PLACE, &(nevi_min), 1, MPI_INT, MPI_MIN, 0, hpddm_op->HA->get_comm());
            } else {
                MPI_Reduce(&(nevi_mean), &(nevi_mean), 1, MPI_INT, MPI_SUM, 0, hpddm_op->HA->get_comm());
                MPI_Reduce(&(nevi_max), &(nevi_max), 1, MPI_INT, MPI_MAX, 0, hpddm_op->HA->get_comm());
                MPI_Reduce(&(nevi_min), &(nevi_min), 1, MPI_INT, MPI_MIN, 0, hpddm_op->HA->get_comm());
            }

            infos["DDM_local_coarse_size_mean"] = NbrToStr((double)nevi_mean / (double)sizeWorld);
            infos["DDM_local_coarse_size_max"]  = NbrToStr(nevi_max);
            infos["DDM_local_coarse_size_min"]  = NbrToStr(nevi_min);

            switch (opt.val("schwarz_coarse_correction", -1)) {
            case HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED:
                infos["Coarse_correction"] = "Balanced";
                break;
            case HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED:
                infos["Coarse_correction"] = "Deflated";
                break;
            case HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE:
                infos["Coarse_correction"] = "Additive";
                break;
            default:
                infos["Coarse_correction"] = "None";
            }
        }
        infos["htool_solver"] = "ddm";
    }

    void print_infos() const {
        int rankWorld;
        MPI_Comm_rank(hpddm_op->HA->get_comm(), &rankWorld);
        if (rankWorld == 0) {
            for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
                std::cout << it->first << "\t" << it->second << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const {
        int rankWorld;
        MPI_Comm_rank(hpddm_op->HA->get_comm(), &rankWorld);
        if (rankWorld == 0) {
            std::ofstream outputfile(outputname, mode);
            if (outputfile) {
                for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
                    outputfile << it->first << sep << it->second << std::endl;
                }
                outputfile.close();
            } else {
                std::cout << "Unable to create " << outputname << std::endl;
            }
        }
    }

    void add_infos(std::string key, std::string value) const {
        if (hpddm_op->HA->get_rankworld() == 0) {
            if (infos.find(key) == infos.end()) {
                infos[key] = value;
            } else {
                infos[key] = NbrToStr(StrToNbr<double>(infos[key]) + StrToNbr<double>(value));
            }
        }
    }

    void set_infos(std::string key, std::string value) const {
        if (hpddm_op->HA->get_rankworld() == 0) {
            infos[key] = value;
        }
    }

    std::string get_information(const std::string &key) const {
        return infos[key];
    }
    std::map<std::string, std::string> get_information() const {
        return infos;
    }
    int get_nevi() const {
        return nevi;
    }
    int get_local_size() const {
        return n;
    }

    // MPI_Comm get_comm() const {
    //     return comm;
    // }
};

} // namespace htool
#endif
