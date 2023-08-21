#ifndef HTOOL_DDM_HPP
#define HTOOL_DDM_HPP

#include "../basic_types/matrix.hpp"
#include "../distributed_operator/distributed_operator.hpp"
#include "../misc/logger.hpp"
#include "../misc/misc.hpp"
#include "../wrappers/wrapper_hpddm.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "coarse_space.hpp"

namespace htool {

template <typename CoefficientPrecision>
class DDM {
  private:
    int n;
    int n_inside;
    std::vector<int> neighbors;
    std::vector<int> renum_to_global;
    // const std::vector<int> cluster_to_ovr_subdomain;
    std::vector<std::vector<int>> intersections;
    std::vector<CoefficientPrecision> vec_ovr;
    std::unique_ptr<HPDDMDense<CoefficientPrecision>> hpddm_op;
    std::unique_ptr<Matrix<CoefficientPrecision>> mat_loc;
    std::vector<double> D;
    int nevi;
    int size_E;
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

    // Without overlap
    DDM(const DistributedOperator<CoefficientPrecision> *distributed_operator, std::unique_ptr<Matrix<CoefficientPrecision>> local_dense_matrix) : n(local_dense_matrix->nb_rows()), n_inside(n), hpddm_op(std::make_unique<HPDDMDense<CoefficientPrecision>>(distributed_operator)), mat_loc(std::move(local_dense_matrix)), D(n), nevi(0), size_E(0), one_level(0), two_level(0) {
        double mytime, maxtime;
        double time = MPI_Wtime();
        bool sym    = false;
        if (distributed_operator->get_symmetry_type() == 'S' || (distributed_operator->get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) {
            sym = true;
            if (distributed_operator->get_storage_type() == 'U') {
                htool::Logger::get_instance().log(LogLevel::ERROR, "HPDDM takes lower symmetric/hermitian matrices or regular matrices"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] HPDDM takes lower symmetric/hermitian matrices or regular matrices"); // LCOV_EXCL_LINE
            }
            if (distributed_operator->get_symmetry_type() == 'S' && is_complex<CoefficientPrecision>()) {
                htool::Logger::get_instance().log(LogLevel::WARNING, "A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver"); // LCOV_EXCL_LINE
                // std::cout << "[Htool warning] A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver." << std::endl;
            }
        }

        hpddm_op->initialize(n, sym, mat_loc->data(), neighbors, intersections);

        fill(D.begin(), D.begin() + n_inside, 1);
        fill(D.begin() + n_inside, D.end(), 0);

        hpddm_op->HPDDMDense<CoefficientPrecision>::super::super::initialize(D.data());
        mytime = MPI_Wtime() - time;
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

        infos["DDM_setup_one_level_max"] = NbrToStr(maxtime);
    }

    // DDM(const DistributedOperator<CoefficientPrecision> *const hmat_0, std::shared_ptr<HPDDMDense<CoefficientPrecision>> myddm_op = nullptr, const Matrix<CoefficientPrecision> *localblock = nullptr) : n(hmat_0->get_local_target_cluster().get_size()), n_inside(n), mat_loc(n * n), D(n), comm(hmat_0->get_comm()), nevi(0), size_E(0), one_level(0), two_level(0) {
    //     if (myddm_op == nullptr)
    //         hpddm_op = std::make_shared<HPDDMDense<CoefficientPrecision>>(hmat_0);
    //     else
    //         hpddm_op = myddm_op;
    //     // Timing
    //     double mytime, maxtime;
    //     double time = MPI_Wtime();

    //     // Building Ai
    //     bool sym = false;
    //     if (hmat_0->get_symmetry_type() == 'S' || (hmat_0->get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) {
    //         sym = true;
    //         if (hmat_0->get_storage_type() == 'U') {
    //             throw std::invalid_argument("[Htool error] HPDDM takes lower symmetric/hermitian matrices or regular matrices"); // LCOV_EXCL_LINE
    //         }
    //         if (hmat_0->get_symmetry_type() == 'S' && is_complex<CoefficientPrecision>()) {
    //             std::cout << "[Htool warning] A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver." << std::endl;
    //         }
    //     }
    //     int rankWorld;
    //     int sizeWorld;
    //     MPI_Comm_rank(comm, &rankWorld);
    //     MPI_Comm_size(comm, &sizeWorld);
    //     if (!localblock) {
    //         hpddm_op->HA->copy_local_diagonal_block_to_dense(mat_loc.data());

    //         // Matrix<CoefficientPrecision> diagonal_block = hpddm_op->HA->get_local_diagonal_block(false);
    //         // std::copy_n(diagonal_block.data(), diagonal_block.nb_rows() * diagonal_block.nb_cols(), mat_loc.data());
    //     } else {
    //         std::copy_n(localblock->data(), localblock->nb_rows() * localblock->nb_cols(), mat_loc.data());
    //     }
    //     // int rankWorld;
    //     // int sizeWorld;
    //     // MPI_Comm_rank(comm, &rankWorld);
    //     // MPI_Comm_size(comm, &sizeWorld);
    //     // Matrix<CoefficientPrecision> test;
    //     // test.assign(n, n, mat_loc.data(), false);
    //     // test.csv_save("local_mat_" + NbrToStr(rankWorld) + ".csv");

    //     std::vector<int> neighbors;
    //     std::vector<std::vector<int>> intersections;
    //     hpddm_op->initialize(n, sym, mat_loc.data(), neighbors, intersections);

    //     fill(D.begin(), D.begin() + n_inside, 1);
    //     fill(D.begin() + n_inside, D.end(), 0);

    //     hpddm_op->HPDDMDense<CoefficientPrecision>::super::super::initialize(D.data());
    //     mytime = MPI_Wtime() - time;

    //     // Timing
    //     MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

    //     infos["DDM_setup_one_level_max"] = NbrToStr(maxtime);
    // }

    // With overlap
    DDM(const DistributedOperator<CoefficientPrecision> *distributed_operator, const Matrix<CoefficientPrecision> &local_dense_matrix, const VirtualGeneratorWithPermutation<CoefficientPrecision> &mat0, const std::vector<int> &ovr_subdomain_to_global0, const std::vector<int> &cluster_to_ovr_subdomain0, const std::vector<int> &neighbors0, const std::vector<std::vector<int>> &intersections0) : n(ovr_subdomain_to_global0.size()), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(n), hpddm_op(std::make_unique<HPDDMDense<CoefficientPrecision>>(distributed_operator)), mat_loc(std::make_unique<Matrix<CoefficientPrecision>>(n, n)), D(n), one_level(0), two_level(0) {
        // Timing
        double mytime, maxtime;
        double time = MPI_Wtime();

        std::vector<int> renum(n, -1);
        renum_to_global.resize(n);

        for (int i = 0; i < cluster_to_ovr_subdomain0.size(); i++) {
            renum[cluster_to_ovr_subdomain0[i]] = i;
            renum_to_global[i]                  = ovr_subdomain_to_global0[cluster_to_ovr_subdomain0[i]];
        }
        int count = cluster_to_ovr_subdomain0.size();
        // std::cout << count << std::endl;
        for (int i = 0; i < n; i++) {
            if (renum[i] == -1) {
                renum[i]                 = count;
                renum_to_global[count++] = ovr_subdomain_to_global0[i];
            }
        }

        intersections.resize(neighbors.size());
        for (int i = 0; i < neighbors.size(); i++) {
            intersections[i].resize(intersections0[i].size());
            for (int j = 0; j < intersections[i].size(); j++) {
                intersections[i][j] = renum[intersections0[i][j]];
            }
        }

        // Symmetry and storage
        bool sym = false;
        if (distributed_operator->get_symmetry_type() == 'S' || (distributed_operator->get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) {
            sym = true;

            if (distributed_operator->get_storage_type() == 'U') {
                htool::Logger::get_instance().log(LogLevel::ERROR, "HPDDM takes lower symmetric/hermitian matrices or regular matrices"); // LCOV_EXCL_LINE
                // throw std::invalid_argument("[Htool error] HPDDM takes lower symmetric/hermitian matrices or regular matrices");                  // LCOV_EXCL_LINE
            }
            if (distributed_operator->get_symmetry_type() == 'S' && is_complex<CoefficientPrecision>()) {
                htool::Logger::get_instance().log(LogLevel::WARNING, "A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver"); // LCOV_EXCL_LINE
                // std::cout << "[Htool warning] A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver." << std::endl;
            }
        }

        // Building Ai
        // Matrix<CoefficientPrecision> main_diagonal_block = hpddm_op->HA->get_local_diagonal_block(false);
        for (int j = 0; j < n_inside; j++) {
            std::copy_n(local_dense_matrix.data() + j * n_inside, n_inside, mat_loc->data() + j * n);
        }

        // Overlap
        std::vector<CoefficientPrecision> horizontal_block((n - n_inside) * n_inside), diagonal_block((n - n_inside) * (n - n_inside));

        std::vector<int> overlap_num(renum_to_global.begin() + n_inside, renum_to_global.end());
        std::vector<int> inside_num(renum_to_global.begin(), renum_to_global.begin() + n_inside);

        mat0.copy_submatrix(n - n_inside, n_inside, overlap_num.data(), inside_num.data(), horizontal_block.data());
        for (int j = 0; j < n_inside; j++) {
            std::copy_n(horizontal_block.begin() + j * (n - n_inside), n - n_inside, mat_loc->data() + n_inside + j * n);
        }

        mat0.copy_submatrix(n - n_inside, n - n_inside, overlap_num.data(), overlap_num.data(), diagonal_block.data());
        for (int j = 0; j < n - n_inside; j++) {
            std::copy_n(diagonal_block.begin() + j * (n - n_inside), n - n_inside, mat_loc->data() + n_inside + (j + n_inside) * n);
        }

        if (!sym) {
            std::vector<CoefficientPrecision> vertical_block(n_inside * (n - n_inside));
            mat0.copy_submatrix(n_inside, n - n_inside, inside_num.data(), overlap_num.data(), vertical_block.data());
            for (int j = n_inside; j < n; j++) {
                std::copy_n(vertical_block.begin() + (j - n_inside) * n_inside, n_inside, mat_loc->data() + j * n);
            }
        }

        // TODO: add symmetric eigensolver
        if (sym) {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < j; i++) {

                    (*mat_loc)(i, j) = (*mat_loc)(j, i);
                }
            }
        }

        hpddm_op->initialize(n, sym, mat_loc->data(), neighbors, intersections);

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

    // TODO: take local VirtualHMatrix instead
    // void build_coarse_space(Matrix<T> &Mi, VirtualGenerator<T> &generator_Bi, const std::vector<R3> &x) {
    //     // Timing
    //     double mytime, maxtime;
    //     double time = MPI_Wtime();

    //     //
    //     int info;

    //     // Building Neumann matrix
    //     htool::VirtualHMatrix *const Bi(generator_Bi, hpddm_op->HA.get_cluster_tree_t().get_local_cluster_tree(), x, -1, MPI_COMM_SELF);
    //     Matrix<T> Bi(n, n);

    //     // Building Bi
    //     bool sym                                                                = false;
    //     const std::vector<LowRankMatrix<T, ClusterImpl> *> &MyLocalFarFieldMats = HBi.get_MyFarFieldMats();
    //     const std::vector<SubMatrix<T> *> &MyLocalNearFieldMats                 = HBi.get_MyNearFieldMats();

    //     // Internal dense blocks
    //     for (int i = 0; i < MyLocalNearFieldMats.size(); i++) {
    //         const SubMatrix<T> &submat = *(MyLocalNearFieldMats[i]);
    //         int local_nr               = submat.nb_rows();
    //         int local_nc               = submat.nb_cols();
    //         int offset_i               = submat.get_offset_i() - hpddm_op->HA.get_local_offset();
    //         int offset_j               = submat.get_offset_j() - hpddm_op->HA.get_local_offset();
    //         for (int i = 0; i < local_nc; i++) {
    //             std::copy_n(&(submat(0, i)), local_nr, Bi.data() + offset_i + (offset_j + i) * n);
    //         }
    //     }

    //     // Internal compressed block
    //     Matrix<T> FarFielBlock(n, n);
    //     for (int i = 0; i < MyLocalFarFieldMats.size(); i++) {
    //         const LowRankMatrix<T, ClusterImpl> &lmat = *(MyLocalFarFieldMats[i]);
    //         int local_nr                              = lmat.nb_rows();
    //         int local_nc                              = lmat.nb_cols();
    //         int offset_i                              = lmat.get_offset_i() - hpddm_op->HA.get_local_offset();
    //         int offset_j                              = lmat.get_offset_j() - hpddm_op->HA.get_local_offset();
    //         ;
    //         FarFielBlock.resize(local_nr, local_nc);
    //         lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
    //         for (int i = 0; i < local_nc; i++) {
    //             std::copy_n(&(FarFielBlock(0, i)), local_nr, Bi.data() + offset_i + (offset_j + i) * n);
    //         }
    //     }

    //     // Overlap
    //     std::vector<T> horizontal_block(n - n_inside, n_inside), vertical_block(n, n - n_inside);
    //     horizontal_block = generator_Bi.get_submatrix(std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end()), std::vector<int>(renum_to_global.begin(), renum_to_global.begin() + n_inside)).get_mat();
    //     vertical_block   = generator_Bi.get_submatrix(renum_to_global, std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end())).get_mat();
    //     for (int j = 0; j < n_inside; j++) {
    //         std::copy_n(horizontal_block.begin() + j * (n - n_inside), n - n_inside, Bi.data() + n_inside + j * n);
    //     }
    //     for (int j = n_inside; j < n; j++) {
    //         std::copy_n(vertical_block.begin() + (j - n_inside) * n, n, Bi.data() + j * n);
    //     }

    //     // LU facto for mass matrix
    //     int lda = n;
    //     std::vector<int> _ipiv_mass(n);
    //     HPDDM::Lapack<Cplx>::getrf(&n, &n, Mi.data(), &lda, _ipiv_mass.data(), &info);

    //     // Partition of unity
    //     Matrix<T> DAiD(n, n);
    //     for (int i = 0; i < n_inside; i++) {
    //         std::copy_n(&(mat_loc[i * n]), n_inside, &(DAiD(0, i)));
    //     }

    //     // M^-1
    //     const char l = 'N';
    //     lda          = n;
    //     int ldb      = n;
    //     HPDDM::Lapack<Cplx>::getrs(&l, &n, &n, Mi.data(), &lda, _ipiv_mass.data(), DAiD.data(), &ldb, &info);
    //     HPDDM::Lapack<Cplx>::getrs(&l, &n, &n, Mi.data(), &lda, _ipiv_mass.data(), Bi.data(), &ldb, &info);

    //     // Build local eigenvalue problem
    //     Matrix<T> evp(n, n);
    //     Bi.mvprod(DAiD.data(), evp.data(), n);

    //     // eigenvalue problem
    //     hpddm_op->solveEVP(evp.data());
    //     T *const *Z        = const_cast<T *const *>(hpddm_op->getVectors());
    //     HPDDM::Option &opt = *HPDDM::Option::get();
    //     nevi               = opt.val("geneo_nu", 2);

    //     // timing
    //     mytime = MPI_Wtime() - time;
    //     time   = MPI_Wtime();
    //     MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());
    //     infos["DDM_geev_max"] = NbrToStr(maxtime);

    //     //
    //     build_E(Z);
    // }

    void build_coarse_space(Matrix<CoefficientPrecision> &Ki) {
        // Timing
        double mytime, maxtime;
        double time = MPI_Wtime();

        //
        int info;

        // Partition of unity
        Matrix<CoefficientPrecision> DAiD(n, n);
        for (int i = 0; i < n_inside; i++) {
            std::copy_n(mat_loc->data() + i * n, n_inside, &(DAiD(0, i)));
        }

        // Build local eigenvalue problem
        int ldvl = n, ldvr = n, lwork = -1;
        int lda = n, ldb = n;
        std::vector<CoefficientPrecision> alphar(n), alphai((is_complex<CoefficientPrecision>() ? 0 : n)), beta(n);
        std::vector<CoefficientPrecision> work(n);
        std::vector<double> rwork(8 * n);
        std::vector<CoefficientPrecision> vl(n * n), vr(n * n);
        std::vector<int> index(n, 0);

        int rankWorld;
        MPI_Comm_rank(hpddm_op->HA->get_comm(), &rankWorld);
        // if (rankWorld == 0) {
        //     DAiD.csv_save("DAiD_" + NbrToStr(rankWorld));
        //     Ki.csv_save("Ki_" + NbrToStr(rankWorld));
        //     // std::cout << vr << "\n";
        // }
        // MPI_Barrier(comm);
        // if (rankWorld == 1) {
        //     std::cout << DAiD.nb_rows() << " " << DAiD.nb_cols() << "\n";
        //     std::cout << "taille " << alphai.size() << "\n";
        //     std::cout << Ki.nb_rows() << " " << Ki.nb_cols() << "\n";
        //     DAiD.csv_save("DAiD_" + NbrToStr(rankWorld));
        //     Ki.csv_save("Ki_" + NbrToStr(rankWorld));
        //     // std::cout << vr << "\n";
        // }

        HPDDM::Lapack<CoefficientPrecision>::ggev("N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alphar.data(), alphai.data(), beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<CoefficientPrecision>::ggev("N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alphar.data(), alphai.data(), beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);

        for (int i = 0; i != index.size(); i++) {
            index[i] = i;
        }
        std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
            return ((std::abs(beta[a]) < 1e-15 || (std::abs(alphar[a] / beta[a]) > std::abs(alphar[b] / beta[b]))) && !(std::abs(beta[b]) < 1e-15));
        });

        // if (rankWorld == 0) {
        //     for (int i = 0; i < index.size(); i++) {
        //         std::cout << std::abs(alphar[index[i]] / beta[index[i]]) << " ";
        //     }
        //     std::cout << "\n";
        //     // std::cout << vr << "\n";
        // }
        // MPI_Barrier(comm);
        // if (rankWorld == 1) {
        //     std::cout << "alphar : " << alphar << "\n";
        //     std::cout << "alphai : " << alphai << "\n";
        //     std::cout << "beta : " << beta << "\n";
        //     std::cout << "info: " << info << "\n";
        //     for (int i = 0; i < index.size(); i++) {
        //         std::cout << std::abs(alphar[index[i]] / beta[index[i]]) << " ";
        //     }
        //     std::cout << "\n";
        //     // std::cout << vr << "\n";
        // }
        HPDDM::Option &opt = *HPDDM::Option::get();
        nevi               = 0;
        double threshold   = opt.val("geneo_threshold", -1.0);
        if (threshold > 0.0) {
            while (std::abs(beta[index[nevi]]) < 1e-15 || (std::abs(alphar[index[nevi]] / beta[index[nevi]]) > threshold && nevi < index.size())) {
                nevi++;
            }

        } else {
            nevi = opt.val("geneo_nu", 2);
        }

        opt["geneo_nu"] = nevi;
        m_Z             = new CoefficientPrecision *[nevi];
        *m_Z            = new CoefficientPrecision[nevi * n];
        for (int i = 0; i < nevi; i++) {
            m_Z[i] = *m_Z + i * n;
            std::copy_n(vr.data() + index[i] * n, n_inside, m_Z[i]);
            for (int j = n_inside; j < n; j++) {

                m_Z[i][j] = 0;
            }
        }

        hpddm_op->setVectors(m_Z);

        // timing
        mytime = MPI_Wtime() - time;
        MPI_Barrier(hpddm_op->HA->get_comm());
        time = MPI_Wtime();
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());
        infos["DDM_geev_max"] = NbrToStr(maxtime);

        //
        build_E(m_Z);
    }

    void build_E(CoefficientPrecision *const *Z) {
        //
        int sizeWorld;
        MPI_Comm_size(hpddm_op->HA->get_comm(), &sizeWorld);

        // Timing
        std::vector<double> mytime(2), maxtime(2);
        double time = MPI_Wtime();

        // Allgather
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);
        MPI_Allgather(&nevi, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, hpddm_op->HA->get_comm());

        displs[0] = 0;
        for (int i = 1; i < sizeWorld; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        std::vector<CoefficientPrecision> E;
        build_coarse_space_outside(hpddm_op->HA, nevi, n, Z, E);
        size_E    = sqrt(E.size());
        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hpddm_op->HA->get_comm());
        time = MPI_Wtime();

        int rankWorld;
        MPI_Comm_rank(hpddm_op->HA->get_comm(), &rankWorld);
        if (rankWorld == 0) {
            Matrix<CoefficientPrecision> E_mat(size_E, size_E);
            E_mat.assign(size_E, size_E, E.data(), false);
            E_mat.print(std::cout, ",");
        }

        hpddm_op->buildTwo(MPI_COMM_WORLD, E.data());

        mytime[1] = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, hpddm_op->HA->get_comm());

        infos["DDM_setup_ZtAZ_max"] = NbrToStr(maxtime[0]);
        infos["DDM_facto_ZtAZ_max"] = NbrToStr(maxtime[1]);
        two_level                   = 1;
    }

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
            infos["GenEO_coarse_size"]          = "0";
            infos["Coarse_correction"]          = "None";
            infos["DDM_local_coarse_size_mean"] = "0";
            infos["DDM_local_coarse_size_max"]  = "0";
            infos["DDM_local_coarse_size_min"]  = "0";
        } else {
            infos["GenEO_coarse_size"] = NbrToStr(size_E);
            int nevi_mean              = nevi;
            int nevi_max               = nevi;
            int nevi_min               = nevi;

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
                infos["Coarse_correction"]          = "None";
                infos["GenEO_coarse_size"]          = "0";
                infos["DDM_local_coarse_size_mean"] = "0";
                infos["DDM_local_coarse_size_max"]  = "0";
                infos["DDM_local_coarse_size_min"]  = "0";
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

    std::string get_infos(const std::string &key) const {
        if (hpddm_op->HA->get_rankworld() == 0) {
            return infos[key];
        }
        return "";
    }

    int get_nevi() const {
        return nevi;
    }
    int get_local_size() const {
        return n;
    }
    const std::vector<int> &get_local_to_global_numbering() const {
        return renum_to_global;
    }
    // MPI_Comm get_comm() const {
    //     return comm;
    // }
};

} // namespace htool
#endif
