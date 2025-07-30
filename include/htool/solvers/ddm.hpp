#ifndef HTOOL_DDM_HPP
#define HTOOL_DDM_HPP

#include "../distributed_operator/distributed_operator.hpp"
#include "../matrix/matrix.hpp"
#include "../misc/logger.hpp"
#include "../misc/misc.hpp"
#include "../wrappers/wrapper_hpddm.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "./interfaces/virtual_coarse_operator_builder.hpp"
#include "./interfaces/virtual_coarse_space_builder.hpp"
#include "./local_solvers/local_hmatrix_plus_overlap_solvers.hpp"
#include "./local_solvers/local_hmatrix_solvers.hpp"
#include "htool/hmatrix/hmatrix.hpp"
#include "htool/misc/user.hpp"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream> // for cout
#include <map>      // for map
#include <memory>
#include <mpi.h>
#include <string>
#include <utility> // for pair
#include <vector>

namespace htool {

template <typename CoefficientPrecision, template <class> class LocalSolver>
class DDM {
  private:
    std::function<int(MPI_Comm)> get_rankWorld = [](MPI_Comm comm) {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    return rankWorld; };

    int m_size_w_overlap;
    int n_inside;
    std::unique_ptr<HPDDMOperator<CoefficientPrecision, LocalSolver>> m_hpddm_op;
    std::vector<htool::underlying_type<CoefficientPrecision>> D;
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
        m_hpddm_op.reset();
    }

    DDM(int size_w_overlap, const DistributedOperator<CoefficientPrecision> &distributed_operator, std::unique_ptr<HPDDMOperator<CoefficientPrecision, LocalSolver>> hpddm_op) : m_size_w_overlap(size_w_overlap), n_inside(distributed_operator.get_target_partition().get_size_of_partition(get_rankWorld(distributed_operator.get_comm()))), m_hpddm_op(std::move(hpddm_op)), D(size_w_overlap), nevi(0), one_level(0), two_level(0) {
        fill(D.begin(), D.begin() + n_inside, htool::underlying_type<CoefficientPrecision>(1));
        fill(D.begin() + n_inside, D.end(), htool::underlying_type<CoefficientPrecision>(0));
        m_hpddm_op->HPDDMOperator<CoefficientPrecision, LocalSolver>::super::super::initialize(D.data());
    }

    void facto_one_level() {
        double time = MPI_Wtime();
        double mytime(0), maxtime(0);
        m_hpddm_op->callNumfact();
        mytime = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, m_hpddm_op->HA->get_comm());

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
        CoefficientPrecision **Z_ptr_ptr = nevi ? new CoefficientPrecision *[nevi] : nullptr;
        CoefficientPrecision *Z_ptr      = Z.release();
        for (int i = 0; i < nevi; i++) {
            Z_ptr_ptr[i] = Z_ptr + i * m_size_w_overlap;
        }
        m_hpddm_op->setVectors(Z_ptr_ptr);

        //
        // int rankWorld;
        // MPI_Comm_rank(hpddm_op->HA->get_comm(), &rankWorld);
        int sizeWorld;
        MPI_Comm_size(m_hpddm_op->HA->get_comm(), &sizeWorld);

        time                                         = MPI_Wtime();
        Matrix<CoefficientPrecision> coarse_operator = coarse_operator_builder.build_coarse_operator(Z.nb_rows(), Z.nb_cols(), Z_ptr_ptr);
        mytime[1]                                    = MPI_Wtime() - time;

        time = MPI_Wtime();
        m_hpddm_op->buildTwo(MPI_COMM_WORLD, coarse_operator.data());
        mytime[2] = MPI_Wtime() - time;

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), mytime.size(), MPI_DOUBLE, MPI_MAX, 0, m_hpddm_op->HA->get_comm());

        infos["DDM_geev_max"]       = NbrToStr(maxtime[0]);
        infos["DDM_setup_ZtAZ_max"] = NbrToStr(maxtime[1]);
        infos["DDM_facto_ZtAZ_max"] = NbrToStr(maxtime[2]);
        infos["GenEO_coarse_size"]  = NbrToStr(coarse_operator.nb_cols());
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
            m_hpddm_op->setType(HPDDMOperator<CoefficientPrecision, LocalSolver>::Prcndtnr::NO);
            break;
        case HPDDM_SCHWARZ_METHOD_RAS:
            m_hpddm_op->setType(HPDDMOperator<CoefficientPrecision, LocalSolver>::Prcndtnr::GE);
            break;
        case HPDDM_SCHWARZ_METHOD_ASM:
            m_hpddm_op->setType(HPDDMOperator<CoefficientPrecision, LocalSolver>::Prcndtnr::SY);
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
        MPI_Comm comm = m_hpddm_op->HA->get_comm();
        int rankWorld;
        int sizeWorld;
        MPI_Comm_rank(comm, &rankWorld);
        MPI_Comm_size(comm, &sizeWorld);
        int offset = m_hpddm_op->HA->get_target_partition().get_offset_of_partition(rankWorld);
        // int size        = hpddm_op->HA->get_local_size();
        int nb_rows = m_hpddm_op->HA->get_target_partition().get_global_size();
        // int nb_vec_prod = StrToNbr<int>(hpddm_op->HA->get_infos("nb_mat_vec_prod"));
        double time = MPI_Wtime();

        //
        std::vector<CoefficientPrecision> rhs_perm(nb_rows);
        std::vector<CoefficientPrecision> x_local(m_size_w_overlap * mu, 0);
        std::vector<CoefficientPrecision> local_rhs(m_size_w_overlap * mu, 0);
        m_hpddm_op->in_global->resize(nb_rows * (mu == 1 ? 1 : 2 * mu));
        m_hpddm_op->buffer->resize(n_inside * (mu == 1 ? 1 : 2 * mu));

        // TODO: blocking ?
        for (int i = 0; i < mu; i++) {
            // Permutation
            m_hpddm_op->HA->get_target_partition().global_to_partition_numbering(rhs + i * nb_rows, rhs_perm.data());
            // global_to_root_cluster(hpddm_op->HA->get_root_target_cluster(), rhs + i * nb_rows, rhs_perm.data());
            // hpddm_op->HA->target_to_cluster_permutation(rhs + i * nb_rows, rhs_perm.data());

            std::copy_n(rhs_perm.begin() + offset, n_inside, local_rhs.begin() + i * m_size_w_overlap);
        }

        // TODO: avoid com here
        // for (int i=0;i<n-n_inside;i++){
        //   local_rhs[i]=rhs_perm[]
        // }
        m_hpddm_op->scaledexchange(local_rhs.data(), mu);

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*m_hpddm_op, local_rhs.data(), x_local.data(), mu, comm);

        // Delete the overlap (useful only when mu>1 and n!=n_inside)
        for (int i = 0; i < mu; i++) {
            std::copy_n(x_local.data() + i * m_size_w_overlap, n_inside, local_rhs.data() + i * n_inside);
        }

        // Local to global
        // hpddm_op->HA->local_to_global(x_local.data(),hpddm_op->in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            // recvcounts[i] = (hpddm_op->HA->get_root_target_cluster().get_clusters_on_partition()[i]->get_size()) * mu;
            recvcounts[i] = (m_hpddm_op->HA->get_target_partition().get_size_of_partition(i)) * mu;
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(local_rhs.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), m_hpddm_op->in_global->data() + (mu == 1 ? 0 : mu * nb_rows), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

        //

        for (int i = 0; i < mu; i++) {
            if (mu != 1) {
                for (int j = 0; j < sizeWorld; j++) {
                    std::copy_n(m_hpddm_op->in_global->data() + mu * nb_rows + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, m_hpddm_op->in_global->data() + i * nb_rows + displs[j] / mu);
                }
            }

            // Permutation
            m_hpddm_op->HA->get_target_partition().partition_to_global_numbering(m_hpddm_op->in_global->data() + i * nb_rows, x + i * nb_rows);
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
                MPI_Reduce(MPI_IN_PLACE, &(nevi_mean), 1, MPI_INT, MPI_SUM, 0, m_hpddm_op->HA->get_comm());
                MPI_Reduce(MPI_IN_PLACE, &(nevi_max), 1, MPI_INT, MPI_MAX, 0, m_hpddm_op->HA->get_comm());
                MPI_Reduce(MPI_IN_PLACE, &(nevi_min), 1, MPI_INT, MPI_MIN, 0, m_hpddm_op->HA->get_comm());
            } else {
                MPI_Reduce(&(nevi_mean), &(nevi_mean), 1, MPI_INT, MPI_SUM, 0, m_hpddm_op->HA->get_comm());
                MPI_Reduce(&(nevi_max), &(nevi_max), 1, MPI_INT, MPI_MAX, 0, m_hpddm_op->HA->get_comm());
                MPI_Reduce(&(nevi_min), &(nevi_min), 1, MPI_INT, MPI_MIN, 0, m_hpddm_op->HA->get_comm());
            }

            infos["DDM_local_coarse_size_mean"] = NbrToStr(static_cast<double>(nevi_mean) / static_cast<double>(sizeWorld));
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
        MPI_Comm_rank(m_hpddm_op->HA->get_comm(), &rankWorld);
        if (rankWorld == 0) {
            for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
                std::cout << it->first << "\t" << it->second << std::endl;
            }
            std::cout << std::endl;
        }
    }

    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const {
        int rankWorld;
        MPI_Comm_rank(m_hpddm_op->HA->get_comm(), &rankWorld);
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
        if (get_rankWorld(m_hpddm_op->HA->get_comm()) == 0) {
            if (infos.find(key) == infos.end()) {
                infos[key] = value;
            } else {
                infos[key] = NbrToStr(StrToNbr<double>(infos[key]) + StrToNbr<double>(value));
            }
        }
    }

    void set_infos(std::string key, std::string value) const {
        if (get_rankWorld(m_hpddm_op->HA->get_comm()) == 0) {
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
        return m_size_w_overlap;
    }
};

template <typename CoefficientPrecision>
DDM<CoefficientPrecision, HPDDM::LapackTRSub> make_DDM_solver(const DistributedOperator<CoefficientPrecision> &distributed_operator, Matrix<CoefficientPrecision> &local_dense_matrix, char symmetry, char UPLO, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) {
    int rankWorld;
    MPI_Comm_rank(distributed_operator.get_comm(), &rankWorld);
    int n = local_dense_matrix.nb_rows();

    // Timing
    double mytime(0), maxtime(0);
    double time = MPI_Wtime();

    // Symmetry and storage
    bool sym = false;
    if (symmetry == 'S' || (symmetry == 'H' && is_complex<CoefficientPrecision>())) {
        sym = true;

        if (UPLO == 'U') {
            htool::Logger::get_instance().log(LogLevel::ERROR, "HPDDM takes lower symmetric/hermitian matrices or regular matrices"); // LCOV_EXCL_LINE
            // throw std::invalid_argument("[Htool error] HPDDM takes lower symmetric/hermitian matrices or regular matrices");                  // LCOV_EXCL_LINE
        }
        if (symmetry == 'S' && is_complex<CoefficientPrecision>()) {
            htool::Logger::get_instance().log(LogLevel::WARNING, "A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver"); // LCOV_EXCL_LINE
            // std::cout << "[Htool warning] A symmetric matrix with UPLO='L' has been given to DDM solver. It will be considered hermitian by the solver." << std::endl;
        }
    }

    std::unique_ptr<HPDDMOperator<CoefficientPrecision, HPDDM::LapackTRSub>> hpddm_op = std::make_unique<HPDDMOperator<CoefficientPrecision, HPDDM::LapackTRSub>>(&distributed_operator);
    hpddm_op->initialize(n, sym, local_dense_matrix.data(), neighbors, intersections);

    mytime = MPI_Wtime() - time;

    // Timing
    MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, distributed_operator.get_comm());

    DDM<CoefficientPrecision, HPDDM::LapackTRSub> ddm_solver(n, distributed_operator, std::move(hpddm_op));
    ddm_solver.set_infos("DDM_setup_one_level_max", NbrToStr(maxtime));

    return ddm_solver;
}

// template <typename CoefficientPrecision>
// DDM<CoefficientPrecision, HPDDMCustomLocalSolver> make_DDM_solver_w_custom_local_solver(const DistributedOperator<CoefficientPrecision> &distributed_operator, Matrix<CoefficientPrecision> &local_dense_matrix, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) {
//     int rankWorld;
//     MPI_Comm_rank(distributed_operator.get_comm(), &rankWorld);
//     int n = local_dense_matrix.nb_rows();

//     // Timing
//     double mytime, maxtime;
//     double time = MPI_Wtime();

//     // Symmetry and storage
//     bool sym = false;
//     if (distributed_operator.get_symmetry_type() == 'S' || (distributed_operator.get_symmetry_type() == 'H' && is_complex<CoefficientPrecision>())) {
//         sym = true;
//     }

//     std::unique_ptr<HPDDMOperator<CoefficientPrecision, HPDDMCustomLocalSolver>> hpddm_op = std::make_unique<HPDDMOperator<CoefficientPrecision, HPDDMCustomLocalSolver>>(&distributed_operator);
//     hpddm_op->initialize(n, sym, local_dense_matrix.data(), neighbors, intersections); // we should not give a local dense matrix

//     auto dense_local_solver = std::make_unique<LocalDenseSolver<CoefficientPrecision>>(local_dense_matrix);
//     HPDDMCustomLocalSolver<CoefficientPrecision> hpddm_custom_local_solver(dense_local_solver);
//     hpddm_op->set_local_solver();

//     mytime = MPI_Wtime() - time;

//     // Timing
//     MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, distributed_operator.get_comm());

//     DDM<CoefficientPrecision, HPDDMCustomLocalSolver> ddm_solver(n, distributed_operator, std::move(hpddm_op));
//     ddm_solver.set_infos("DDM_setup_one_level_max", NbrToStr(maxtime));

//     return ddm_solver;
// }

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
DDM<CoefficientPrecision, HPDDMCustomLocalSolver> make_DDM_solver_w_custom_local_solver(const DistributedOperator<CoefficientPrecision> &distributed_operator, HMatrix<CoefficientPrecision, CoordinatePrecision> &local_hmatrix, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections, bool use_permutation) {
    int n = local_hmatrix.get_target_cluster().get_size();

    // Timing
    double mytime(0), maxtime(0);
    double time = MPI_Wtime();

    // Symmetry and storage
    bool sym = false;
    if (local_hmatrix.get_symmetry() == 'S' || (local_hmatrix.get_symmetry() == 'H' && is_complex<CoefficientPrecision>())) {
        sym = true;
    }

    std::unique_ptr<HPDDMOperator<CoefficientPrecision, HPDDMCustomLocalSolver>> hpddm_op = std::make_unique<HPDDMOperator<CoefficientPrecision, HPDDMCustomLocalSolver>>(&distributed_operator);
    hpddm_op->initialize(n, sym, nullptr, neighbors, intersections); // we should not give a local dense matrix

    auto local_hmatrix_solver = std::make_unique<LocalHMatrixSolver<CoefficientPrecision, CoordinatePrecision>>(local_hmatrix, use_permutation);
    hpddm_op->getSolver().set_local_solver(std::move(local_hmatrix_solver));

    mytime = MPI_Wtime() - time;

    // Timing
    MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, distributed_operator.get_comm());

    DDM<CoefficientPrecision, HPDDMCustomLocalSolver> ddm_solver(n, distributed_operator, std::move(hpddm_op));
    ddm_solver.set_infos("DDM_setup_one_level_max", NbrToStr(maxtime));

    return ddm_solver;
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
DDM<CoefficientPrecision, HPDDMCustomLocalSolver> make_DDM_solver_w_custom_local_solver(const DistributedOperator<CoefficientPrecision> &distributed_operator, HMatrix<CoefficientPrecision, CoordinatePrecision> &local_hmatrix, Matrix<CoefficientPrecision> &B, Matrix<CoefficientPrecision> &C, Matrix<CoefficientPrecision> &D, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections) {
    int n = local_hmatrix.get_target_cluster().get_size() + D.nb_rows();

    // Timing
    double mytime(0), maxtime(0);
    double time = MPI_Wtime();

    // Symmetry and storage
    bool sym = false;
    if (local_hmatrix.get_symmetry() == 'S' || (local_hmatrix.get_symmetry() == 'H' && is_complex<CoefficientPrecision>())) {
        sym = true;
    }

    std::unique_ptr<HPDDMOperator<CoefficientPrecision, HPDDMCustomLocalSolver>> hpddm_op = std::make_unique<HPDDMOperator<CoefficientPrecision, HPDDMCustomLocalSolver>>(&distributed_operator);
    hpddm_op->initialize(n, sym, nullptr, neighbors, intersections); // we should not give a local dense matrix

    auto local_hmatrix_solver = std::make_unique<LocalHMatrixPlusOverlapSolver<CoefficientPrecision, CoordinatePrecision>>(local_hmatrix, B, C, D);
    hpddm_op->getSolver().set_local_solver(std::move(local_hmatrix_solver));

    mytime = MPI_Wtime() - time;

    // Timing
    MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, distributed_operator.get_comm());

    DDM<CoefficientPrecision, HPDDMCustomLocalSolver> ddm_solver(n, distributed_operator, std::move(hpddm_op));
    ddm_solver.set_infos("DDM_setup_one_level_max", NbrToStr(maxtime));

    return ddm_solver;
}

} // namespace htool
#endif
