#ifndef HTOOL_PROTO_DDM_HPP
#define HTOOL_PROTO_DDM_HPP

#include "../types/hmatrix.hpp"
#include "../types/matrix.hpp"
#include "../wrappers/wrapper_hpddm.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include <HPDDM.hpp>

namespace htool {

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class Proto_HPDDM;

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class Proto_DDM {
  private:
    int n;
    int n_inside;
    const std::vector<int> neighbors;
    std::vector<std::vector<int>> intersections;
    std::vector<T> vec_ovr;
    std::vector<T> mat_loc;
    std::vector<double> D;
    const MPI_Comm &comm;
    mutable std::map<std::string, std::string> infos;
    std::vector<std::vector<T>> snd, rcv;
    std::vector<int> _ipiv;
    std::vector<int> _ipiv_coarse;
    int nevi;
    std::vector<T> evi;
    std::vector<int> renum_to_global;
    Matrix<T> E;
    const HMatrix<T, LowRankMatrix, AdmissibleCondition> &hmat;
    const T *const *Z;
    std::vector<int> recvcounts;
    std::vector<int> displs;

    void synchronize(bool scaled) {

        // Partition de l'unité
        if (scaled)
            fill(vec_ovr.begin() + n_inside, vec_ovr.end(), 0);
        for (int i = 0; i < neighbors.size(); i++) {
            for (int j = 0; j < intersections[i].size(); j++) {
                snd[i][j] = vec_ovr[intersections[i][j]];
            }
        }

        // Communications
        std::vector<MPI_Request> rq(2 * neighbors.size());

        for (int i = 0; i < neighbors.size(); i++) {
            MPI_Isend(snd[i].data(), snd[i].size(), wrapper_mpi<T>::mpi_type(), neighbors[i], 0, comm, &(rq[i]));
            MPI_Irecv(rcv[i].data(), rcv[i].size(), wrapper_mpi<T>::mpi_type(), neighbors[i], MPI_ANY_TAG, comm, &(rq[i + neighbors.size()]));
        }
        MPI_Waitall(rq.size(), rq.data(), MPI_STATUSES_IGNORE);

        for (int i = 0; i < neighbors.size(); i++) {
            for (int j = 0; j < intersections[i].size(); j++) {
                vec_ovr[intersections[i][j]] += rcv[i][j];
            }
        }
    }

  public:
    double timing_one_level;
    double timing_Q;

    Proto_DDM(const IMatrix<T> &mat0, const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &hmat_0, const std::vector<int> &ovr_subdomain_to_global0, const std::vector<int> &cluster_to_ovr_subdomain0, const std::vector<int> &neighbors0, const std::vector<std::vector<int>> &intersections0) : n(ovr_subdomain_to_global0.size()), n_inside(cluster_to_ovr_subdomain0.size()), neighbors(neighbors0), vec_ovr(n), mat_loc(n * n), D(n), comm(hmat_0.get_comm()), _ipiv(n), hmat(hmat_0), timing_Q(0), timing_one_level(0), recvcounts(hmat_0.get_sizeworld()), displs(hmat_0.get_sizeworld()) {

        // Timing
        MPI_Barrier(hmat.get_comm());
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

        // Building Ai
        bool sym                                                               = false;
        const std::vector<LowRankMatrix<T, ClusterImpl> *> &MyDiagFarFieldMats = hmat_0.get_MyDiagFarFieldMats();
        const std::vector<SubMatrix<T> *> &MyDiagNearFieldMats                 = hmat_0.get_MyDiagNearFieldMats();

        // Internal dense blocks
        for (int i = 0; i < MyDiagNearFieldMats.size(); i++) {
            const SubMatrix<T> &submat = *(MyDiagNearFieldMats[i]);
            int local_nr               = submat.nb_rows();
            int local_nc               = submat.nb_cols();
            int offset_i               = submat.get_offset_i() - hmat_0.get_local_offset();
            int offset_j               = submat.get_offset_j() - hmat_0.get_local_offset();
            for (int i = 0; i < local_nc; i++) {
                std::copy_n(&(submat(0, i)), local_nr, &mat_loc[offset_i + (offset_j + i) * n]);
            }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n, n);
        for (int i = 0; i < MyDiagFarFieldMats.size(); i++) {
            const LowRankMatrix<T, ClusterImpl> &lmat = *(MyDiagFarFieldMats[i]);
            int local_nr                              = lmat.nb_rows();
            int local_nc                              = lmat.nb_cols();
            int offset_i                              = lmat.get_offset_i() - hmat_0.get_local_offset();
            int offset_j                              = lmat.get_offset_j() - hmat_0.get_local_offset();
            ;
            FarFielBlock.resize(local_nr, local_nc);
            lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
            for (int i = 0; i < local_nc; i++) {
                std::copy_n(&(FarFielBlock(0, i)), local_nr, &mat_loc[offset_i + (offset_j + i) * n]);
            }
        }

        // Overlap
        std::vector<T> horizontal_block(n - n_inside, n_inside), vertical_block(n, n - n_inside);
        horizontal_block = mat0.get_submatrix(std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end()), std::vector<int>(renum_to_global.begin(), renum_to_global.begin() + n_inside)).get_mat();
        vertical_block   = mat0.get_submatrix(renum_to_global, std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end())).get_mat();
        for (int j = 0; j < n_inside; j++) {
            std::copy_n(horizontal_block.begin() + j * (n - n_inside), n - n_inside, &mat_loc[n_inside + j * n]);
        }
        for (int j = n_inside; j < n; j++) {
            std::copy_n(vertical_block.begin() + (j - n_inside) * n, n, &mat_loc[j * n]);
        }

        // Timing
        mytime = MPI_Wtime() - time;
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, this->comm);

        infos["DDM_setup_one_level_max"] = NbrToStr(maxtime);
        // infos["DDM_facto_one_level_max" ]= NbrToStr(maxtime[1]);

        //

        snd.resize(neighbors.size());
        rcv.resize(neighbors.size());

        for (int i = 0; i < neighbors.size(); i++) {
            snd[i].resize(intersections[i].size());
            rcv[i].resize(intersections[i].size());
        }
    }

    void facto_one_level() {
        double time = MPI_Wtime();
        double mytime, maxtime;
        int lda = n;
        int info;

        HPDDM::Lapack<T>::getrf(&n, &n, mat_loc.data(), &lda, _ipiv.data(), &info);
        if (info != 0)
            std::cout << "Error in getrf from Lapack for mat_loc: info=" << info << std::endl;

        mytime = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());

        // Timing
        MPI_Reduce(&(mytime), &(maxtime), 1, MPI_DOUBLE, MPI_MAX, 0, this->comm);

        infos["DDM_facto_one_level_max"] = NbrToStr(maxtime);
    }

    void build_coarse_space_geev(Matrix<T> &Mi, IMatrix<T> &generator_Bi, const std::vector<R3> &x) {
        // Timing
        std::vector<double> mytime(2), maxtime(2);
        double time = MPI_Wtime();
        // Data
        int n_global  = hmat.nb_cols();
        int sizeWorld = hmat.get_sizeworld();
        int rankWorld = hmat.get_rankworld();
        int info;

        // Building Neumann matrix
        htool::HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> HBi(generator_Bi, hmat.get_cluster_tree_t().get_local_cluster_tree(), x, -1, MPI_COMM_SELF);

        Matrix<T> Bi(n, n);

        // Building Bi
        bool sym                                                                = false;
        const std::vector<LowRankMatrix<T, ClusterImpl> *> &MyLocalFarFieldMats = HBi.get_MyFarFieldMats();
        const std::vector<SubMatrix<T> *> &MyLocalNearFieldMats                 = HBi.get_MyNearFieldMats();
        // std::cout << MyLocalNearFieldMats.size()<<std::endl;
        // std::cout << MyLocalFarFieldMats.size()<<std::endl;

        // Internal dense blocks
        for (int i = 0; i < MyLocalNearFieldMats.size(); i++) {
            const SubMatrix<T> &submat = *(MyLocalNearFieldMats[i]);
            int local_nr               = submat.nb_rows();
            int local_nc               = submat.nb_cols();
            int offset_i               = submat.get_offset_i() - hmat.get_local_offset();
            int offset_j               = submat.get_offset_j() - hmat.get_local_offset();
            for (int i = 0; i < local_nc; i++) {
                std::copy_n(&(submat(0, i)), local_nr, Bi.data() + offset_i + (offset_j + i) * n);
            }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n, n);
        for (int i = 0; i < MyLocalFarFieldMats.size(); i++) {
            const LowRankMatrix<T, ClusterImpl> &lmat = *(MyLocalFarFieldMats[i]);
            int local_nr                              = lmat.nb_rows();
            int local_nc                              = lmat.nb_cols();
            int offset_i                              = lmat.get_offset_i() - hmat.get_local_offset();
            int offset_j                              = lmat.get_offset_j() - hmat.get_local_offset();
            ;
            FarFielBlock.resize(local_nr, local_nc);
            lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
            for (int i = 0; i < local_nc; i++) {
                std::copy_n(&(FarFielBlock(0, i)), local_nr, Bi.data() + offset_i + (offset_j + i) * n);
            }
        }

        // Overlap
        std::vector<T> horizontal_block(n - n_inside, n_inside), vertical_block(n, n - n_inside);
        horizontal_block = generator_Bi.get_submatrix(std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end()), std::vector<int>(renum_to_global.begin(), renum_to_global.begin() + n_inside)).get_mat();
        vertical_block   = generator_Bi.get_submatrix(renum_to_global, std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end())).get_mat();
        for (int j = 0; j < n_inside; j++) {
            std::copy_n(horizontal_block.begin() + j * (n - n_inside), n - n_inside, Bi.data() + n_inside + j * n);
        }
        for (int j = n_inside; j < n; j++) {
            std::copy_n(vertical_block.begin() + (j - n_inside) * n, n, Bi.data() + j * n);
        }

        // test
        // double error=0;
        // double norm_frob=0;
        // for (int i=0;i<n;i++){
        //     for (int j=0;j<n;j++){
        //         error+=abs(pow(Bi(i,j)-Bi_test[i+j*n],2));
        //         norm_frob+=abs(pow(Bi(i,j),2));
        //     }
        // }
        // std::cout << "COUCOU "<<normFrob(Bi-Bi_test)/normFrob(Bi)<<std::endl;

        // LU facto for mass matrix
        int lda = n;
        std::vector<int> _ipiv_mass(n);
        HPDDM::Lapack<Cplx>::getrf(&n, &n, Mi.data(), &lda, _ipiv_mass.data(), &info);
        if (info != 0)
            std::cout << "Error in getrf from Lapack for Mi: info=" << info << std::endl;

        // Partition of unity
        Matrix<T> DAiD(n, n);
        for (int i = 0; i < n_inside; i++) {
            std::copy_n(&(mat_loc[i * n]), n_inside, &(DAiD(0, i)));
        }

        // M^-1
        const char l = 'N';
        lda          = n;
        int ldb      = n;
        HPDDM::Lapack<Cplx>::getrs(&l, &n, &n, Mi.data(), &lda, _ipiv_mass.data(), DAiD.data(), &ldb, &info);
        if (info != 0)
            std::cout << "Error in getrs from Lapack for Mi: info=" << info << std::endl;
        HPDDM::Lapack<Cplx>::getrs(&l, &n, &n, Mi.data(), &lda, _ipiv_mass.data(), Bi.data(), &ldb, &info);
        if (info != 0)
            std::cout << "Error in getrs from Lapack for Bi: info=" << info << std::endl;

        // Build local eigenvalue problem
        Matrix<T> evp(n, n);
        Bi.mvprod(DAiD.data(), evp.data(), n);

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Local eigenvalue problem
        int ldvl = n, ldvr = n, lwork = -1;
        lda = n;
        std::vector<T> work(n);
        std::vector<double> rwork(2 * n);
        std::vector<T> w(n);
        std::vector<T> vl(n * n), vr(n * n);

        HPDDM::Lapack<T>::geev("N", "V", &n, evp.data(), &lda, w.data(), nullptr, vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<T>::geev("N", "V", &n, evp.data(), &lda, w.data(), nullptr, vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        std::vector<int> index(n, 0);
        if (info != 0)
            std::cout << "Error in geev from Lapack: info=" << info << std::endl;

        for (int i = 0; i != index.size(); i++) {
            index[i] = i;
        }
        std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
            return (std::abs(w[a]) > std::abs(w[b]));
        });
        HPDDM::Option &opt = *HPDDM::Option::get();
        nevi               = 0;
        double threshold   = opt.val("geneo_threshold", -1.0);
        if (threshold > 0.0) {
            while (std::abs(w[index[nevi]]) > threshold && nevi < index.size()) {
                nevi++;
            }

        } else {
            nevi = opt.val("geneo_nu", 2);
        }

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, this->comm);
        infos["DDM_setup_geev_max"] = NbrToStr(maxtime[0]);
        infos["DDM_geev_max"]       = NbrToStr(maxtime[1]);

        // build the coarse space
        build_ZtAZ(vr, index);
    }

    void build_coarse_space(Matrix<T> &Mi, IMatrix<T> &generator_Bi, const std::vector<R3> &x) {
        // Timing
        std::vector<double> mytime(2), maxtime(2);
        double time = MPI_Wtime();
        // Data
        int n_global  = hmat.nb_cols();
        int sizeWorld = hmat.get_sizeworld();
        int rankWorld = hmat.get_rankworld();
        int info;

        // Building Neumann matrix
        htool::HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> HBi(generator_Bi, hmat.get_cluster_tree_t().get_local_cluster_tree(), x, -1, MPI_COMM_SELF);

        Matrix<T> Bi(n, n);

        // Building Bi
        bool sym                                                                = false;
        const std::vector<LowRankMatrix<T, ClusterImpl> *> &MyLocalFarFieldMats = HBi.get_MyFarFieldMats();
        const std::vector<SubMatrix<T> *> &MyLocalNearFieldMats                 = HBi.get_MyNearFieldMats();

        // Internal dense blocks
        for (int i = 0; i < MyLocalNearFieldMats.size(); i++) {
            const SubMatrix<T> &submat = *(MyLocalNearFieldMats[i]);
            int local_nr               = submat.nb_rows();
            int local_nc               = submat.nb_cols();
            int offset_i               = submat.get_offset_i() - hmat.get_local_offset();
            int offset_j               = submat.get_offset_j() - hmat.get_local_offset();
            for (int i = 0; i < local_nc; i++) {
                std::copy_n(&(submat(0, i)), local_nr, Bi.data() + offset_i + (offset_j + i) * n);
            }
        }

        // Internal compressed block
        Matrix<T> FarFielBlock(n, n);
        for (int i = 0; i < MyLocalFarFieldMats.size(); i++) {
            const LowRankMatrix<T, ClusterImpl> &lmat = *(MyLocalFarFieldMats[i]);
            int local_nr                              = lmat.nb_rows();
            int local_nc                              = lmat.nb_cols();
            int offset_i                              = lmat.get_offset_i() - hmat.get_local_offset();
            int offset_j                              = lmat.get_offset_j() - hmat.get_local_offset();
            ;
            FarFielBlock.resize(local_nr, local_nc);
            lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
            for (int i = 0; i < local_nc; i++) {
                std::copy_n(&(FarFielBlock(0, i)), local_nr, Bi.data() + offset_i + (offset_j + i) * n);
            }
        }

        // Overlap
        std::vector<T> horizontal_block(n - n_inside, n_inside), vertical_block(n, n - n_inside);
        horizontal_block = generator_Bi.get_submatrix(std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end()), std::vector<int>(renum_to_global.begin(), renum_to_global.begin() + n_inside)).get_mat();
        vertical_block   = generator_Bi.get_submatrix(renum_to_global, std::vector<int>(renum_to_global.begin() + n_inside, renum_to_global.end())).get_mat();
        for (int j = 0; j < n_inside; j++) {
            std::copy_n(horizontal_block.begin() + j * (n - n_inside), n - n_inside, Bi.data() + n_inside + j * n);
        }
        for (int j = n_inside; j < n; j++) {
            std::copy_n(vertical_block.begin() + (j - n_inside) * n, n, Bi.data() + j * n);
        }

        // Partition of unity
        Matrix<T> DAiD(n, n);
        for (int i = 0; i < n_inside; i++) {
            std::copy_n(&(mat_loc[i * n]), n_inside, &(DAiD(0, i)));
        }

        // Build local eigenvalue problem
        Matrix<T> evp(n, n);
        evp = Mi + Bi;

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Local eigenvalue problem
        int ldvl = n, ldvr = n, lwork = -1;
        int lda = n, ldb = n;
        std::vector<T> alpha(n), beta(n);
        std::vector<T> work(n);
        std::vector<double> rwork(8 * n);
        std::vector<T> vl(n * n), vr(n * n);
        std::vector<int> index(n, 0);

        HPDDM::Lapack<T>::ggev("N", "V", &n, DAiD.data(), &lda, evp.data(), &ldb, alpha.data(), nullptr, beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<T>::ggev("N", "V", &n, DAiD.data(), &lda, evp.data(), &ldb, alpha.data(), nullptr, beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        if (info != 0)
            std::cout << "Error in ggev from Lapack: info=" << info << std::endl;

        for (int i = 0; i != index.size(); i++) {
            index[i] = i;
        }
        std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
            return ((std::abs(beta[a]) < 1e-15 || (std::abs(alpha[a] / beta[a]) > std::abs(alpha[b] / beta[b]))) && !(std::abs(beta[b]) < 1e-15));
        });

        HPDDM::Option &opt = *HPDDM::Option::get();
        nevi               = 0;
        double threshold   = opt.val("geneo_threshold", -1.0);
        if (threshold > 0.0) {
            while (std::abs(beta[index[nevi]]) < 1e-15 || (std::abs(alpha[index[nevi]] / beta[index[nevi]]) > threshold && nevi < index.size())) {
                nevi++;
            }

        } else {
            nevi = opt.val("geneo_nu", 2);
        }

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, this->comm);
        infos["DDM_setup_geev_max"] = NbrToStr(maxtime[0]);
        infos["DDM_geev_max"]       = NbrToStr(maxtime[1]);

        // build the coarse space
        build_ZtAZ(vr, index);
    }

    void build_coarse_space(Matrix<T> &Ki, const std::vector<R3> &x) {
        // Timing
        std::vector<double> mytime(2), maxtime(2);
        double time = MPI_Wtime();
        // Data
        int n_global  = hmat.nb_cols();
        int sizeWorld = hmat.get_sizeworld();
        int rankWorld = hmat.get_rankworld();
        int info;

        // Partition of unity
        Matrix<T> DAiD(n, n);
        for (int i = 0; i < n_inside; i++) {
            std::copy_n(&(mat_loc[i * n]), n_inside, &(DAiD(0, i)));
        }

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Local eigenvalue problem
        int ldvl = n, ldvr = n, lwork = -1;
        int lda = n, ldb = n;
        std::vector<T> alpha(n), beta(n);
        std::vector<T> work(n);
        std::vector<double> rwork(8 * n);
        std::vector<T> vl(n * n), vr(n * n);
        std::vector<int> index(n, 0);

        HPDDM::Lapack<T>::ggev("N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alpha.data(), nullptr, beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        HPDDM::Lapack<T>::ggev("N", "V", &n, DAiD.data(), &lda, Ki.data(), &ldb, alpha.data(), nullptr, beta.data(), vl.data(), &ldvl, vr.data(), &ldvr, work.data(), &lwork, rwork.data(), &info);
        if (info != 0)
            std::cout << "Error in ggev from Lapack: info=" << info << std::endl;

        for (int i = 0; i != index.size(); i++) {
            index[i] = i;
        }
        std::sort(index.begin(), index.end(), [&](const int &a, const int &b) {
            return ((std::abs(beta[a]) < 1e-15 || (std::abs(alpha[a] / beta[a]) > std::abs(alpha[b] / beta[b]))) && !(std::abs(beta[b]) < 1e-15));
        });

        HPDDM::Option &opt = *HPDDM::Option::get();
        nevi               = 0;
        double threshold   = opt.val("geneo_threshold", -1.0);
        if (threshold > 0.0) {
            while (std::abs(beta[index[nevi]]) < 1e-15 || (std::abs(alpha[index[nevi]] / beta[index[nevi]]) > threshold && nevi < index.size())) {
                nevi++;
            }

        } else {
            nevi = opt.val("geneo_nu", 2);
        }

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, this->comm);
        infos["DDM_setup_geev_max"] = NbrToStr(maxtime[0]);
        infos["DDM_geev_max"]       = NbrToStr(maxtime[1]);

        // Cleaning eigenvectors associated with kernel of Ki
        int count = 0;
        std::vector<int> nb_comp_conn;

        while (std::abs(beta[index[count]]) < 1e-15) {
            std::cout << std::setprecision(18);
            std::vector<T> values;
            std::vector<T> temp(n, 0);

            // Find values
            for (int i = 0; i < n; i++) {
                bool in = 0;
                for (int j = 0; j < values.size(); j++) {
                    if (std::abs(values[j] - vr[i + index[count] * n]) < 1e-6) {
                        in = 1;
                        break;
                    }
                }
                if (!in) {
                    values.push_back(vr[i + index[count] * n]);
                }
            }

            // Find connex component
            for (int i = 0; i < n; i++) {
                if (std::abs(values[count] - vr[i + index[count] * n]) < 1e-6) {
                    temp[i] = 1;
                }
            }
            std::copy_n(temp.data(), n, vr.data() + index[count] * n);
            nb_comp_conn.push_back(values.size());

            count++;
        }

        count = 0;
        std::vector<T> test_comp(n, 0);
        while (std::abs(beta[index[count]]) < 1e-15) {
            for (int i = 0; i < n; i++) {
                test_comp[i] = test_comp[i] + vr[i + index[count] * n];
            }
            count++;
        }
        if (std::abs(std::accumulate(test_comp.begin(), test_comp.end(), std::complex<double>(0)) - test_comp.size() * 1.0) > 1e-10 && std::abs(beta[index[0]]) < 1e-15) {
            std::cout << "WARNING: something wrong happened computing the eigenvectors in the kernel of the Neumann matrix" << std::endl;
        }

        // build the coarse space
        build_ZtAZ(vr, index);
    }
    // // Not working
    // void build_coarse_space_arpack( Matrix<T>& Ki, const std::vector<R3>& x ){
    //     // Timing
    //     std::vector<double> mytime(2), maxtime(2);
    //     double time = MPI_Wtime();
    //     // Data
    //     int n_global= hmat.nb_cols();
    //     int sizeWorld = hmat.get_sizeworld();
    //     int rankWorld = hmat.get_rankworld();
    //     int info;

    //     // Partition of unity
    //     Matrix<T> DAiD(n,n);
    //     for (int i =0 ;i < n_inside;i++){
    //         std::copy_n(&(mat_loc[i*n]),n_inside,&(DAiD(0,i)));
    //     }

    //     mytime[0] = MPI_Wtime() - time;
    //     MPI_Barrier(hmat.get_comm());
    //     time = MPI_Wtime();

    //     // Local eigenvalue problem
    //     HPDDM::Option& opt = *HPDDM::Option::get();
    //     int nu = opt.val("geneo_nu",2);
    //     int threshold = opt.val("geneo_threshold",2);
    //     bool sym= false;
    //     T** Z;
    //     HPDDM::MatrixCSR<T>* A = new HPDDM::MatrixCSR<T>(n, n, n * n, DAiD.data(), nullptr, nullptr, sym);
    //     HPDDM::MatrixCSR<T>* B = new HPDDM::MatrixCSR<T>(n, n, n * n, Ki.data(), nullptr, nullptr, sym);

    //     HPDDM::Arpack<T> eps(threshold, n, nu);
    //     std::cout <<"Warning"<<std::endl;
    //     eps.template solve<HPDDM::LapackTRSub>(A,B,Z,hmat.get_comm());

    //     int nevi = eps._nu;
    //     std::cout << "(rankWorld,nevi): "<<rankWorld<<" "<<nevi<<std::endl;

    //     mytime[1] = MPI_Wtime() - time;
    //     MPI_Barrier(hmat.get_comm());
    //     time = MPI_Wtime();

    //     // Timing
    //     MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0,this->comm);
    //     infos["DDM_setup_geev_max" ]= NbrToStr(maxtime[0]);
    //     infos["DDM_geev_max" ]= NbrToStr(maxtime[1]);

    //     // build the coarse space
    //     build_ZtAZ_arpack(Z,nevi);

    // }

    void build_ZtAZ(const std::vector<T> &vr, const std::vector<int> &index) {
        // Timing
        std::vector<double> mytime(2), maxtime(2);
        double time = MPI_Wtime();

        // Data
        int sizeWorld = hmat.get_sizeworld();
        int rankWorld = hmat.get_rankworld();
        int info      = 0;

        // Allgather
        MPI_Allgather(&nevi, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
        displs[0] = 0;

        for (int i = 1; i < sizeWorld; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        //
        int size_E   = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
        int nevi_max = *std::max_element(recvcounts.begin(), recvcounts.end());
        evi.resize(nevi * n, 0);

        for (int i = 0; i < nevi; i++) {
            // std::fill_n(evi.data()+i*n,n_inside,rankWorld+1);
            // std::copy_n(Z[i],n_inside,evi.data()+i*n);
            std::copy_n(vr.data() + index[i] * n, n_inside, evi.data() + i * n);
        }

        int local_max_size_j                                               = 0;
        const std::vector<LowRankMatrix<T, ClusterImpl> *> &MyFarFieldMats = hmat.get_MyFarFieldMats();
        const std::vector<SubMatrix<T> *> &MyNearFieldMats                 = hmat.get_MyNearFieldMats();
        for (int i = 0; i < MyFarFieldMats.size(); i++) {
            if (local_max_size_j < (*MyFarFieldMats[i]).nb_cols())
                local_max_size_j = (*MyFarFieldMats[i]).nb_cols();
        }
        for (int i = 0; i < MyNearFieldMats.size(); i++) {
            if (local_max_size_j < (*MyNearFieldMats[i]).nb_cols())
                local_max_size_j = (*MyNearFieldMats[i]).nb_cols();
        }

        std::vector<T> AZ(nevi_max * n_inside, 0);
        E.resize(size_E, size_E);

        for (int i = 0; i < sizeWorld; i++) {
            std::vector<T> buffer((hmat.get_MasterOffset_t(i).second + 2 * local_max_size_j) * recvcounts[i], 0);
            std::fill_n(AZ.data(), recvcounts[i] * n_inside, 0);

            if (rankWorld == i) {
                for (int j = 0; j < recvcounts[i]; j++) {
                    for (int k = 0; k < n_inside; k++) {
                        buffer[recvcounts[i] * (k + local_max_size_j) + j] = evi[j * n + k];
                    }
                }
            }
            MPI_Bcast(buffer.data() + local_max_size_j * recvcounts[i], hmat.get_MasterOffset_t(i).second * recvcounts[i], wrapper_mpi<T>::mpi_type(), i, comm);

            hmat.mvprod_subrhs(buffer.data(), AZ.data(), recvcounts[i], hmat.get_MasterOffset_t(i).first, hmat.get_MasterOffset_t(i).second, local_max_size_j);

            for (int j = 0; j < recvcounts[i]; j++) {
                for (int k = 0; k < n_inside; k++) {
                    vec_ovr[k] = AZ[j + recvcounts[i] * k];
                }
                // Parce que partition de l'unité...
                // synchronize(true);
                for (int jj = 0; jj < nevi; jj++) {
                    int coord_E_i           = displs[i] + j;
                    int coord_E_j           = displs[rankWorld] + jj;
                    E(coord_E_i, coord_E_j) = std::inner_product(evi.data() + jj * n, evi.data() + jj * n + n_inside, vec_ovr.data(), T(0), std::plus<T>(), [](T u, T v) { return u * std::conj(v); });
                }
            }
        }
        if (rankWorld == 0)
            MPI_Reduce(MPI_IN_PLACE, E.data(), size_E * size_E, wrapper_mpi<T>::mpi_type(), MPI_SUM, 0, comm);
        else
            MPI_Reduce(E.data(), E.data(), size_E * size_E, wrapper_mpi<T>::mpi_type(), MPI_SUM, 0, comm);

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        int n_coarse = size_E;
        _ipiv_coarse.resize(n_coarse);

        HPDDM::Lapack<T>::getrf(&n_coarse, &n_coarse, E.data(), &n_coarse, _ipiv_coarse.data(), &info);
        if (info != 0)
            std::cout << "Error in getrf from Lapack for E: info=" << info << std::endl;

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, this->comm);

        infos["DDM_setup_ZtAZ_max"] = NbrToStr(maxtime[0]);
        infos["DDM_facto_ZtAZ_max"] = NbrToStr(maxtime[1]);
    }

    void build_ZtAZ_arpack(T **Z, int nevi) {
        // Timing
        std::vector<double> mytime(2), maxtime(2);
        double time = MPI_Wtime();

        // Data
        int sizeWorld = hmat.get_sizeworld();
        int rankWorld = hmat.get_rankworld();
        int info      = 0;

        // Allgather
        MPI_Allgather(&nevi, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
        displs[0] = 0;

        for (int i = 1; i < sizeWorld; i++) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        //
        int size_E   = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
        int nevi_max = *std::max_element(recvcounts.begin(), recvcounts.end());
        evi.resize(nevi * n, 0);

        for (int i = 0; i < nevi; i++) {
            // std::fill_n(evi.data()+i*n,n_inside,rankWorld+1);
            // std::copy_n(Z[i],n_inside,evi.data()+i*n);
            std::copy_n(Z[i], n_inside, evi.data() + i * n);
        }

        int local_max_size_j                                               = 0;
        const std::vector<LowRankMatrix<T, ClusterImpl> *> &MyFarFieldMats = hmat.get_MyFarFieldMats();
        const std::vector<SubMatrix<T> *> &MyNearFieldMats                 = hmat.get_MyNearFieldMats();
        for (int i = 0; i < MyFarFieldMats.size(); i++) {
            if (local_max_size_j < (*MyFarFieldMats[i]).nb_cols())
                local_max_size_j = (*MyFarFieldMats[i]).nb_cols();
        }
        for (int i = 0; i < MyNearFieldMats.size(); i++) {
            if (local_max_size_j < (*MyNearFieldMats[i]).nb_cols())
                local_max_size_j = (*MyNearFieldMats[i]).nb_cols();
        }

        std::vector<T> AZ(nevi_max * n_inside, 0);
        E.resize(size_E, size_E);

        for (int i = 0; i < sizeWorld; i++) {
            std::vector<T> buffer((hmat.get_MasterOffset_t(i).second + 2 * local_max_size_j) * recvcounts[i], 0);
            std::fill_n(AZ.data(), recvcounts[i] * n_inside, 0);

            if (rankWorld == i) {
                for (int j = 0; j < recvcounts[i]; j++) {
                    for (int k = 0; k < n_inside; k++) {
                        buffer[recvcounts[i] * (k + local_max_size_j) + j] = evi[j * n + k];
                    }
                }
            }
            MPI_Bcast(buffer.data() + local_max_size_j * recvcounts[i], hmat.get_MasterOffset_t(i).second * recvcounts[i], wrapper_mpi<T>::mpi_type(), i, comm);

            hmat.mvprod_subrhs(buffer.data(), AZ.data(), recvcounts[i], hmat.get_MasterOffset_t(i).first, hmat.get_MasterOffset_t(i).second, local_max_size_j);

            for (int j = 0; j < recvcounts[i]; j++) {
                for (int k = 0; k < n_inside; k++) {
                    vec_ovr[k] = AZ[j + recvcounts[i] * k];
                }
                // Parce que partition de l'unité...
                // synchronize(true);
                for (int jj = 0; jj < nevi; jj++) {
                    int coord_E_i           = displs[i] + j;
                    int coord_E_j           = displs[rankWorld] + jj;
                    E(coord_E_i, coord_E_j) = std::inner_product(evi.data() + jj * n, evi.data() + jj * n + n_inside, vec_ovr.data(), T(0), std::plus<T>(), [](T u, T v) { return u * std::conj(v); });
                }
            }
        }
        if (rankWorld == 0)
            MPI_Reduce(MPI_IN_PLACE, E.data(), size_E * size_E, wrapper_mpi<T>::mpi_type(), MPI_SUM, 0, comm);
        else
            MPI_Reduce(E.data(), E.data(), size_E * size_E, wrapper_mpi<T>::mpi_type(), MPI_SUM, 0, comm);

        mytime[0] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        int n_coarse = size_E;
        _ipiv_coarse.resize(n_coarse);

        HPDDM::Lapack<T>::getrf(&n_coarse, &n_coarse, E.data(), &n_coarse, _ipiv_coarse.data(), &info);
        if (info != 0)
            std::cout << "Error in getrf from Lapack for E: info=" << info << std::endl;

        mytime[1] = MPI_Wtime() - time;
        MPI_Barrier(hmat.get_comm());
        time = MPI_Wtime();

        // Timing
        MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, this->comm);

        infos["DDM_setup_ZtAZ_max"] = NbrToStr(maxtime[0]);
        infos["DDM_facto_ZtAZ_max"] = NbrToStr(maxtime[1]);
    }

    void one_level(const T *const in, T *const out) {
        int sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);

        // Without overlap to with overlap
        std::copy_n(in, n_inside, vec_ovr.data());
        // std::cout << n<<" "<<n_inside <<std::endl;
        // std::fill(vec_ovr.begin(),vec_ovr.end(),0);
        // std::fill_n(vec_ovr.begin(),n_inside,1);
        synchronize(true);

        // Timing
        MPI_Barrier(hmat.get_comm());
        double time = MPI_Wtime();

        // std::cout << n_inside<<std::endl;
        // std::cout << n<<std::endl;
        const char l = 'N';
        int lda      = n;
        int ldb      = n;
        int nrhs     = 1;
        int info;
        // std::cout << n <<" "<<n_inside<<" "<<mat_loc.size()<<" "<<vec_ovr.size()<<std::endl;
        HPDDM::Lapack<T>::getrs(&l, &n, &nrhs, mat_loc.data(), &lda, _ipiv.data(), vec_ovr.data(), &ldb, &info);
        if (info != 0)
            std::cout << "Error in getrs from Lapack for mat_loc: info=" << info << std::endl;

        timing_one_level += MPI_Wtime() - time;

        // std::cout << info << std::endl;
        HPDDM::Option &opt = *HPDDM::Option::get();
        int schwarz_method = opt.val("schwarz_method", HPDDM_SCHWARZ_METHOD_RAS);
        if (schwarz_method == HPDDM_SCHWARZ_METHOD_RAS) {
            synchronize(true);
        } else if (schwarz_method == HPDDM_SCHWARZ_METHOD_ASM) {
            synchronize(false);
        }

        std::copy_n(vec_ovr.data(), n_inside, out);
    }

    void Q(const T *const in, T *const out) {
        std::copy_n(in, n_inside, vec_ovr.data());
        synchronize(true);
        std::vector<T> zti(nevi);
        int sizeWorld;
        int rankWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        // Timing
        MPI_Barrier(hmat.get_comm());
        double time = MPI_Wtime();

        for (int i = 0; i < nevi; i++) {
            zti[i] = std::inner_product(evi.begin() + i * n, evi.begin() + i * n + n, vec_ovr.begin(), T(0), std::plus<T>(), [](T u, T v) { return u * std::conj(v); });
            // zti[i]=std::inner_product(Z[i],Z[i+1],vec_ovr.begin(),T(0),std::plus<T>(), [](T u,T v){return u*std::conj(v);});
        }
        std::vector<T> zt(E.nb_cols(), 0);
        // if (rankWorld==0){
        //     // std::cout << zt.size() <<" " << zti.size() << std::endl;
        // }
        MPI_Gatherv(zti.data(), zti.size(), wrapper_mpi<T>::mpi_type(), zt.data(), recvcounts.data(), displs.data(), wrapper_mpi<T>::mpi_type(), 0, comm);

        if (rankWorld == 0) {
            const char l = 'N';
            int zt_size  = zt.size();
            int lda      = zt_size;
            int ldb      = zt_size;
            int nrhs     = 1;
            int info;
            // std::cout << n <<" "<<n_inside<<" "<<mat_loc.size()<<" "<<vec_ovr.size()<<std::endl;
            HPDDM::Lapack<T>::getrs(&l, &zt_size, &nrhs, E.data(), &lda, _ipiv_coarse.data(), zt.data(), &ldb, &info);
            if (info != 0)
                std::cout << "Error in getrs from Lapack for E: info=" << info << std::endl;
            // std::cout << "GETRS : "<<info << std::endl;
        }

        MPI_Scatterv(zt.data(), recvcounts.data(), displs.data(), wrapper_mpi<T>::mpi_type(), zti.data(), zti.size(), wrapper_mpi<T>::mpi_type(), 0, comm);

        std::fill_n(vec_ovr.data(), n, 0);
        for (int i = 0; i < nevi; i++) {

            std::transform(vec_ovr.begin(), vec_ovr.begin() + n, evi.begin() + n * i, vec_ovr.begin(), [&i, &zti](T u, T v) { return u + v * zti[i]; });
            // std::transform(vec_ovr.begin(),vec_ovr.begin()+n,Z[i],vec_ovr.begin(),[&i,&zti](T u, T v){return u+v*zti[i];});
        }

        timing_Q += MPI_Wtime() - time;

        synchronize(true);
        std::copy_n(vec_ovr.data(), n_inside, out);
    }

    void apply(const T *const in, T *const out) {
        std::vector<T> out_one_level(n_inside, 0);
        std::vector<T> out_Q(n_inside, 0);
        std::vector<T> buffer(hmat.nb_cols());
        std::vector<T> aq(n_inside);
        std::vector<T> p(n_inside);
        std::vector<T> am1p(n_inside);
        std::vector<T> qam1p(n_inside);
        std::vector<T> ptm1p(n_inside);

        HPDDM::Option &opt = *HPDDM::Option::get();
        int schwarz_method = opt.val("schwarz_method", HPDDM_SCHWARZ_METHOD_RAS);
        if (schwarz_method == HPDDM_SCHWARZ_METHOD_NONE) {
            std::copy_n(in, n_inside, out);
        } else {
            switch (opt.val("schwarz_coarse_correction", 42)) {
            case HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED:
                Q(in, out_Q.data());
                hmat.mvprod_local(out_Q.data(), aq.data(), buffer.data(), 1);
                std::transform(in, in + n_inside, aq.begin(), p.begin(), std::minus<T>());

                one_level(p.data(), out_one_level.data());
                hmat.mvprod_local(out_one_level.data(), am1p.data(), buffer.data(), 1);
                Q(am1p.data(), qam1p.data());

                std::transform(out_one_level.begin(), out_one_level.begin() + n_inside, qam1p.begin(), ptm1p.data(), std::minus<T>());
                std::transform(out_Q.begin(), out_Q.begin() + n_inside, ptm1p.begin(), out, std::plus<T>());
                break;
            case HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED:
                Q(in, out_Q.data());
                hmat.mvprod_local(out_Q.data(), aq.data(), buffer.data(), 1);
                std::transform(in, in + n_inside, aq.begin(), p.begin(), std::minus<T>());
                one_level(p.data(), out_one_level.data());
                std::transform(out_one_level.begin(), out_one_level.begin() + n_inside, out_Q.begin(), out, std::plus<T>());
                break;
            case HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE:
                Q(in, out_Q.data());
                one_level(in, out_one_level.data());
                std::transform(out_one_level.begin(), out_one_level.begin() + n_inside, out_Q.begin(), out, std::plus<T>());
                break;
            default:
                one_level(in, out);
                break;
            }
        }
        // ASM
        // one_level(in,out);
        // Q(in,out_Q.data());
        // std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,out_Q.begin(),out,std::plus<T>());

        // // ADEF1
        // Q(in,out_Q.data());
        // std::vector<T> aq(n_inside);
        // hmat.mvprod_local(out_Q.data(),aq.data(),buffer.data(),1);
        // std::vector<T> p(n_inside);
        //
        // std::transform(in, in+n_inside , aq.begin(),p.begin(),std::minus<T>());
        //
        // one_level(p.data(),out_one_level.data());
        //
        // std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,out_Q.begin(),out,std::plus<T>());

        // // BNN
        //
        // Q(in,out_Q.data());
        // hmat.mvprod_local(out_Q.data(),aq.data(),buffer.data(),1);
        // std::transform(in, in+n_inside , aq.begin(),p.begin(),std::minus<T>());
        //
        // one_level(p.data(),out_one_level.data());
        // hmat.mvprod_local(out_one_level.data(),am1p.data(),buffer.data(),1);
        // Q(am1p.data(),qam1p.data());
        //
        // std::transform(out_one_level.begin(),out_one_level.begin()+n_inside,qam1p.begin(),ptm1p.data(),std::minus<T>());
        // std::transform(out_Q.begin(),out_Q.begin()+n_inside,ptm1p.begin(),out,std::plus<T>());
    }

    void init_hpddm(Proto_HPDDM<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &hpddm_op) {
        bool sym = false;
        hpddm_op.initialize(n, sym, nullptr, neighbors, intersections);
    }
    int get_n() const { return n; }
    int get_n_inside() const { return n_inside; }
    int get_nevi() const { return nevi; }
    int get_size_E() const { return E.nb_cols(); } // E is not rectangular...
    std::map<std::string, std::string> &get_infos() const { return infos; }
    std::string get_infos(const std::string &key) const { return infos[key]; }
    void set_infos(const std::string &key, const std::string &value) const { infos[key] = value; }
    double get_timing_one_level() const { return timing_one_level; }
    double get_timing_Q() const { return timing_Q; }
};

} // namespace htool
#endif
