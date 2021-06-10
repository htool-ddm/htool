#ifndef HTOOL_WRAPPER_HPDDM_HPP
#define HTOOL_WRAPPER_HPDDM_HPP

#define HPDDM_NUMBERING 'F'
#define HPDDM_DENSE 1
#define HPDDM_FETI 0
#define HPDDM_BDD 0
#define LAPACKSUB
#define DLAPACK
#define EIGENSOLVER 1
// #include "../solvers/proto_ddm.hpp"
#include "../types/hmatrix_virtual.hpp"
#include "../types/matrix.hpp"
#include <HPDDM.hpp>

namespace htool {

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class Proto_DDM;

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class Proto_HPDDM : public HpDense<T, 'G'> {
  private:
    const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &HA;
    std::vector<T> *in_global;
    Proto_DDM<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &P;

  public:
    typedef HpDense<T, 'G'> super;

    Proto_HPDDM(const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &A, Proto_DDM<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &P0) : HA(A), P(P0) {
        in_global = new std::vector<T>(A.nb_cols());
        P.init_hpddm(*this);
    }
    ~Proto_HPDDM() { delete in_global; }

    int GMV(const T *const in, T *const out, const int &mu = 1) const {
        HA.mvprod_local(in, out, in_global->data(), 1);
        return 0;
    }

    template <bool = true>
    int apply(const T *const in, T *const out, const unsigned short &mu = 1, T * = nullptr, const unsigned short & = 0) const {
        P.apply(in, out);
        // std::copy_n(in, this->_n, out);
        return 0;
    }

    void build_coarse_space_geev(Matrix<T> &Mi, IMatrix<T> &generator_Bi, const std::vector<R3> &x) {
        // Coarse space
        P.build_coarse_space_geev(Mi, generator_Bi, x);
    }

    void build_coarse_space(Matrix<T> &Mi, IMatrix<T> &generator_Bi, const std::vector<R3> &x) {
        // Coarse space
        P.build_coarse_space(Mi, generator_Bi, x);
    }

    void build_coarse_space(Matrix<T> &Mi, const std::vector<R3> &x) {
        // Coarse space
        P.build_coarse_space(Mi, x);
    }

    void facto_one_level() {
        P.facto_one_level();
    }
    void solve(const T *const rhs, T *const x, const int &mu = 1) {
        //
        int rankWorld        = HA.get_rankworld();
        int sizeWorld        = HA.get_sizeworld();
        int offset           = HA.get_local_offset();
        int nb_cols          = HA.nb_cols();
        int nb_rows          = HA.nb_rows();
        double time          = MPI_Wtime();
        int n                = P.get_n();
        int n_inside         = P.get_n_inside();
        double time_vec_prod = StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"));
        int nb_vec_prod      = StrToNbr<int>(HA.get_infos("nb_mat_vec_prod"));
        P.timing_Q           = 0;
        P.timing_one_level   = 0;

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n, 0);
        std::vector<T> local_rhs(n, 0);

        // Permutation
        HA.source_to_cluster_permutation(rhs, rhs_perm.data());
        std::copy_n(rhs_perm.begin() + offset, n_inside, local_rhs.begin());

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*this, local_rhs.data(), x_local.data(), mu, HA.get_comm());

        // // Delete the overlap (useful only when mu>1 and n!=n_inside)
        // for (int i=0;i<mu;i++){
        //     std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        // }

        // Local to global
        // hpddm_op.HA.local_to_global(x_local.data(),hpddm_op.in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            recvcounts[i] = (HA.get_MasterOffset_t(i).second);
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(x_local.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_global->data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), HA.get_comm());

        // Permutation
        HA.cluster_to_target_permutation(in_global->data(), x);

        // Infos
        HPDDM::Option &opt = *HPDDM::Option::get();
        time               = MPI_Wtime() - time;
        P.set_infos("Solve", NbrToStr(time));
        P.set_infos("Nb_it", NbrToStr(nb_it));
        P.set_infos("Nb_subdomains", NbrToStr(sizeWorld));
        P.set_infos("nb_mat_vec_prod", NbrToStr(StrToNbr<int>(HA.get_infos("nb_mat_vec_prod")) - nb_vec_prod));
        P.set_infos("mean_time_mat_vec_prod", NbrToStr((StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod")) - time_vec_prod) / (StrToNbr<double>(HA.get_infos("nb_mat_vec_prod")) - nb_vec_prod)));
        switch (opt.val("schwarz_method", 0)) {
        case HPDDM_SCHWARZ_METHOD_NONE:
            P.set_infos("Precond", "None");
            break;
        case HPDDM_SCHWARZ_METHOD_RAS:
            P.set_infos("Precond", "RAS");
            break;
        case HPDDM_SCHWARZ_METHOD_ASM:
            P.set_infos("Precond", "ASM");
            break;
        case HPDDM_SCHWARZ_METHOD_OSM:
            P.set_infos("Precond", "OSM");
            break;
        case HPDDM_SCHWARZ_METHOD_ORAS:
            P.set_infos("Precond", "ORAS");
            break;
        case HPDDM_SCHWARZ_METHOD_SORAS:
            P.set_infos("Precond", "SORAS");
            break;
        }

        switch (opt.val("krylov_method", 8)) {
        case HPDDM_KRYLOV_METHOD_GMRES:
            P.set_infos("krylov_method", "gmres");
            break;
        case HPDDM_KRYLOV_METHOD_BGMRES:
            P.set_infos("krylov_method", "bgmres");
            break;
        case HPDDM_KRYLOV_METHOD_CG:
            P.set_infos("krylov_method", "cg");
            break;
        case HPDDM_KRYLOV_METHOD_BCG:
            P.set_infos("krylov_method", "bcg");
            break;
        case HPDDM_KRYLOV_METHOD_GCRODR:
            P.set_infos("krylov_method", "gcrodr");
            break;
        case HPDDM_KRYLOV_METHOD_BGCRODR:
            P.set_infos("krylov_method", "bgcrodr");
            break;
        case HPDDM_KRYLOV_METHOD_BFBCG:
            P.set_infos("krylov_method", "bfbcg");
            break;
        case HPDDM_KRYLOV_METHOD_RICHARDSON:
            P.set_infos("krylov_method", "richardson");
            break;
        case HPDDM_KRYLOV_METHOD_NONE:
            P.set_infos("krylov_method", "none");
            break;
        }

        //
        if (P.get_infos("Precond") == "None") {
            P.set_infos("GenEO_coarse_size", "0");
            P.set_infos("Coarse_correction", "None");
            P.set_infos("DDM_local_coarse_size_mean", "0");
            P.set_infos("DDM_local_coarse_size_max", "0");
            P.set_infos("DDM_local_coarse_size_min", "0");
        } else {
            P.set_infos("GenEO_coarse_size", NbrToStr(P.get_size_E()));
            int nevi     = P.get_nevi();
            int nevi_max = P.get_nevi();
            int nevi_min = P.get_nevi();

            if (rankWorld == 0) {
                MPI_Reduce(MPI_IN_PLACE, &(nevi), 1, MPI_INT, MPI_SUM, 0, HA.get_comm());
                MPI_Reduce(MPI_IN_PLACE, &(nevi_max), 1, MPI_INT, MPI_MAX, 0, HA.get_comm());
                MPI_Reduce(MPI_IN_PLACE, &(nevi_min), 1, MPI_INT, MPI_MIN, 0, HA.get_comm());
            } else {
                MPI_Reduce(&(nevi), &(nevi), 1, MPI_INT, MPI_SUM, 0, HA.get_comm());
                MPI_Reduce(&(nevi_max), &(nevi_max), 1, MPI_INT, MPI_MAX, 0, HA.get_comm());
                MPI_Reduce(&(nevi_min), &(nevi_min), 1, MPI_INT, MPI_MIN, 0, HA.get_comm());
            }
            P.set_infos("DDM_local_coarse_size_mean", NbrToStr((double)nevi / (double)sizeWorld));
            P.set_infos("DDM_local_coarse_size_max", NbrToStr(nevi_max));
            P.set_infos("DDM_local_coarse_size_min", NbrToStr(nevi_min));
            switch (opt.val("schwarz_coarse_correction", 42)) {
            case HPDDM_SCHWARZ_COARSE_CORRECTION_BALANCED:
                P.set_infos("Coarse_correction", "Balanced");
                break;
            case HPDDM_SCHWARZ_COARSE_CORRECTION_ADDITIVE:
                P.set_infos("Coarse_correction", "Additive");
                break;
            case HPDDM_SCHWARZ_COARSE_CORRECTION_DEFLATED:
                P.set_infos("Coarse_correction", "Deflated");
                break;
            default:
                P.set_infos("Coarse_correction", "None");
                P.set_infos("GenEO_coarse_size", "0");
                P.set_infos("DDM_local_coarse_size_mean", "0");
                P.set_infos("DDM_local_coarse_size_max", "0");
                P.set_infos("DDM_local_coarse_size_min", "0");
                break;
            }
        }
        P.set_infos("htool_solver", "protoddm");

        double timing_one_level = P.get_timing_one_level();
        double timing_Q         = P.get_timing_Q();
        double maxtiming_one_level, maxtiming_Q;

        // Timing
        MPI_Reduce(&(timing_one_level), &(maxtiming_one_level), 1, MPI_DOUBLE, MPI_MAX, 0, HA.get_comm());
        MPI_Reduce(&(timing_Q), &(maxtiming_Q), 1, MPI_DOUBLE, MPI_MAX, 0, HA.get_comm());

        P.set_infos("DDM_apply_one_level_max", NbrToStr(maxtiming_one_level));
        P.set_infos("DDM_apply_Q_max", NbrToStr(maxtiming_Q));
    }

    void print_infos() const {
        if (HA.get_rankworld() == 0) {
            for (std::map<std::string, std::string>::const_iterator it = P.get_infos().begin(); it != P.get_infos().end(); ++it) {
                std::cout << it->first << "\t" << it->second << std::endl;
            }
            std::cout << std::endl;
        }
    }
    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const {
        if (HA.get_rankworld() == 0) {
            std::ofstream outputfile(outputname, mode);
            if (outputfile) {
                for (std::map<std::string, std::string>::const_iterator it = P.get_infos().begin(); it != P.get_infos().end(); ++it) {
                    outputfile << it->first << sep << it->second << std::endl;
                }
                outputfile.close();
            } else {
                std::cout << "Unable to create " << outputname << std::endl;
            }
        }
    }

    void add_infos(std::string key, std::string value) const {
        if (HA.get_rankworld() == 0) {
            if (P.get_infos().find(key) == P.get_infos().end()) {
                P.set_infos(key, value);
            } else {
                P.set_infos(key, value);
            }
        }
    }

    std::string get_infos(const std::string &key) const { return P.get_infos(key); }

    int get_nevi() const { return P.get_nevi(); }
};

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class Calderon : public HPDDM::EmptyOperator<T> {
  private:
    const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &HA;
    const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &HB;
    Matrix<T> &M;
    std::vector<int> _ipiv;
    std::vector<T> *in_global, *buffer;
    mutable std::map<std::string, std::string> infos;

  public:
    Calderon(const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &A, const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &B, Matrix<T> &M0) : HPDDM::EmptyOperator<T>(A.get_local_size()), HA(A), HB(B), M(M0), _ipiv(M.nb_rows()) {
        in_global = new std::vector<T>;
        buffer    = new std::vector<T>;

        // LU facto
        int size = M.nb_rows();
        int lda  = M.nb_rows();
        int info;
        HPDDM::Lapack<Cplx>::getrf(&size, &size, M.data(), &lda, _ipiv.data(), &info);
    }

    ~Calderon() {
        delete in_global;
        in_global = nullptr;
        delete buffer;
        buffer = nullptr;
    }

    int GMV(const T *const in, T *const out, const int &mu = 1) const {
        int local_size = HA.get_local_size();

        // Tranpose without overlap
        if (mu != 1) {
            for (int i = 0; i < mu; i++) {
                for (int j = 0; j < local_size; j++) {
                    (*buffer)[i + j * mu] = in[i * this->getDof() + j];
                }
            }
        }

        // All gather
        if (mu == 1) { // C'est moche
            HA.mvprod_local(in, out, in_global->data(), mu);
        } else {
            HA.mvprod_local(buffer->data(), buffer->data() + local_size * mu, in_global->data(), mu);
        }

        // Tranpose
        if (mu != 1) {
            for (int i = 0; i < mu; i++) {
                for (int j = 0; j < local_size; j++) {
                    out[i * this->getDof() + j] = (*buffer)[i + j * mu + local_size * mu];
                }
            }
        }
        return 0;
    }

    template <bool = true>
    int apply(const T *const in, T *const out, const unsigned short &mu = 1, T * = nullptr, const unsigned short & = 0) const {
        int local_size = HB.get_local_size();
        int offset     = HB.get_local_offset();

        if (mu != 1) {
            std::cerr << "Calderon preconditioning not working for multiple rhs" << std::endl;
            exit(1);
        }
        // // Tranpose
        // if (mu!=1){
        //     for (int i=0;i<mu;i++){
        //         for (int j=0;j<local_size;j++){
        //             (*buffer)[i+j*mu]=in[i*this->getDof()+j];
        //         }
        //     }
        // }

        // M^-1
        HA.local_to_global(in, in_global->data(), mu);
        const char l = 'N';
        int n        = M.nb_rows();
        int lda      = M.nb_rows();
        int ldb      = M.nb_rows();
        int nrhs     = mu;
        int info;
        HPDDM::Lapack<T>::getrs(&l, &n, &nrhs, M.data(), &lda, _ipiv.data(), in_global->data(), &ldb, &info);

        // All gather
        // if (mu==1){// C'est moche
        HB.mvprod_local(in_global->data() + offset, out, in_global->data() + M.nb_rows(), mu);
        // }
        // else{
        //     HB.mvprod_local(buffer->data(),buffer->data()+local_size*mu,in_global->data(),mu);
        // }

        // M^-1
        HA.local_to_global(out, in_global->data(), mu);
        HPDDM::Lapack<T>::getrs(&l, &n, &nrhs, M.data(), &lda, _ipiv.data(), in_global->data(), &ldb, &info);
        std::copy_n(in_global->data() + offset, local_size, out);
        // // Tranpose
        // if (mu!=1){
        //     for (int i=0;i<mu;i++){
        //         for (int j=0;j<local_size;j++){
        //             out[i*this->getDof()+j]=(*buffer)[i+j*mu+local_size*mu];
        //         }
        //     }
        // }
    }

    void solve(const T *const rhs, T *const x, const int &mu = 1) {
        //
        int rankWorld        = HA.get_rankworld();
        int sizeWorld        = HA.get_sizeworld();
        int offset           = HA.get_local_offset();
        int nb_cols          = HA.nb_cols();
        int nb_rows          = HA.nb_rows();
        int n_local          = this->_n;
        double time          = MPI_Wtime();
        double time_vec_prod = StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod"));
        int nb_vec_prod      = StrToNbr<int>(HA.get_infos("nb_mat_vec_prod"));
        in_global->resize(nb_cols * 2 * mu);
        buffer->resize(n_local * (mu == 1 ? 1 : 2 * mu));

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n_local, 0);
        std::vector<T> local_rhs(n_local, 0);

        // Permutation
        HA.source_to_cluster_permutation(rhs, rhs_perm.data());
        std::copy_n(rhs_perm.begin() + offset, n_local, local_rhs.begin());

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*this, local_rhs.data(), x_local.data(), mu, HA.get_comm());

        // // Delete the overlap (useful only when mu>1 and n!=n_inside)
        // for (int i=0;i<mu;i++){
        //     std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        // }

        // Local to global
        // hpddm_op.HA.local_to_global(x_local.data(),hpddm_op.in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            recvcounts[i] = (HA.get_MasterOffset_t(i).second);
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(x_local.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_global->data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), HA.get_comm());

        // Permutation
        HA.cluster_to_target_permutation(in_global->data(), x);

        // Timing
        HPDDM::Option &opt              = *HPDDM::Option::get();
        time                            = MPI_Wtime() - time;
        infos["Solve"]                  = NbrToStr(time);
        infos["Nb_it"]                  = NbrToStr(nb_it);
        infos["nb_mat_vec_prod"]        = NbrToStr(StrToNbr<int>(HA.get_infos("nb_mat_vec_prod")) - nb_vec_prod);
        infos["mean_time_mat_vec_prod"] = NbrToStr((StrToNbr<double>(HA.get_infos("total_time_mat_vec_prod")) - time_vec_prod) / (StrToNbr<double>(HA.get_infos("nb_mat_vec_prod")) - nb_vec_prod));
    }

    void print_infos() const {
        if (HA.get_rankworld() == 0) {
            for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
                std::cout << it->first << "\t" << it->second << std::endl;
            }
            std::cout << std::endl;
        }
    }
    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const {
        if (HA.get_rankworld() == 0) {
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
        if (HA.get_rankworld() == 0) {
            if (infos.find(key) == infos.end()) {
                infos[key] = infos[key] + value;
            } else {
                infos[key] = infos[key] + value;
            }
        }
    }

    std::string get_infos(const std::string &key) const { return infos[key]; }
};

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class ContinuousOperator : public HPDDM::EmptyOperator<T> {
  private:
    const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &H;
    Matrix<T> &M;
    std::vector<int> _ipiv;
    std::vector<T> *in_global, *buffer;
    mutable std::map<std::string, std::string> infos;

  public:
    ContinuousOperator(const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &A, Matrix<T> &M0) : HPDDM::EmptyOperator<T>(A.get_local_size()), H(A), M(M0), _ipiv(M.nb_rows()) {
        in_global = new std::vector<T>;
        buffer    = new std::vector<T>;

        // LU facto
        int size = M.nb_rows();
        int lda  = M.nb_rows();
        int info;
        HPDDM::Lapack<Cplx>::getrf(&size, &size, M.data(), &lda, _ipiv.data(), &info);
    }

    ~ContinuousOperator() {
        delete in_global;
        in_global = nullptr;
        delete buffer;
        buffer = nullptr;
    }

    int GMV(const T *const in, T *const out, const int &mu = 1) const {
        int local_size = H.get_local_size();

        // Tranpose without overlap
        if (mu != 1) {
            for (int i = 0; i < mu; i++) {
                for (int j = 0; j < local_size; j++) {
                    (*buffer)[i + j * mu] = in[i * this->getDof() + j];
                }
            }
        }

        // All gather
        if (mu == 1) { // C'est moche
            H.mvprod_local(in, out, in_global->data(), mu);
        } else {
            H.mvprod_local(buffer->data(), buffer->data() + local_size * mu, in_global->data(), mu);
        }

        // Tranpose
        if (mu != 1) {
            for (int i = 0; i < mu; i++) {
                for (int j = 0; j < local_size; j++) {
                    out[i * this->getDof() + j] = (*buffer)[i + j * mu + local_size * mu];
                }
            }
        }
        return 0;
    }

    template <bool = true>
    int apply(const T *const in, T *const out, const unsigned short &mu = 1, T * = nullptr, const unsigned short & = 0) const {
        int local_size = H.get_local_size();
        int offset     = H.get_local_offset();
        // // Tranpose
        // if (mu!=1){
        //     for (int i=0;i<mu;i++){
        //         for (int j=0;j<local_size;j++){
        //             (*buffer)[i+j*mu]=in[i*this->getDof()+j];
        //         }
        //     }
        // }

        // M^-1
        H.local_to_global(in, in_global->data(), mu);
        const char l = 'N';
        int n        = M.nb_rows();
        int lda      = M.nb_rows();
        int ldb      = M.nb_rows();
        int nrhs     = mu;
        int info;
        HPDDM::Lapack<T>::getrs(&l, &n, &nrhs, M.data(), &lda, _ipiv.data(), in_global->data(), &ldb, &info);
        std::copy_n(in_global->data() + offset, local_size, out);

        // // All gather
        // if (mu==1){// C'est moche
        //     HB.mvprod_local(in_global->data()+offset,out,in_global->data()+M.nb_rows(),mu);
        // }
        // else{
        //     HB.mvprod_local(buffer->data(),buffer->data()+local_size*mu,in_global->data(),mu);
        // }

        // // M^-1
        // HA.local_to_global(out, in_global->data(),mu);
        // HPDDM::Lapack<T>::getrs(&l,&n,&nrhs,M.data(),&lda,_ipiv.data(),in_global->data()+M.nb_rows(),&ldb,&info);

        // // Tranpose
        // if (mu!=1){
        //     for (int i=0;i<mu;i++){
        //         for (int j=0;j<local_size;j++){
        //             out[i*this->getDof()+j]=(*buffer)[i+j*mu+local_size*mu];
        //         }
        //     }
        // }

        return 0;
    }

    void solve(const T *const rhs, T *const x, const int &mu = 1) {
        //
        int rankWorld        = H.get_rankworld();
        int sizeWorld        = H.get_sizeworld();
        int offset           = H.get_local_offset();
        int nb_cols          = H.nb_cols();
        int nb_rows          = H.nb_rows();
        int n_local          = this->_n;
        double time          = MPI_Wtime();
        double time_vec_prod = StrToNbr<double>(H.get_infos("total_time_mat_vec_prod"));
        int nb_vec_prod      = StrToNbr<int>(H.get_infos("nb_mat_vec_prod"));
        in_global->resize(nb_cols * 2 * mu);
        buffer->resize(n_local * (mu == 1 ? 1 : 2 * mu));

        //
        std::vector<T> rhs_perm(nb_cols);
        std::vector<T> x_local(n_local, 0);
        std::vector<T> local_rhs(n_local, 0);

        // Permutation
        H.source_to_cluster_permutation(rhs, rhs_perm.data());
        std::copy_n(rhs_perm.begin() + offset, n_local, local_rhs.begin());

        // Solve
        int nb_it = HPDDM::IterativeMethod::solve(*this, local_rhs.data(), x_local.data(), mu, H.get_comm());

        // // Delete the overlap (useful only when mu>1 and n!=n_inside)
        // for (int i=0;i<mu;i++){
        //     std::copy_n(x_local.data()+i*n,n_inside,local_rhs.data()+i*n_inside);
        // }

        // Local to global
        // hpddm_op.HA.local_to_global(x_local.data(),hpddm_op.in_global->data(),mu);
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            recvcounts[i] = (H.get_MasterOffset_t(i).second);
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(x_local.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_global->data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), H.get_comm());

        // Permutation
        H.cluster_to_target_permutation(in_global->data(), x);

        // Timing
        HPDDM::Option &opt              = *HPDDM::Option::get();
        time                            = MPI_Wtime() - time;
        infos["Solve"]                  = NbrToStr(time);
        infos["Nb_it"]                  = NbrToStr(nb_it);
        infos["nb_mat_vec_prod"]        = NbrToStr(StrToNbr<int>(H.get_infos("nb_mat_vec_prod")) - nb_vec_prod);
        infos["mean_time_mat_vec_prod"] = NbrToStr((StrToNbr<double>(H.get_infos("total_time_mat_vec_prod")) - time_vec_prod) / (StrToNbr<double>(H.get_infos("nb_mat_vec_prod")) - nb_vec_prod));
    }

    void print_infos() const {
        if (H.get_rankworld() == 0) {
            for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
                std::cout << it->first << "\t" << it->second << std::endl;
            }
            std::cout << std::endl;
        }
    }
    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const {
        if (H.get_rankworld() == 0) {
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
        if (H.get_rankworld() == 0) {
            if (infos.find(key) == infos.end()) {
                infos[key] = infos[key] + value;
            } else {
                infos[key] = infos[key] + value;
            }
        }
    }

    std::string get_infos(const std::string &key) const { return infos[key]; }
};

} // namespace htool

template <typename T>
struct HPDDM::hpddm_method_id<htool::HPDDMDense<T>> { static constexpr char value = HPDDM::hpddm_method_id<HpDense<T, 'G'>>::value; };
#endif
