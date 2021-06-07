#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if _OPENMP
#    include <omp.h>
#endif

#include "../blocks/blocks.hpp"
#include "../clustering/virtual_cluster.hpp"
#include "../misc/misc.hpp"
#include "../types/hmatrix_virtual.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "matrix.hpp"
#include "point.hpp"
#include <cassert>
#include <fstream>
#include <map>
#include <memory>
#include <mpi.h>

namespace htool {

//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//
// Friend functions --- forward declaration
template <typename T, template <typename> class MultiLowRankMatrix, class AdmissibleCondition>
class MultiHMatrix;
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
class HMatrix;

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
double Frobenius_absolute_error(const HMatrix<T, LowRankMatrix, AdmissibleCondition> &B, const IMatrix<T> &A);

// Class
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
class HMatrix : public VirtualHMatrix<T> {

  private:
    // Data members
    int nr;
    int nc;
    int space_dim;
    int reqrank;
    int local_size;
    int local_offset;
    char symmetry;
    char UPLO;
    int false_positive;
    bool use_permutation;

    // Parameters
    int ndofperelt;
    double epsilon;
    double eta;
    int minclustersize;
    int maxblocksize;
    int minsourcedepth;
    int mintargetdepth;

    std::shared_ptr<VirtualCluster> cluster_tree_t;
    std::shared_ptr<VirtualCluster> cluster_tree_s;

    std::unique_ptr<Block<AdmissibleCondition>> BlockTree;

    std::vector<std::unique_ptr<LowRankMatrix<T>>> MyFarFieldMats;
    std::vector<std::unique_ptr<SubMatrix<T>>> MyNearFieldMats;
    std::vector<LowRankMatrix<T> *> MyDiagFarFieldMats;
    std::vector<SubMatrix<T> *> MyDiagNearFieldMats;
    std::vector<LowRankMatrix<T> *> MyStrictlyDiagFarFieldMats;
    std::vector<SubMatrix<T> *> MyStrictlyDiagNearFieldMats;

    std::vector<int> no_permutation_target, no_permutation_source;
    mutable std::map<std::string, std::string> infos;

    const MPI_Comm comm;
    int rankWorld, sizeWorld;

    // Internal methods
    void ComputeBlocks(IMatrix<T> &mat, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs);
    void ComputeSymBlocks(IMatrix<T> &mat, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs);
    bool UpdateBlocks(IMatrix<T> &mat, Block<AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &, std::vector<std::unique_ptr<LowRankMatrix<T>>> &, int &);
    bool UpdateSymBlocks(IMatrix<T> &mat, Block<AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &, std::vector<std::unique_ptr<LowRankMatrix<T>>> &, int &);
    void AddNearFieldMat(IMatrix<T> &mat, Block<AdmissibleCondition> &task, std::vector<std::unique_ptr<SubMatrix<T>>> &);
    void AddFarFieldMat(IMatrix<T> &mat, Block<AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<LowRankMatrix<T>>> &, const int &reqrank = -1);
    void ComputeInfos(const std::vector<double> &mytimes);

    // Check arguments
    void check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs) const;
    void check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<int> &tabs) const;
    void check_arguments_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt) const;
    void check_arguments_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt) const;

    // Friends
    template <typename U, template <typename> class MultiLowRankMatrix, class AdmissibleConditionU>
    friend class MultiHMatrix;

  public:
    // Special constructor for hand-made build (for MultiHMatrix for example)

    HMatrix(int space_dim0, int nr0, int nc0, const std::shared_ptr<VirtualCluster> &cluster_tree_t0, const std::shared_ptr<VirtualCluster> &cluster_tree_s0, char symmetry0 = 'N', char UPLO = 'N', const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(nr0), nc(nc0), space_dim(space_dim0), symmetry(symmetry0), use_permutation(true), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0), comm(comm0){};

    // Constructor
    HMatrix(const std::shared_ptr<VirtualCluster> &cluster_tree_t0, const std::shared_ptr<VirtualCluster> &cluster_tree_s0, double epsilon0 = 1e-6, double eta0 = 10, char Symmetry = 'N', char UPLO = 'N', const int &reqrank0 = -1, const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(0), nc(0), space_dim(cluster_tree_t0->get_space_dim()), reqrank(reqrank0), local_size(0), local_offset(0), symmetry(Symmetry), UPLO(UPLO), false_positive(0), use_permutation(true), ndofperelt(1), epsilon(epsilon0), eta(eta0), minclustersize(10), maxblocksize(1e6), minsourcedepth(0), mintargetdepth(0), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0), comm(comm0) {
        if (!((symmetry == 'N' || symmetry == 'H' || symmetry == 'S')
              && (UPLO == 'N' || UPLO == 'L' || UPLO == 'U')
              && ((symmetry == 'N' && UPLO == 'N') || (symmetry != 'N' && UPLO != 'N'))
              && ((symmetry == 'H' && is_complex<T>()) || symmetry != 'H'))) {
            throw std::invalid_argument("[Htool error] Invalid arguments to create HMatrix");
        }
    };

    // Build
    void build(IMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt, const double *const xs, const double *const rs, const int *const tabs, const double *const gs);

    void build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs) {
        if (this->space_dim != 3) {
            throw std::logic_error("[Htool error] Wrong space dimension");
        }
        std::vector<double> x_array_t(xt.size() * this->space_dim), x_array_s(xs.size() * this->space_dim);
        for (int p = 0; p < xt.size(); p++) {
            std::copy_n(xt[p].data(), space_dim, &(x_array_t[this->space_dim * p]));
        }
        for (int p = 0; p < xs.size(); p++) {
            std::copy_n(xs[p].data(), space_dim, &(x_array_s[this->space_dim * p]));
        }
        this->check_arguments(mat, xt, rt, tabt, gt, xs, rs, tabs, gs);
        this->build(mat, x_array_t.data(), rt.data(), tabt.data(), gt.data(), x_array_s.data(), rs.data(), tabs.data(), gs.data());
    }

    // Symmetry build
    void build_sym(IMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt);

    void build_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt) {
        if (this->space_dim != 3) {
            throw std::logic_error("[Htool error] Wrong space dimension");
        }
        std::vector<double> x_array_t(xt.size() * this->space_dim);
        for (int p = 0; p < xt.size(); p++) {
            std::copy_n(xt[p].data(), space_dim, &(x_array_t[this->space_dim * p]));
        }
        this->check_arguments_sym(mat, xt, rt, tabt, gt);
        this->build_sym(mat, x_array_t.data(), rt.data(), tabt.data(), gt.data());
    }

    // Build auto
    void build_auto(IMatrix<T> &mat, const double *const xt, const double *const xs);

    void build_auto(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<R3> &xs) {
        if (this->space_dim != 3) {
            throw std::logic_error("[Htool error] Wrong space dimension");
        }
        std::vector<double> x_array_t(xt.size() * this->space_dim), x_array_s(xs.size() * this->space_dim);
        for (int p = 0; p < xt.size(); p++) {
            std::copy_n(xt[p].data(), space_dim, &(x_array_t[this->space_dim * p]));
        }
        for (int p = 0; p < xs.size(); p++) {
            std::copy_n(xs[p].data(), space_dim, &(x_array_s[this->space_dim * p]));
        }
        this->build_auto(mat, x_array_t.data(), x_array_s.data());
    }

    // Symmetry auto build
    void build_auto_sym(IMatrix<T> &mat, const double *const xt);

    void build_auto_sym(IMatrix<T> &mat, const std::vector<R3> &xt) {
        if (this->space_dim != 3) {
            throw std::logic_error("[Htool error] Wrong space dimension");
        }
        std::vector<double> x_array_t(xt.size() * this->space_dim);
        for (int p = 0; p < xt.size(); p++) {
            std::copy_n(xt[p].data(), space_dim, &(x_array_t[this->space_dim * p]));
        }
        this->build_auto_sym(mat, x_array_t.data());
    }

    // Getters
    int nb_rows() const { return nr; }
    int nb_cols() const { return nc; }
    MPI_Comm get_comm() const { return comm; }
    int get_nlrmat() const {
        int res = MyFarFieldMats.size();
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm);
        return res;
    }
    int get_ndmat() const {
        int res = MyNearFieldMats.size();
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm);
        return res;
    }
    int get_rankworld() const { return rankWorld; }
    int get_sizeworld() const { return sizeWorld; }
    int get_local_size() const { return local_size; }
    int get_local_offset() const { return local_offset; }

    char get_symmetry_type() const { return symmetry; }
    char get_storage_type() const { return UPLO; }

    const VirtualCluster &get_cluster_tree_t() const { return *(cluster_tree_t.get()); }
    const VirtualCluster &get_cluster_tree_s() const { return *(cluster_tree_s.get()); }
    std::vector<std::pair<int, int>> get_MasterOffset_t() const { return cluster_tree_t->get_masteroffset(); }
    std::vector<std::pair<int, int>> get_MasterOffset_s() const { return cluster_tree_s->get_masteroffset(); }
    std::pair<int, int> get_MasterOffset_t(int i) const { return cluster_tree_t->get_masteroffset(i); }
    std::pair<int, int> get_MasterOffset_s(int i) const { return cluster_tree_s->get_masteroffset(i); }
    const std::vector<int> &get_permt() const { return cluster_tree_t->get_perm(); }
    const std::vector<int> &get_perms() const { return cluster_tree_s->get_perm(); }
    std::vector<int> get_local_perm_target() const { return cluster_tree_t->get_local_perm(); }
    std::vector<int> get_local_perm_source() const { return cluster_tree_s->get_local_perm(); }
    int get_permt(int i) const { return cluster_tree_t->get_perm(i); }
    int get_perms(int i) const { return cluster_tree_s->get_perm(i); }
    const std::vector<std::unique_ptr<SubMatrix<T>>> &get_MyNearFieldMats() const { return MyNearFieldMats; }
    const std::vector<std::unique_ptr<LowRankMatrix<T>>> &get_MyFarFieldMats() const { return MyFarFieldMats; }
    const std::vector<SubMatrix<T> *> &get_MyDiagNearFieldMats() const { return MyDiagNearFieldMats; }
    const std::vector<LowRankMatrix<T> *> &get_MyDiagFarFieldMats() const { return MyDiagFarFieldMats; }
    const std::vector<SubMatrix<T> *> &get_MyStrictlyDiagNearFieldMats() const { return MyStrictlyDiagNearFieldMats; }
    const std::vector<LowRankMatrix<T> *> &get_MyStrictlyDiagFarFieldMats() const { return MyStrictlyDiagFarFieldMats; }
    std::vector<T> get_local_diagonal(bool = true) const;
    void copy_local_diagonal(T *, bool = true) const;
    Matrix<T> get_local_diagonal_block(bool = true) const;
    void copy_local_diagonal_block(T *, bool = true) const;
    std::pair<int, int> get_max_size_blocks() const;

    double get_epsilon() const { return this->epsilon; };
    double get_eta() const { return this->eta; };
    int get_ndofperelt() const { return this->ndofperelt; };
    int get_minclustersize() const { return this->minclustersize; };
    int get_minsourcedepth() const { return this->minsourcedepth; };
    int get_mintargetdepth() const { return this->mintargetdepth; };
    int get_maxblocksize() const { return this->maxblocksize; };
    void set_epsilon(double epsilon0) { this->epsilon = epsilon0; };
    void set_eta(double eta0) { this->eta = eta0; };
    void set_ndofperelt(unsigned int ndofperelt0) { this->ndofperelt = ndofperelt0; };
    void set_minsourcedepth(unsigned int minsourcedepth0) { this->minsourcedepth = minsourcedepth0; };
    void set_mintargetdepth(unsigned int mintargetdepth0) { this->mintargetdepth = mintargetdepth0; };
    void set_maxblocksize(unsigned int maxblocksize0) { this->maxblocksize = maxblocksize0; };
    void set_use_permutation(bool choice) { this->use_permutation = choice; };

    // Infos
    const std::map<std::string, std::string> &get_infos() const { return infos; }
    std::string get_infos(const std::string &key) const { return infos[key]; }
    void add_info(const std::string &keyname, const std::string &value) const { infos[keyname] = value; }
    void print_infos() const;
    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const;
    void save_plot(const std::string &outputname) const;
    double compression() const; // 1- !!!
    friend double Frobenius_absolute_error<T, LowRankMatrix>(const HMatrix<T, LowRankMatrix, AdmissibleCondition> &B, const IMatrix<T> &A);

    // Mat vec prod
    void mvprod_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    void mvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    void mymvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    void mymvprod_global_to_local(const T *const in, T *const out, const int &mu = 1) const;

    void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const;
    std::vector<T> operator*(const std::vector<T> &x) const;
    Matrix<T> operator*(const Matrix<T> &x) const;

    // Permutations
    // template <typename U>
    // void source_to_cluster_permutation(const U *const in, U *const out) const;
    // template <typename U>
    // void cluster_to_target_permutation(const U *const in, U *const out) const;
    void source_to_cluster_permutation(const T *const in, T *const out) const;
    void cluster_to_target_permutation(const T *const in, T *const out) const;
    void local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    void local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;

    // local to global
    void local_to_global_source(const T *const in, T *const out, const int &mu) const;
    void local_to_global_target(const T *const in, T *const out, const int &mu) const;

    // Convert
    Matrix<T> get_local_dense() const;
    Matrix<T> get_local_dense_perm() const;
    void copy_local_dense_perm(T *) const;

    // Apply Dirichlet condition
    void apply_dirichlet(const std::vector<int> &boundary);
};

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs) const {
    if (!(mat.nb_rows() == tabt.size() && mat.nb_cols() == tabs.size()
          && mat.nb_rows() == ndofperelt * xt.size() && mat.nb_cols() == ndofperelt * xs.size()
          && mat.nb_rows() == ndofperelt * rt.size() && mat.nb_cols() == ndofperelt * rs.size()
          && mat.nb_rows() == ndofperelt * gt.size() && mat.nb_cols() == ndofperelt * gs.size())) {
        throw std::invalid_argument("[Htool error] Invalid size in arguments for building HMatrix");
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<int> &tabs) const {
    if (!(mat.nb_rows() == tabt.size() && mat.nb_cols() == tabs.size()
          && mat.nb_rows() == ndofperelt * xt.size() && mat.nb_cols() == ndofperelt * xs.size())) {
        throw std::invalid_argument("[Htool error] Invalid size in arguments for building HMatrix");
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::check_arguments_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt) const {
    this->check_arguments(mat, xt, rt, tabt, gt, xt, rt, tabt, gt);
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::check_arguments_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt) const {
    this->check_arguments(mat, xt, tabt, xt, tabt);
}

// build
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::build(IMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt, const double *const xs, const double *const rs, const int *const tabs, const double *const gs) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(3), maxtime(3), meantime(3);

    this->nc = mat.nb_cols();
    this->nr = mat.nb_rows();

    // Use no_permutation if needed
    if (use_permutation == false) {
        no_permutation_target.resize(nr);
        no_permutation_source.resize(nc);
        std::iota(no_permutation_target.begin(), no_permutation_target.end(), int(0));
        std::iota(no_permutation_source.begin(), no_permutation_source.end(), int(0));
    }

    // Construction arbre des paquets
    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    // Construction arbre des blocs
    double time = MPI_Wtime();
    this->BlockTree.reset(new Block<AdmissibleCondition>(*cluster_tree_t, *cluster_tree_s));
    this->BlockTree->set_mintargetdepth(this->mintargetdepth);
    this->BlockTree->set_minsourcedepth(this->minsourcedepth);
    this->BlockTree->set_maxblocksize(this->maxblocksize);
    this->BlockTree->set_eta(this->eta);
    bool force_sym = false;
    this->BlockTree->build(UPLO, force_sym, comm);
    mytimes[0] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, tabt, xs, tabs);
    mytimes[1] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

// Symmetry build
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::build_sym(IMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(3), maxtime(3), meantime(3);

    this->nc = mat.nb_cols();
    this->nr = mat.nb_rows();

    // Use no_permutation if needed
    if (use_permutation == false) {
        no_permutation_target.resize(nr);
        no_permutation_source.resize(nc);
        std::iota(no_permutation_target.begin(), no_permutation_target.end(), int(0));
        std::iota(no_permutation_source.begin(), no_permutation_source.end(), int(0));
    }

    // Construction arbre des paquets
    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    // Construction arbre des blocs
    double time = MPI_Wtime();

    this->BlockTree.reset(new Block<AdmissibleCondition>(*cluster_tree_t, *cluster_tree_s));
    this->BlockTree->set_mintargetdepth(this->mintargetdepth);
    this->BlockTree->set_minsourcedepth(this->minsourcedepth);
    this->BlockTree->set_maxblocksize(this->maxblocksize);
    this->BlockTree->set_eta(this->eta);
    bool force_sym = true;
    this->BlockTree->build(UPLO, force_sym, comm);

    mytimes[0] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, tabt, xt, tabt);
    mytimes[1] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

// Build auto
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::build_auto(IMatrix<T> &mat, const double *const xt, const double *const xs) {
    std::vector<int> tabt(this->ndofperelt * mat.nb_rows()), tabs(this->ndofperelt * mat.nb_cols());
    std::iota(tabt.begin(), tabt.end(), int(0));
    std::iota(tabs.begin(), tabs.end(), int(0));
    this->build(mat, xt, std::vector<double>(mat.nb_rows(), 0).data(), tabt.data(), std::vector<double>(mat.nb_rows(), 1).data(), xs, std::vector<double>(mat.nb_cols(), 0).data(), tabs.data(), std::vector<double>(mat.nb_cols(), 1).data());
}

// Symmetry auto build
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::build_auto_sym(IMatrix<T> &mat, const double *const xt) {
    std::vector<int> tabt(this->ndofperelt * mat.nb_rows());
    std::iota(tabt.begin(), tabt.end(), int(0));
    this->build_sym(mat, xt, std::vector<double>(mat.nb_rows(), 0).data(), tabt.data(), std::vector<double>(mat.nb_rows(), 1).data());
}

// Compute blocks recursively
// TODO: recursivity -> stack for compute blocks
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::ComputeBlocks(IMatrix<T> &mat, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs) {
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp parallel
#endif
    {
        std::vector<std::unique_ptr<SubMatrix<T>>> MyNearFieldMats_local;
        std::vector<std::unique_ptr<LowRankMatrix<T>>> MyFarFieldMats_local;
        std::vector<Block<AdmissibleCondition> *> local_tasks = BlockTree->get_local_tasks();

        int false_positive_local = 0;
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp for schedule(guided)
#endif
        for (int p = 0; p < local_tasks.size(); p++) {
            bool not_pushed;
            if (symmetry == 'H' || symmetry == 'S') {
                not_pushed = UpdateSymBlocks(mat, *(local_tasks[p]), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            } else {
                not_pushed = UpdateBlocks(mat, *(local_tasks[p]), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if (not_pushed) {
                AddNearFieldMat(mat, *(local_tasks[p]), MyNearFieldMats_local);
            }
        }
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp critical
#endif
        {
            MyFarFieldMats.insert(MyFarFieldMats.end(), std::make_move_iterator(MyFarFieldMats_local.begin()), std::make_move_iterator(MyFarFieldMats_local.end()));
            MyNearFieldMats.insert(MyNearFieldMats.end(), std::make_move_iterator(MyNearFieldMats_local.begin()), std::make_move_iterator(MyNearFieldMats_local.end()));
            false_positive += false_positive_local;
        }
    }

    // Build vectors of pointers for diagonal blocks
    for (int i = 0; i < MyFarFieldMats.size(); i++) {
        if (local_offset <= MyFarFieldMats[i]->get_offset_j() && MyFarFieldMats[i]->get_offset_j() < local_offset + local_size) {
            MyDiagFarFieldMats.push_back(MyFarFieldMats[i].get());
            if (MyFarFieldMats[i]->get_offset_j() == MyFarFieldMats[i]->get_offset_i())
                MyStrictlyDiagFarFieldMats.push_back(MyFarFieldMats[i].get());
        }
    }
    for (int i = 0; i < MyNearFieldMats.size(); i++) {
        if (local_offset <= MyNearFieldMats[i]->get_offset_j() && MyNearFieldMats[i]->get_offset_j() < local_offset + local_size) {
            MyDiagNearFieldMats.push_back(MyNearFieldMats[i].get());
            if (MyNearFieldMats[i]->get_offset_j() == MyNearFieldMats[i]->get_offset_i())
                MyStrictlyDiagNearFieldMats.push_back(MyNearFieldMats[i].get());
        }
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
bool HMatrix<T, LowRankMatrix, AdmissibleCondition>::UpdateBlocks(IMatrix<T> &mat, Block<AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local, std::vector<std::unique_ptr<LowRankMatrix<T>>> &MyFarFieldMats_local, int &false_positive_local) {
    if (task.IsAdmissible()) {
        AddFarFieldMat(mat, task, xt, tabt, xs, tabs, MyFarFieldMats_local, reqrank);
        if (MyFarFieldMats_local.back()->rank_of() != -1) {
            return false;
        } else {
            MyFarFieldMats_local.pop_back();
            false_positive_local += 1;
            // AddNearFieldMat(mat, task, MyNearFieldMats_local);
            // return false;
        }
    } else {
        AddNearFieldMat(mat, task, MyNearFieldMats_local);
        return false;
    }

    int bsize               = task.get_size();
    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (s.IsLeaf()) {
        if (t.IsLeaf()) {
            return true;
        } else {

            std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
            for (int p = 0; p < t.get_nb_sons(); p++) {
                task.build_son(t.get_son(p), s);

                Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                    }
                }
                return false;
            }
        }
    } else {
        if (t.IsLeaf()) {
            std::vector<bool> Blocks_not_pushed(s.get_nb_sons());
            for (int p = 0; p < s.get_nb_sons(); p++) {
                task.build_son(t, s.get_son(p));
                Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                    }
                }
                return false;
            }
        } else {
            if (t.get_size() > s.get_size()) {
                std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    task.build_son(t.get_son(p), s);
                    Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                    task.clear_sons();
                    return true;
                } else {
                    for (int p = 0; p < t.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                        }
                    }
                    return false;
                }
            } else {
                std::vector<bool> Blocks_not_pushed(s.get_nb_sons());
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    task.build_son(t, s.get_son(p));
                    Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                    task.clear_sons();
                    return true;
                } else {
                    for (int p = 0; p < s.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                        }
                    }
                    return false;
                }
            }
        }
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
bool HMatrix<T, LowRankMatrix, AdmissibleCondition>::UpdateSymBlocks(IMatrix<T> &mat, Block<AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local, std::vector<std::unique_ptr<LowRankMatrix<T>>> &MyFarFieldMats_local, int &false_positive_local) {

    if (task.IsAdmissible()) {

        AddFarFieldMat(mat, task, xt, tabt, xs, tabs, MyFarFieldMats_local, reqrank);
        if (MyFarFieldMats_local.back()->rank_of() != -1) {
            return false;
        } else {
            MyFarFieldMats_local.pop_back();
            false_positive_local += 1;
            // AddNearFieldMat(mat, task, MyNearFieldMats_local);
            // return false;
        }
    } else {
        AddNearFieldMat(mat, task, MyNearFieldMats_local);
        return false;
    }
    int bsize               = task.get_size();
    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (s.IsLeaf()) {
        if (t.IsLeaf()) {
            return true;
        } else {
            std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
            for (int p = 0; p < t.get_nb_sons(); p++) {
                task.build_son(t.get_son(p), s);
                Blocks_not_pushed[p] = UpdateSymBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                    }
                }
                return false;
            }
        }
    } else {
        if (t.IsLeaf()) {
            std::vector<bool> Blocks_not_pushed(s.get_nb_sons());
            for (int p = 0; p < s.get_nb_sons(); p++) {
                task.build_son(t, s.get_son(p));
                Blocks_not_pushed[p] = UpdateSymBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                    }
                }
                return false;
            }
        } else {
            std::vector<bool> Blocks_not_pushed(t.get_nb_sons() * s.get_nb_sons());
            for (int l = 0; l < s.get_nb_sons(); l++) {
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    task.build_son(t.get_son(p), s.get_son(l));
                    Blocks_not_pushed[p + l * t.get_nb_sons()] = UpdateSymBlocks(mat, task.get_son(p + l * t.get_nb_sons()), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }
            }
            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyNearFieldMats_local);
                    }
                }
                return false;
            }
        }
    }
}

// Build a dense block
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::AddNearFieldMat(IMatrix<T> &mat, Block<AdmissibleCondition> &task, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local) {

    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (use_permutation) {
        MyNearFieldMats_local.emplace_back(new SubMatrix<T>(mat, t.get_size(), s.get_size(), cluster_tree_t->get_perm().data() + t.get_offset(), cluster_tree_s->get_perm().data() + s.get_offset(), t.get_offset(), s.get_offset()));
    } else {
        MyNearFieldMats_local.emplace_back(new SubMatrix<T>(mat, t.get_size(), s.get_size(), no_permutation_target.data() + t.get_offset(), no_permutation_source.data() + s.get_offset(), t.get_offset(), s.get_offset()));
    }
}

// Build a low rank block
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::AddFarFieldMat(IMatrix<T> &mat, Block<AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<LowRankMatrix<T>>> &MyFarFieldMats_local, const int &reqrank) {

    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (use_permutation) {
        MyFarFieldMats_local.emplace_back(new LowRankMatrix<T>(std::vector<int>(cluster_tree_t->get_perm_start() + t.get_offset(), cluster_tree_t->get_perm_start() + t.get_offset() + t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start() + s.get_offset(), cluster_tree_s->get_perm_start() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset(), reqrank, this->epsilon));
    }

    else {
        MyFarFieldMats_local.emplace_back(new LowRankMatrix<T>(std::vector<int>(no_permutation_target.data() + t.get_offset(), no_permutation_target.data() + t.get_offset() + t.get_size()), std::vector<int>(no_permutation_source.data() + s.get_offset(), no_permutation_source.data() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset(), reqrank, this->epsilon));
    }
    MyFarFieldMats_local.back()->set_ndofperelt(this->ndofperelt);
    MyFarFieldMats_local.back()->build(mat, t, xt, tabt, s, xs, tabs);
}

// Compute infos
template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::ComputeInfos(const std::vector<double> &mytime) {
    // 0 : block tree ; 1 : compute blocks ;
    std::vector<double> maxtime(2), meantime(2);
    // 0 : dense mat ; 1 : lr mat ; 2 : rank ; 3 : local_size
    std::vector<int> maxinfos(4, 0), mininfos(4, std::max(nc, nr));
    std::vector<double> meaninfos(4, 0);
    // Infos
    for (int i = 0; i < MyNearFieldMats.size(); i++) {
        int size    = MyNearFieldMats[i]->nb_rows() * MyNearFieldMats[i]->nb_cols();
        maxinfos[0] = std::max(maxinfos[0], size);
        mininfos[0] = std::min(mininfos[0], size);
        meaninfos[0] += size;
    }
    for (int i = 0; i < MyFarFieldMats.size(); i++) {
        int size    = MyFarFieldMats[i]->nb_rows() * MyFarFieldMats[i]->nb_cols();
        int rank    = MyFarFieldMats[i]->rank_of();
        maxinfos[1] = std::max(maxinfos[1], size);
        mininfos[1] = std::min(mininfos[1], size);
        meaninfos[1] += size;
        maxinfos[2] = std::max(maxinfos[2], rank);
        mininfos[2] = std::min(mininfos[2], rank);
        meaninfos[2] += rank;
    }
    maxinfos[3]  = local_size;
    mininfos[3]  = local_size;
    meaninfos[3] = local_size;

    if (rankWorld == 0) {
        MPI_Reduce(MPI_IN_PLACE, &(maxinfos[0]), 4, MPI_INT, MPI_MAX, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, &(mininfos[0]), 4, MPI_INT, MPI_MIN, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
    } else {
        MPI_Reduce(&(maxinfos[0]), &(maxinfos[0]), 4, MPI_INT, MPI_MAX, 0, comm);
        MPI_Reduce(&(mininfos[0]), &(mininfos[0]), 4, MPI_INT, MPI_MIN, 0, comm);
        MPI_Reduce(&(meaninfos[0]), &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&(false_positive), &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
    }

    int nlrmat   = this->get_nlrmat();
    int ndmat    = this->get_ndmat();
    meaninfos[0] = (ndmat == 0 ? 0 : meaninfos[0] / ndmat);
    meaninfos[1] = (nlrmat == 0 ? 0 : meaninfos[1] / nlrmat);
    meaninfos[2] = (nlrmat == 0 ? 0 : meaninfos[2] / nlrmat);
    meaninfos[3] = meaninfos[3] / sizeWorld;
    mininfos[0]  = (ndmat == 0 ? 0 : mininfos[0]);
    mininfos[1]  = (nlrmat == 0 ? 0 : mininfos[1]);
    mininfos[2]  = (nlrmat == 0 ? 0 : mininfos[2]);

    // timing
    MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&(mytime[0]), &(meantime[0]), 2, MPI_DOUBLE, MPI_SUM, 0, comm);

    meantime /= sizeWorld;

    infos["Block_tree_mean"] = NbrToStr(meantime[0]);
    infos["Block_tree_max"]  = NbrToStr(maxtime[0]);
    infos["Blocks_mean"]     = NbrToStr(meantime[1]);
    infos["Blocks_max"]      = NbrToStr(maxtime[1]);

    // Size
    infos["Source_size"]              = NbrToStr(this->nc);
    infos["Target_size"]              = NbrToStr(this->nr);
    infos["Dense_block_size_max"]     = NbrToStr(maxinfos[0]);
    infos["Dense_block_size_mean"]    = NbrToStr(meaninfos[0]);
    infos["Dense_block_size_min"]     = NbrToStr(mininfos[0]);
    infos["Low_rank_block_size_max"]  = NbrToStr(maxinfos[1]);
    infos["Low_rank_block_size_mean"] = NbrToStr(meaninfos[1]);
    infos["Low_rank_block_size_min"]  = NbrToStr(mininfos[1]);

    infos["Rank_max"]                 = NbrToStr(maxinfos[2]);
    infos["Rank_mean"]                = NbrToStr(meaninfos[2]);
    infos["Rank_min"]                 = NbrToStr(mininfos[2]);
    infos["Number_of_lrmat"]          = NbrToStr(nlrmat);
    infos["Number_of_dmat"]           = NbrToStr(ndmat);
    infos["Number_of_false_positive"] = NbrToStr(false_positive);
    infos["Compression"]              = NbrToStr(this->compression());
    infos["Local_size_max"]           = NbrToStr(maxinfos[3]);
    infos["Local_size_mean"]          = NbrToStr(meaninfos[3]);
    infos["Local_size_min"]           = NbrToStr(mininfos[3]);

    infos["Number_of_MPI_tasks"] = NbrToStr(sizeWorld);
#if _OPENMP
    infos["Number_of_threads_per_tasks"] = NbrToStr(omp_get_max_threads());
    infos["Number_of_procs"]             = NbrToStr(sizeWorld * omp_get_max_threads());
#else
    infos["Number_of_procs"] = NbrToStr(sizeWorld);
#endif

    infos["Eta"]                  = NbrToStr(eta);
    infos["Eps"]                  = NbrToStr(epsilon);
    infos["MinTargetDepth"]       = NbrToStr(mintargetdepth);
    infos["MinSourceDepth"]       = NbrToStr(minsourcedepth);
    infos["MinClusterSizeTarget"] = NbrToStr(cluster_tree_t->get_minclustersize());
    infos["MinClusterSizeSource"] = NbrToStr(cluster_tree_s->get_minclustersize());
    infos["MaxBlockSize"]         = NbrToStr(maxblocksize);
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::mymvprod_global_to_local(const T *const in, T *const out, const int &mu) const {

    std::fill(out, out + local_size * mu, 0);

    // To localize the rhs with multiple rhs, it is transpose. So instead of A*B, we do transpose(B)*transpose(A)
    char transb = 'T';
    // In case of a hermitian matrix, the rhs is conjugate transpose
    if (symmetry == 'H') {
        transb = 'C';
    }

    // Contribution champ lointain
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<T> temp(local_size * mu, 0);
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
        for (int b = 0; b < MyFarFieldMats.size(); b++) {
            const LowRankMatrix<T> &M = *(MyFarFieldMats[b]);
            int offset_i              = M.get_offset_i();
            int offset_j              = M.get_offset_j();
            if (!(symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
                M.add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
            }
        }
// Contribution champ proche
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
        for (int b = 0; b < MyNearFieldMats.size(); b++) {
            const SubMatrix<T> &M = *(MyNearFieldMats[b]);
            int offset_i          = M.get_offset_i();
            int offset_j          = M.get_offset_j();

            if (!(symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
                M.add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
            }
        }

        // Symmetry part of the diagonal part
        if (symmetry != 'N') {
            transb      = 'N';
            char op_sym = 'T';
            if (symmetry == 'H') {
                op_sym = 'C';
            }

#if _OPENMP
#    pragma omp for schedule(guided)
#endif
            for (int b = 0; b < MyDiagFarFieldMats.size(); b++) {
                const LowRankMatrix<T> &M = *(MyDiagFarFieldMats[b]);
                int offset_i              = M.get_offset_j();
                int offset_j              = M.get_offset_i();

                if (offset_i != offset_j) { // remove strictly diagonal blocks
                    M.add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
                }
            }
// #if _OPENMP
// #pragma omp for schedule(guided)
// #endif
// for(int b=0; b<MyStrictlyDiagFarFieldMats.size(); b++){
// 	const LowRankMatrix<T,ClusterImpl>&  M  = *(MyStrictlyDiagFarFieldMats[b]);
// 	int offset_i     = M.get_offset_j();
// 	int offset_j     = M.get_offset_i();

// 	M.add_mvprod_row_major_sym(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);

// }

// Contribution champ proche
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
            for (int b = 0; b < MyDiagNearFieldMats.size(); b++) {
                const SubMatrix<T> &M = *(MyDiagNearFieldMats[b]);
                int offset_i          = M.get_offset_j();
                int offset_j          = M.get_offset_i();

                if (offset_i != offset_j) { // remove strictly diagonal blocks
                    M.add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
                }
            }

#if _OPENMP
#    pragma omp for schedule(guided)
#endif
            for (int b = 0; b < MyStrictlyDiagNearFieldMats.size(); b++) {
                const SubMatrix<T> &M = *(MyStrictlyDiagNearFieldMats[b]);
                int offset_i          = M.get_offset_j();
                int offset_j          = M.get_offset_i();
                M.add_mvprod_row_major_sym(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, this->UPLO, this->symmetry);
            }
        }

#if _OPENMP
#    pragma omp critical
#endif
        std::transform(temp.begin(), temp.end(), out, out, std::plus<T>());
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::local_to_global_target(const T *const in, T *const out, const int &mu) const {
    // Allgather
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = (cluster_tree_t->get_masteroffset(i).second) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::local_to_global_source(const T *const in, T *const out, const int &mu) const {
    // Allgather
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);

    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = (cluster_tree_s->get_masteroffset(i).second) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::mymvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
    double time      = MPI_Wtime();
    bool need_delete = false;
    if (work == nullptr) {
        work        = new T[this->nc * mu];
        need_delete = true;
    }
    this->local_to_global_source(in, work, mu);
    this->mymvprod_global_to_local(work, out, mu);

    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::mvprod_global_to_global(const T *const in, T *const out, const int &mu) const {
    double time = MPI_Wtime();

    if (mu == 1) {
        std::vector<T> out_perm(local_size);
        std::vector<T> buffer(std::max(nc, nr));

        // Permutation
        if (use_permutation) {
            this->source_to_cluster_permutation(in, buffer.data());
            mymvprod_global_to_local(buffer.data(), out_perm.data(), 1);

        } else {
            mymvprod_global_to_local(in, out_perm.data(), 1);
            // Allgather
            std::vector<int> recvcounts(sizeWorld);
            std::vector<int> displs(sizeWorld);

            displs[0] = 0;

            for (int i = 0; i < sizeWorld; i++) {
                recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
                if (i > 0)
                    displs[i] = displs[i - 1] + recvcounts[i - 1];
            }

            MPI_Allgatherv(out, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
        }

        // Allgather
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        if (use_permutation) {
            MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);

            // Permutation
            this->cluster_to_target_permutation(buffer.data(), out);
        } else {
            MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
        }

    } else {

        std::vector<T> in_perm(std::max(nr, nc) * mu * 2);
        std::vector<T> out_perm(local_size * mu);
        std::vector<T> buffer(nc);

        for (int i = 0; i < mu; i++) {
            // Permutation
            if (use_permutation) {
                this->source_to_cluster_permutation(in + i * nc, buffer.data());
                // Transpose
                for (int j = 0; j < nc; j++) {
                    in_perm[i + j * mu] = buffer[j];
                }
            } else {
                // Transpose
                for (int j = 0; j < nc; j++) {
                    in_perm[i + j * mu] = in[j + i * nc];
                }
            }
        }

        if (symmetry == 'H') {
            conj_if_complex(in_perm.data(), nc * mu);
        }

        mymvprod_global_to_local(in_perm.data(), in_perm.data() + nc * mu, mu);

        // Tranpose
        for (int i = 0; i < mu; i++) {
            for (int j = 0; j < local_size; j++) {
                out_perm[i * local_size + j] = in_perm[i + j * mu + nc * mu];
            }
        }

        if (symmetry == 'H') {
            conj_if_complex(out_perm.data(), out_perm.size());
        }

        // Allgather
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_perm.data() + mu * nr, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);

        for (int i = 0; i < mu; i++) {
            if (use_permutation) {
                for (int j = 0; j < sizeWorld; j++) {
                    std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, in_perm.data() + i * nr + displs[j] / mu);
                }

                // Permutation
                this->cluster_to_target_permutation(in_perm.data() + i * nr, out + i * nr);
            } else {
                for (int j = 0; j < sizeWorld; j++) {
                    std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, out + i * nr + displs[j] / mu);
                }
            }
        }
    }
    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::mvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
    double time      = MPI_Wtime();
    bool need_delete = false;
    if (work == nullptr) {
        work        = new T[this->nc * mu];
        need_delete = true;
    }

    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

    if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
        throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used");
    }
    if (mu == 1) {
        std::vector<T> in_perm(local_size_source), out_perm(local_size);

        // local permutation
        if (use_permutation) {
            this->local_source_to_local_cluster(in, in_perm.data());

            // prod
            mymvprod_local_to_local(in_perm.data(), out_perm.data(), 1, work);

            // permutation
            if (use_permutation) {
                this->local_cluster_to_local_target(out_perm.data(), out, comm);
            }
        } else {
            mymvprod_local_to_local(in, out, 1, work);
        }

    } else {

        std::vector<T> in_perm(local_size_source * mu);
        std::vector<T> out_perm(local_size * mu);
        std::vector<T> buffer(std::max(local_size_source, local_size));

        for (int i = 0; i < mu; i++) {
            // local permutation
            if (use_permutation) {
                this->local_source_to_local_cluster(in + i * local_size_source, buffer.data());

                // Transpose
                for (int j = 0; j < local_size_source; j++) {
                    in_perm[i + j * mu] = buffer[j];
                }
            } else {
                // Transpose
                for (int j = 0; j < local_size_source; j++) {
                    in_perm[i + j * mu] = in[j + i * local_size_source];
                }
            }
        }

        if (symmetry == 'H') {
            conj_if_complex(in_perm.data(), local_size_source * mu);
        }

        mymvprod_local_to_local(in_perm.data(), out_perm.data(), mu, work);

        for (int i = 0; i < mu; i++) {
            if (use_permutation) {
                // Tranpose
                for (int j = 0; j < local_size; j++) {
                    buffer[j] = out_perm[i + j * mu];
                }

                // local permutation
                this->local_cluster_to_local_target(buffer.data(), out + i * local_size);
            } else {
                // Tranpose
                for (int j = 0; j < local_size; j++) {
                    out[j + i * local_size] = out_perm[i + j * mu];
                }
            }
        }

        if (symmetry == 'H') {
            conj_if_complex(out, out_perm.size());
        }
    }

    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const {
    std::fill(out, out + local_size * mu, 0);

    // To localize the rhs with multiple rhs, it is transpose. So instead of A*B, we do transpose(B)*transpose(A)
    char transb = 'T';
    // In case of a hermitian matrix, the rhs is conjugate transpose
    if (symmetry == 'H') {
        transb = 'C';
    }

    // Contribution champ lointain
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<T> temp(local_size * mu, 0);
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
        for (int b = 0; b < MyFarFieldMats.size(); b++) {
            const LowRankMatrix<T> &M = *(MyFarFieldMats[b]);
            int offset_i              = M.get_offset_i();
            int offset_j              = M.get_offset_j();
            int size_j                = M.nb_cols();

            if ((offset_j <= offset + size && offset <= offset_j + size_j) && (symmetry == 'N' || offset_i != offset_j)) {
                M.add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
            }
        }
// Contribution champ proche
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
        for (int b = 0; b < MyNearFieldMats.size(); b++) {
            const SubMatrix<T> &M = *(MyNearFieldMats[b]);
            int offset_i          = M.get_offset_i();
            int offset_j          = M.get_offset_j();
            int size_j            = M.nb_cols();

            if ((offset_j <= offset + size && offset <= offset_j + size_j) && (symmetry == 'N' || offset_i != offset_j)) {
                M.add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
            }
        }

        // Symmetry part of the diagonal part
        if (symmetry != 'N') {
            transb      = 'N';
            char op_sym = 'T';
            if (symmetry == 'H') {
                op_sym = 'C';
            }
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
            for (int b = 0; b < MyDiagFarFieldMats.size(); b++) {
                const LowRankMatrix<T> &M = *(MyDiagFarFieldMats[b]);
                int offset_i              = M.get_offset_j();
                int offset_j              = M.get_offset_i();
                int size_j                = M.nb_rows();

                if ((offset_j <= offset + size && offset <= offset_j + size_j) && offset_i != offset_j) { // remove strictly diagonal blocks
                    M.add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
                }
            }

// Contribution champ proche
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
            for (int b = 0; b < MyDiagNearFieldMats.size(); b++) {
                const SubMatrix<T> &M = *(MyDiagNearFieldMats[b]);
                int offset_i          = M.get_offset_j();
                int offset_j          = M.get_offset_i();
                int size_j            = M.nb_rows();

                if ((offset_j <= offset + size && offset <= offset_j + size_j) && offset_i != offset_j) { // remove strictly diagonal blocks
                    M.add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
                }
            }
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
            for (int b = 0; b < MyStrictlyDiagNearFieldMats.size(); b++) {
                const SubMatrix<T> &M = *(MyStrictlyDiagNearFieldMats[b]);
                int offset_i          = M.get_offset_j();
                int offset_j          = M.get_offset_i();
                int size_j            = M.nb_cols();
                if (offset_j <= offset + size && offset <= offset_j + size_j) {
                    M.add_mvprod_row_major_sym(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, this->UPLO, this->symmetry);
                }
            }
        }
#if _OPENMP
#    pragma omp critical
#endif
        std::transform(temp.begin(), temp.end(), out, out, std::plus<T>());
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::source_to_cluster_permutation(const T *const in, T *const out) const {
    for (int i = 0; i < nc; i++) {
        out[i] = in[cluster_tree_s->get_perm(i)];
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::cluster_to_target_permutation(const T *const in, T *const out) const {
    for (int i = 0; i < nr; i++) {
        out[cluster_tree_t->get_perm(i)] = in[i];
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
    if (!cluster_tree_s->IsLocal()) {
        throw std::logic_error("[Htool error] Permutation is not local, local_source_to_local_cluster cannot be used");
    } else {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);
        for (int i = 0; i < cluster_tree_s->get_masteroffset(rankWorld).second; i++) {
            out[i] = in[cluster_tree_s->get_perm(cluster_tree_s->get_masteroffset(rankWorld).first + i) - cluster_tree_s->get_masteroffset(rankWorld).first];
        }
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm) const {
    if (!cluster_tree_t->IsLocal()) {
        throw std::logic_error("[Htool error] Permutation is not local, local_cluster_to_local_target cannot be used");
    } else {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);
        for (int i = 0; i < cluster_tree_t->get_masteroffset(rankWorld).second; i++) {
            out[cluster_tree_t->get_perm(cluster_tree_t->get_masteroffset(rankWorld).first + i) - cluster_tree_t->get_masteroffset(rankWorld).first] = in[i];
        }
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
std::vector<T> HMatrix<T, LowRankMatrix, AdmissibleCondition>::operator*(const std::vector<T> &x) const {
    assert(x.size() == nc);
    std::vector<T> result(nr, 0);
    mvprod_global_to_global(x.data(), result.data(), 1);
    return result;
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
double HMatrix<T, LowRankMatrix, AdmissibleCondition>::compression() const {

    double mycomp = 0.;
    double size   = ((long int)this->nr) * this->nc;
    double nr_b, nc_b, rank;

    for (int j = 0; j < MyFarFieldMats.size(); j++) {
        nr_b = MyFarFieldMats[j]->nb_rows();
        nc_b = MyFarFieldMats[j]->nb_cols();
        rank = MyFarFieldMats[j]->rank_of();
        mycomp += rank * (nr_b + nc_b) / size;
    }

    for (int j = 0; j < MyNearFieldMats.size(); j++) {
        nr_b = MyNearFieldMats[j]->nb_rows();
        nc_b = MyNearFieldMats[j]->nb_cols();
        if (MyNearFieldMats[j]->get_offset_i() == MyNearFieldMats[j]->get_offset_j() && this->get_symmetry_type() != 'N' && nr_b == nc_b) {
            mycomp += nr_b * nc_b / (2 * size);
        } else {
            mycomp += nr_b * nc_b / size;
        }
    }

    double comp = 0;
    MPI_Allreduce(&mycomp, &comp, 1, MPI_DOUBLE, MPI_SUM, comm);

    return 1 - comp;
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::print_infos() const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

    if (rankWorld == 0) {
        for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
            std::cout << it->first << "\t" << it->second << std::endl;
        }
        std::cout << std::endl;
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::save_infos(const std::string &outputname, std::ios_base::openmode mode, const std::string &sep) const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

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

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::save_plot(const std::string &outputname) const {

    std::ofstream outputfile((outputname + "_" + NbrToStr(rankWorld) + ".csv").c_str());

    if (outputfile) {
        outputfile << nr << "," << nc << std::endl;
        for (typename std::vector<std::unique_ptr<SubMatrix<T>>>::const_iterator it = MyNearFieldMats.begin(); it != MyNearFieldMats.end(); ++it) {
            outputfile << (*it)->get_offset_i() << "," << (*it)->get_ir().size() << "," << (*it)->get_offset_j() << "," << (*it)->get_ic().size() << "," << -1 << std::endl;
        }
        for (typename std::vector<std::unique_ptr<LowRankMatrix<T>>>::const_iterator it = MyFarFieldMats.begin(); it != MyFarFieldMats.end(); ++it) {
            outputfile << (*it)->get_offset_i() << "," << (*it)->get_ir().size() << "," << (*it)->get_offset_j() << "," << (*it)->get_ic().size() << "," << (*it)->rank_of() << std::endl;
        }
        outputfile.close();
    } else {
        std::cout << "Unable to create " << outputname << std::endl;
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
double Frobenius_absolute_error(const HMatrix<T, LowRankMatrix, AdmissibleCondition> &B, const IMatrix<T> &A) {
    double myerr = 0;
    for (int j = 0; j < B.MyFarFieldMats.size(); j++) {
        double test = Frobenius_absolute_error(*(B.MyFarFieldMats[j]), A);
        myerr += std::pow(test, 2);
    }

    double err = 0;
    MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

    return std::sqrt(err);
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
Matrix<T> HMatrix<T, LowRankMatrix, AdmissibleCondition>::get_local_dense() const {
    Matrix<T> Dense(local_size, nc);
    // Internal dense blocks
    for (int l = 0; l < MyNearFieldMats.size(); l++) {
        const SubMatrix<T> &submat = *(MyNearFieldMats[l]);
        int local_nr               = submat.nb_rows();
        int local_nc               = submat.nb_cols();
        int offset_i               = submat.get_offset_i();
        int offset_j               = submat.get_offset_j();
        for (int k = 0; k < local_nc; k++) {
            std::copy_n(&(submat(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
        }
    }

    // Internal compressed block
    Matrix<T> FarFielBlock(local_size, local_size);
    for (int l = 0; l < MyFarFieldMats.size(); l++) {
        const LowRankMatrix<T> &lmat = *(MyFarFieldMats[l]);
        int local_nr                 = lmat.nb_rows();
        int local_nc                 = lmat.nb_cols();
        int offset_i                 = lmat.get_offset_i();
        int offset_j                 = lmat.get_offset_j();
        FarFielBlock.resize(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int k = 0; k < local_nc; k++) {
            std::copy_n(&(FarFielBlock(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
        }
    }
    return Dense;
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
Matrix<T> HMatrix<T, LowRankMatrix, AdmissibleCondition>::get_local_dense_perm() const {
    Matrix<T> Dense(local_size, nc);
    copy_local_dense_perm(Dense.data());
    return Dense;
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::copy_local_dense_perm(T *ptr) const {
    if (!(cluster_tree_t->IsLocal())) {
        throw std::logic_error("[Htool error] Permutation is not local, get_local_dense_perm cannot be used");
    }

    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

    // Internal dense blocks
    for (int l = 0; l < MyNearFieldMats.size(); l++) {
        const SubMatrix<T> &submat = *(MyNearFieldMats[l]);
        int local_nr               = submat.nb_rows();
        int local_nc               = submat.nb_cols();
        int offset_i               = submat.get_offset_i();
        int offset_j               = submat.get_offset_j();
        for (int k = 0; k < local_nc; k++) {
            for (int j = 0; j < local_nr; j++) {
                ptr[get_permt(j + offset_i) - local_offset + get_perms(k + offset_j) * local_size] = submat(j, k);
            }
        }
    }

    // Internal compressed block
    Matrix<T> FarFielBlock(local_size, local_size);
    for (int l = 0; l < MyFarFieldMats.size(); l++) {
        const LowRankMatrix<T> &lmat = *(MyFarFieldMats[l]);
        int local_nr                 = lmat.nb_rows();
        int local_nc                 = lmat.nb_cols();
        int offset_i                 = lmat.get_offset_i();
        int offset_j                 = lmat.get_offset_j();
        FarFielBlock.resize(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int k = 0; k < local_nc; k++) {
            for (int j = 0; j < local_nr; j++) {
                ptr[get_permt(j + offset_i) - local_offset + get_perms(k + offset_j) * local_size] = FarFielBlock(j, k);
            }
        }
    }

    // Asking for permutation while symmetry!=N means that the block is upper/lower triangular in Htool's numbering, but it is not true in User's numbering

    if (symmetry != 'N') {
        if (UPLO == 'L' && symmetry == 'S') {
            for (int i = 0; i < local_size; i++) {
                for (int j = 0; j < i; j++) {
                    ptr[get_perms(j + local_offset) - local_offset + get_permt(i + local_offset) * local_size] = ptr[get_permt(i + local_offset) - local_offset + get_perms(j + local_offset) * local_size];
                }
            }
        }

        if (UPLO == 'U' && symmetry == 'S') {
            for (int i = 0; i < local_size; i++) {
                for (int j = i + 1; j < local_size_source; j++) {
                    ptr[get_perms(j + local_offset) - local_offset + get_permt(i + local_offset) * local_size] = ptr[get_permt(i + local_offset) - local_offset + get_perms(j + local_offset) * local_size];
                }
            }
        }
        if (UPLO == 'L' && symmetry == 'H') {
            for (int i = 0; i < local_size; i++) {
                for (int j = 0; j < i; j++) {
                    ptr[get_perms(j + local_offset) - local_offset + get_permt(i + local_offset) * local_size] = conj_if_complex(ptr[get_permt(i + local_offset) - local_offset + get_perms(j + local_offset) * local_size]);
                }
            }
        }

        if (UPLO == 'U' && symmetry == 'H') {
            for (int i = 0; i < local_size; i++) {
                for (int j = i + 1; j < local_size_source; j++) {
                    ptr[get_perms(j + local_offset) - local_offset + get_permt(i + local_offset) * local_size] = conj_if_complex(ptr[get_permt(i + local_offset) - local_offset + get_perms(j + local_offset) * local_size]);
                }
            }
        }
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
std::vector<T> HMatrix<T, LowRankMatrix, AdmissibleCondition>::get_local_diagonal(bool permutation) const {
    std::vector<T> diagonal(local_size, 0);
    copy_local_diagonal(diagonal.data(), permutation);
    return diagonal;
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::copy_local_diagonal(T *ptr, bool permutation) const {
    if (!(cluster_tree_t->IsLocal()) && permutation) {
        throw std::logic_error("[Htool error] Permutation is not local, get_local_diagonal cannot be used");
    }
    if (cluster_tree_t != cluster_tree_s) {
        throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal cannot be used");
    }

    std::vector<T> diagonal(local_size, 0);
    T *d = permutation ? diagonal.data() : ptr;

    for (int j = 0; j < MyStrictlyDiagNearFieldMats.size(); j++) {
        SubMatrix<T> &submat = *(MyStrictlyDiagNearFieldMats[j]);
        int local_nr         = submat.nb_rows();
        int local_nc         = submat.nb_cols();
        int offset_i         = submat.get_offset_i();
        // int offset_j         = submat.get_offset_j();
        for (int i = 0; i < std::min(local_nr, local_nc); i++) {
            d[i + offset_i - local_offset] = submat(i, i);
        }
    }

    if (permutation) {
        this->local_cluster_to_local_target(d, ptr);
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
Matrix<T> HMatrix<T, LowRankMatrix, AdmissibleCondition>::get_local_diagonal_block(bool permutation) const {
    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
    Matrix<T> diagonal_block(local_size, local_size_source);
    copy_local_diagonal_block(diagonal_block.data(), permutation);
    return diagonal_block;
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::copy_local_diagonal_block(T *ptr, bool permutation) const {
    if ((!(cluster_tree_t->IsLocal()) || !(cluster_tree_s->IsLocal())) && permutation) {
        throw std::logic_error("[Htool error] Permutation is not local, get_local_diagonal_block cannot be used");
    }
    if (cluster_tree_t != cluster_tree_s) {
        throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal_block cannot be used");
    }

    int local_offset_source = cluster_tree_s->get_masteroffset(rankWorld).first;
    int local_size_source   = cluster_tree_s->get_masteroffset(rankWorld).second;
    // Internal dense blocks
    for (int i = 0; i < MyDiagNearFieldMats.size(); i++) {
        const SubMatrix<T> &submat = *(MyDiagNearFieldMats[i]);
        int local_nr               = submat.nb_rows();
        int local_nc               = submat.nb_cols();
        int offset_i               = submat.get_offset_i() - local_offset;
        int offset_j               = submat.get_offset_j() - local_offset;
        for (int i = 0; i < local_nc; i++) {
            std::copy_n(&(submat(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
        }
    }

    // Internal compressed block
    Matrix<T> FarFielBlock(local_size, local_size_source);
    for (int i = 0; i < MyDiagFarFieldMats.size(); i++) {
        const LowRankMatrix<T> &lmat = *(MyDiagFarFieldMats[i]);
        int local_nr                 = lmat.nb_rows();
        int local_nc                 = lmat.nb_cols();
        int offset_i                 = lmat.get_offset_i() - local_offset;
        int offset_j                 = lmat.get_offset_j() - local_offset;
        ;
        FarFielBlock.resize(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int i = 0; i < local_nc; i++) {
            std::copy_n(&(FarFielBlock(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
        }
    }

    // Asking for permutation while symmetry!=N means that the block is upper/lower triangular in Htool's numbering, but it is not true in User's numbering

    if (permutation && symmetry != 'N') {
        if (UPLO == 'L' && symmetry == 'S') {
            for (int i = 0; i < local_size; i++) {
                for (int j = 0; j < i; j++) {
                    ptr[j + i * local_size] = ptr[i + j * local_size];
                }
            }
        }

        if (UPLO == 'U' && symmetry == 'S') {
            for (int i = 0; i < local_size; i++) {
                for (int j = i + 1; j < local_size_source; j++) {
                    ptr[j + i * local_size] = ptr[i + j * local_size];
                }
            }
        }
        if (UPLO == 'L' && symmetry == 'H') {
            for (int i = 0; i < local_size; i++) {
                for (int j = 0; j < i; j++) {
                    ptr[j + i * local_size] = conj_if_complex(ptr[i + j * local_size]);
                }
            }
        }

        if (UPLO == 'U' && symmetry == 'H') {
            for (int i = 0; i < local_size; i++) {
                for (int j = i + 1; j < local_size_source; j++) {
                    ptr[j + i * local_size] = conj_if_complex(ptr[i + j * local_size]);
                }
            }
        }
    }
    // Permutations
    if (permutation) {
        Matrix<T> diagonal_block_perm(local_size, local_size_source);
        for (int i = 0; i < local_size; i++) {
            for (int j = 0; j < local_size_source; j++) {
                diagonal_block_perm(i, cluster_tree_s->get_perm(j + local_offset_source) - local_offset_source) = ptr[i + j * local_size];
            }
        }

        for (int i = 0; i < local_size; i++) {
            this->local_cluster_to_local_target(diagonal_block_perm.data() + i * local_size, ptr + i * local_size, comm);
        }
    }
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
std::pair<int, int> HMatrix<T, LowRankMatrix, AdmissibleCondition>::get_max_size_blocks() const {
    int local_max_size_j = 0;
    int local_max_size_i = 0;

    for (int i = 0; i < MyFarFieldMats.size(); i++) {
        if (local_max_size_j < (*MyFarFieldMats[i]).nb_cols())
            local_max_size_j = (*MyFarFieldMats[i]).nb_cols();
        if (local_max_size_i < (*MyFarFieldMats[i]).nb_rows())
            local_max_size_i = (*MyFarFieldMats[i]).nb_rows();
    }
    for (int i = 0; i < MyNearFieldMats.size(); i++) {
        if (local_max_size_j < (*MyNearFieldMats[i]).nb_cols())
            local_max_size_j = (*MyNearFieldMats[i]).nb_cols();
        if (local_max_size_i < (*MyNearFieldMats[i]).nb_rows())
            local_max_size_i = (*MyNearFieldMats[i]).nb_rows();
    }

    return std::pair<int, int>(local_max_size_i, local_max_size_j);
}

template <typename T, template <typename> class LowRankMatrix, class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, AdmissibleCondition>::apply_dirichlet(const std::vector<int> &boundary) {
    // Renum
    std::vector<int> boundary_renum(boundary.size());
    this->source_to_cluster_permutation(boundary.data(), boundary_renum.data());

    //
    for (int j = 0; j < MyStrictlyDiagNearFieldMats.size(); j++) {
        SubMatrix<T> &submat = *(MyStrictlyDiagNearFieldMats[j]);
        int local_nr         = submat.nb_rows();
        int local_nc         = submat.nb_cols();
        int offset_i         = submat.get_offset_i();
        for (int i = offset_i; i < offset_i + std::min(local_nr, local_nc); i++) {
            if (boundary_renum[i])
                submat(i - offset_i, i - offset_i) = 1e30;
        }
    }
}

} // namespace htool
#endif
