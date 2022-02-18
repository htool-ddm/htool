#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if _OPENMP
#    include <omp.h>
#endif

#include "../blocks/admissibility_conditions.hpp"
#include "../blocks/blocks.hpp"
#include "../clustering/virtual_cluster.hpp"
#include "../lrmat/lrmat.hpp"
#include "../lrmat/sympartialACA.hpp"
#include "../lrmat/virtual_lrmat_generator.hpp"
#include "../misc/misc.hpp"
#include "../types/virtual_dense_blocks_generator.hpp"
#include "../types/virtual_generator.hpp"
#include "../types/virtual_hmatrix.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "matrix.hpp"
#include "point.hpp"
#include "zero_generator.hpp"
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
// template <typename T>
// class MultiHMatrix;
template <typename T>
class HMatrix;

template <typename T>
underlying_type<T> Frobenius_absolute_error(const HMatrix<T> &B, const VirtualGenerator<T> &A);

// Class
template <typename T>
class HMatrix : public VirtualHMatrix<T> {

  private:
    // Data members
    int nr;
    int nc;
    int space_dim;
    int dimension;
    int reqrank;
    int local_size;
    int local_offset;
    char symmetry;
    char UPLO;
    int false_positive;
    bool use_permutation;
    bool delay_dense_computation;

    // Parameters
    double epsilon;
    double eta;
    int maxblocksize;
    int minsourcedepth;
    int mintargetdepth;

    // Strategies
    std::shared_ptr<VirtualLowRankGenerator<T>> LowRankGenerator;
    std::shared_ptr<VirtualAdmissibilityCondition> AdmissibilityCondition;

    std::shared_ptr<VirtualCluster> cluster_tree_t;
    std::shared_ptr<VirtualCluster> cluster_tree_s;

    std::unique_ptr<Block> BlockTree;

    std::vector<std::unique_ptr<IMatrix<T>>> MyComputedBlocks;
    std::vector<LowRankMatrix<T> *> MyFarFieldMats;
    std::vector<SubMatrix<T> *> MyNearFieldMats;
    std::vector<LowRankMatrix<T> *> MyDiagFarFieldMats;
    std::vector<SubMatrix<T> *> MyDiagNearFieldMats;
    std::vector<IMatrix<T> *> MyDiagComputedBlocks;
    std::vector<LowRankMatrix<T> *> MyStrictlyDiagFarFieldMats;
    std::vector<SubMatrix<T> *> MyStrictlyDiagNearFieldMats;

    std::vector<int> no_permutation_target, no_permutation_source;
    mutable std::map<std::string, std::string> infos;
    std::unique_ptr<ZeroGenerator<T>> zerogenerator;

    const MPI_Comm comm;
    int rankWorld, sizeWorld;

    // Internal methods
    void ComputeBlocks(VirtualGenerator<T> &mat, const double *const xt, const double *const xs);
    void ComputeSymBlocks(VirtualGenerator<T> &mat, const double *const xt, const double *const xs);
    bool ComputeAdmissibleBlock(VirtualGenerator<T> &mat, Block &task, const double *const xt, const double *const xs, std::vector<std::unique_ptr<IMatrix<T>>> &MyComputedBlocks_local, std::vector<SubMatrix<T> *> &, std::vector<LowRankMatrix<T> *> &, int &);
    bool ComputeAdmissibleBlocksSym(VirtualGenerator<T> &mat, Block &task, const double *const xt, const double *const xs, std::vector<std::unique_ptr<IMatrix<T>>> &MyComputedBlocks_local, std::vector<SubMatrix<T> *> &, std::vector<LowRankMatrix<T> *> &, int &);
    void AddNearFieldMat(VirtualGenerator<T> &mat, Block &task, std::vector<std::unique_ptr<IMatrix<T>>> &, std::vector<SubMatrix<T> *> &);
    void AddFarFieldMat(VirtualGenerator<T> &mat, Block &task, const double *const xt, const double *const xs, std::vector<std::unique_ptr<IMatrix<T>>> &, std::vector<LowRankMatrix<T> *> &, const int &reqrank = -1);
    void ComputeInfos(const std::vector<double> &mytimes);

    // Check arguments
    void check_arguments(VirtualGenerator<T> &mat, const std::vector<R3> &xt, const std::vector<R3> &xs) const;
    void check_arguments_sym(VirtualGenerator<T> &mat, const std::vector<R3> &xt) const;

    // Friends
    // friend class MultiHMatrix<T>;

  public:
    // Special constructor for hand-made build (for MultiHMatrix for example)

    HMatrix(int space_dim0, int nr0, int nc0, const std::shared_ptr<VirtualCluster> &cluster_tree_t0, const std::shared_ptr<VirtualCluster> &cluster_tree_s0, char symmetry0 = 'N', char UPLO = 'N', const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(nr0), nc(nc0), space_dim(space_dim0), symmetry(symmetry0), UPLO(UPLO), use_permutation(true), delay_dense_computation(false), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0), comm(comm0){};

    // Constructor
    HMatrix(const std::shared_ptr<VirtualCluster> &cluster_tree_t0, const std::shared_ptr<VirtualCluster> &cluster_tree_s0, double epsilon0 = 1e-6, double eta0 = 10, char Symmetry = 'N', char UPLO = 'N', const int &reqrank0 = -1, const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(0), nc(0), space_dim(cluster_tree_t0->get_space_dim()), dimension(1), reqrank(reqrank0), local_size(0), local_offset(0), symmetry(Symmetry), UPLO(UPLO), false_positive(0), use_permutation(true), delay_dense_computation(false), epsilon(epsilon0), eta(eta0), maxblocksize(1e6), minsourcedepth(0), mintargetdepth(0), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0), comm(comm0) {
        if (!((symmetry == 'N' || symmetry == 'H' || symmetry == 'S')
              && (UPLO == 'N' || UPLO == 'L' || UPLO == 'U')
              && ((symmetry == 'N' && UPLO == 'N') || (symmetry != 'N' && UPLO != 'N'))
              && ((symmetry == 'H' && is_complex<T>()) || symmetry != 'H'))) {
            throw std::invalid_argument("[Htool error] Invalid arguments to create HMatrix"); // LCOV_EXCL_LINE
        }
    };

    // Build
    void build(VirtualGenerator<T> &mat, const double *const xt, const double *const xs);

    void build(VirtualGenerator<T> &mat, const std::vector<R3> &xt, const std::vector<R3> &xs) {
        if (this->space_dim != 3) {
            throw std::logic_error("[Htool error] Wrong space dimension"); // LCOV_EXCL_LINE
        }
        std::vector<double> x_array_t(xt.size() * this->space_dim), x_array_s(xs.size() * this->space_dim);
        for (int p = 0; p < xt.size(); p++) {
            std::copy_n(xt[p].data(), space_dim, &(x_array_t[this->space_dim * p]));
        }
        for (int p = 0; p < xs.size(); p++) {
            std::copy_n(xs[p].data(), space_dim, &(x_array_s[this->space_dim * p]));
        }
        this->check_arguments(mat, xt, xs);
        this->build(mat, x_array_t.data(), x_array_s.data());
    }

    // Symmetry build
    void build(VirtualGenerator<T> &mat, const double *const xt);

    void build(VirtualGenerator<T> &mat, const std::vector<R3> &xt) {
        if (this->space_dim != 3) {
            throw std::logic_error("[Htool error] Wrong space dimension"); // LCOV_EXCL_LINE
        }
        std::vector<double> x_array_t(xt.size() * this->space_dim);
        for (int p = 0; p < xt.size(); p++) {
            std::copy_n(xt[p].data(), space_dim, &(x_array_t[this->space_dim * p]));
        }
        this->check_arguments_sym(mat, xt);
        this->build(mat, x_array_t.data());
    }

    void build_dense_blocks(VirtualDenseBlocksGenerator<T> &dense_block_generator);

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

    const VirtualCluster *get_target_cluster() const { return cluster_tree_t.get(); }
    const VirtualCluster *get_source_cluster() const { return cluster_tree_s.get(); }
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
    const std::vector<SubMatrix<T> *> &get_MyNearFieldMats() const { return MyNearFieldMats; }
    const std::vector<LowRankMatrix<T> *> &get_MyFarFieldMats() const { return MyFarFieldMats; }
    const std::vector<SubMatrix<T> *> &get_MyDiagNearFieldMats() const { return MyDiagNearFieldMats; }
    const std::vector<LowRankMatrix<T> *> &get_MyDiagFarFieldMats() const { return MyDiagFarFieldMats; }
    const std::vector<SubMatrix<T> *> &get_MyStrictlyDiagNearFieldMats() const { return MyStrictlyDiagNearFieldMats; }
    const std::vector<LowRankMatrix<T> *> &get_MyStrictlyDiagFarFieldMats() const { return MyStrictlyDiagFarFieldMats; }
    std::vector<T> get_local_diagonal(bool = true) const;
    void copy_local_diagonal(T *, bool = true) const;
    Matrix<T> get_local_interaction(bool = true) const;
    Matrix<T> get_local_diagonal_block(bool = true) const;
    void copy_local_interaction(T *, bool = true) const;
    void copy_local_diagonal_block(T *, bool = true) const;
    std::pair<int, int> get_max_size_blocks() const;

    double get_epsilon() const { return this->epsilon; };
    double get_eta() const { return this->eta; };
    int get_dimension() const { return this->dimension; };
    int get_minsourcedepth() const { return this->minsourcedepth; };
    int get_mintargetdepth() const { return this->mintargetdepth; };
    int get_maxblocksize() const { return this->maxblocksize; };
    void set_epsilon(double epsilon0) { this->epsilon = epsilon0; };
    void set_eta(double eta0) { this->eta = eta0; };
    void set_minsourcedepth(unsigned int minsourcedepth0) { this->minsourcedepth = minsourcedepth0; };
    void set_mintargetdepth(unsigned int mintargetdepth0) { this->mintargetdepth = mintargetdepth0; };
    void set_maxblocksize(unsigned int maxblocksize0) { this->maxblocksize = maxblocksize0; };
    void set_use_permutation(bool choice) { this->use_permutation = choice; };
    void set_delay_dense_computation(bool choice) { this->delay_dense_computation = choice; };
    void set_compression(std::shared_ptr<VirtualLowRankGenerator<T>> ptr) { LowRankGenerator = ptr; };

    // Infos
    const std::map<std::string, std::string> &get_infos() const { return infos; }
    std::string get_infos(const std::string &key) const { return infos[key]; }
    void add_info(const std::string &keyname, const std::string &value) const { infos[keyname] = value; }
    void print_infos() const;
    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const;
    void save_plot(const std::string &outputname) const;
    double compression_ratio() const;
    double space_saving() const;
    friend underlying_type<T> Frobenius_absolute_error<T>(const HMatrix<T> &B, const VirtualGenerator<T> &A);

    // Mat vec prod
    void mvprod_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    void mvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    void mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    void mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    void mymvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    void mymvprod_global_to_local(const T *const in, T *const out, const int &mu = 1) const;
    void mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    void mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu = 1) const;

    void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const;
    std::vector<T> operator*(const std::vector<T> &x) const;
    Matrix<T> operator*(const Matrix<T> &x) const;

    // Permutations
    void source_to_cluster_permutation(const T *const in, T *const out) const;
    void target_to_cluster_permutation(const T *const in, T *const out) const;
    void cluster_to_target_permutation(const T *const in, T *const out) const;
    void cluster_to_source_permutation(const T *const in, T *const out) const;
    void local_target_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    void local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    void local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    void local_cluster_to_local_source(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;

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

template <typename T>
void HMatrix<T>::check_arguments(VirtualGenerator<T> &mat, const std::vector<R3> &xt, const std::vector<R3> &xs) const {
    if (!(mat.nb_rows() == mat.get_dimension() * xt.size() && mat.nb_cols() == mat.get_dimension() * xs.size())) {
        throw std::invalid_argument("[Htool error] Invalid size in arguments for building HMatrix"); // LCOV_EXCL_LINE
    }
}

template <typename T>
void HMatrix<T>::check_arguments_sym(VirtualGenerator<T> &mat, const std::vector<R3> &xt) const {
    this->check_arguments(mat, xt, xt);
}

// build
template <typename T>
void HMatrix<T>::build(VirtualGenerator<T> &mat, const double *const xt, const double *const xs) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(3), maxtime(3), meantime(3);

    this->nc        = mat.nb_cols();
    this->nr        = mat.nb_rows();
    this->dimension = mat.get_dimension();

    // Default compression: sympartialACA
    if (this->LowRankGenerator == nullptr) {
        this->LowRankGenerator = std::make_shared<sympartialACA<T>>();
    }

    // Default admissibility condition
    if (this->AdmissibilityCondition == nullptr) {
        this->AdmissibilityCondition = std::make_shared<RjasanowSteinbach>();
    }

    // Use no_permutation if needed
    if (use_permutation == false) {
        no_permutation_target.resize(nr);
        no_permutation_source.resize(nc);
        std::iota(no_permutation_target.begin(), no_permutation_target.end(), int(0));
        std::iota(no_permutation_source.begin(), no_permutation_source.end(), int(0));
    }

    // Zero generator when we delay the dense computation
    if (delay_dense_computation) {
        zerogenerator = std::unique_ptr<ZeroGenerator<T>>(new ZeroGenerator<T>(mat.nb_rows(), mat.nb_cols(), mat.get_dimension()));
    }

    // Construction arbre des paquets
    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    // Construction arbre des blocs
    double time = MPI_Wtime();
    this->BlockTree.reset(new Block(this->AdmissibilityCondition.get(), *cluster_tree_t, *cluster_tree_s));
    this->BlockTree->set_mintargetdepth(this->mintargetdepth);
    this->BlockTree->set_minsourcedepth(this->minsourcedepth);
    this->BlockTree->set_maxblocksize(this->maxblocksize);
    this->BlockTree->set_eta(this->eta);
    bool force_sym = false;
    this->BlockTree->build(UPLO, force_sym, comm);
    mytimes[0] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, xs);
    mytimes[1] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

// Symmetry build
template <typename T>
void HMatrix<T>::build(VirtualGenerator<T> &mat, const double *const xt) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(3), maxtime(3), meantime(3);

    this->nc        = mat.nb_cols();
    this->nr        = mat.nb_rows();
    this->dimension = mat.get_dimension();

    // Default compression: sympartialACA
    if (this->LowRankGenerator == nullptr) {
        this->LowRankGenerator = std::make_shared<sympartialACA<T>>();
    }

    // Default admissibility condition
    if (this->AdmissibilityCondition == nullptr) {
        this->AdmissibilityCondition = std::make_shared<RjasanowSteinbach>();
    }

    // Use no_permutation if needed
    if (use_permutation == false) {
        no_permutation_target.resize(nr);
        no_permutation_source.resize(nc);
        std::iota(no_permutation_target.begin(), no_permutation_target.end(), int(0));
        std::iota(no_permutation_source.begin(), no_permutation_source.end(), int(0));
    }

    // Zero generator when we delay the dense computation
    if (delay_dense_computation) {
        zerogenerator = std::unique_ptr<ZeroGenerator<T>>(new ZeroGenerator<T>(mat.nb_rows(), mat.nb_cols(), mat.get_dimension()));
    }

    // Construction arbre des paquets
    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    // Construction arbre des blocs
    double time = MPI_Wtime();

    this->BlockTree.reset(new Block(this->AdmissibilityCondition.get(), *cluster_tree_t, *cluster_tree_s));
    this->BlockTree->set_mintargetdepth(this->mintargetdepth);
    this->BlockTree->set_minsourcedepth(this->minsourcedepth);
    this->BlockTree->set_maxblocksize(this->maxblocksize);
    this->BlockTree->set_eta(this->eta);
    bool force_sym = true;
    this->BlockTree->build(UPLO, force_sym, comm);

    mytimes[0] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, xt);
    mytimes[1] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

template <typename T>
void HMatrix<T>::build_dense_blocks(VirtualDenseBlocksGenerator<T> &dense_block_generator) {

    std::vector<int> row_sizes(this->MyNearFieldMats.size()), col_sizes(this->MyNearFieldMats.size());
    std::vector<const int *> rows(this->MyNearFieldMats.size()), cols(this->MyNearFieldMats.size());
    std::vector<T *> ptr(this->MyNearFieldMats.size());
    for (int i = 0; i < this->MyNearFieldMats.size(); i++) {
        row_sizes[i] = this->MyNearFieldMats[i]->nb_rows();
        col_sizes[i] = this->MyNearFieldMats[i]->nb_cols();
        rows[i]      = (this->MyNearFieldMats[i]->data_ir());
        cols[i]      = (this->MyNearFieldMats[i]->data_ic());
        ptr[i]       = (this->MyNearFieldMats[i]->data());
    }
    dense_block_generator.copy_dense_blocks(row_sizes, col_sizes, rows, cols, ptr);
}

// Compute blocks recursively
// TODO: recursivity -> stack for compute blocks
template <typename T>
void HMatrix<T>::ComputeBlocks(VirtualGenerator<T> &mat, const double *const xt, const double *const xs) {
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp parallel
#endif
    {
        std::vector<SubMatrix<T> *> MyNearFieldMats_local;
        std::vector<LowRankMatrix<T> *> MyFarFieldMats_local;
        std::vector<std::unique_ptr<IMatrix<T>>> MyComputedBlocks_local;
        std::vector<Block *> local_tasks = BlockTree->get_local_tasks();

        int false_positive_local = 0;
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp for schedule(guided)
#endif
        for (int p = 0; p < local_tasks.size(); p++) {
            if (!local_tasks[p]->IsAdmissible()) {
                AddNearFieldMat(mat, *(local_tasks[p]), MyComputedBlocks_local, MyNearFieldMats_local);
            } else {
                bool not_pushed;
                if (symmetry == 'H' || symmetry == 'S') {
                    not_pushed = ComputeAdmissibleBlocksSym(mat, *(local_tasks[p]), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                } else {
                    not_pushed = ComputeAdmissibleBlock(mat, *(local_tasks[p]), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }

                if (not_pushed) {
                    AddNearFieldMat(mat, *(local_tasks[p]), MyComputedBlocks_local, MyNearFieldMats_local);
                }
            }
        }
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp critical
#endif
        {
            MyComputedBlocks.insert(MyComputedBlocks.end(), std::make_move_iterator(MyComputedBlocks_local.begin()), std::make_move_iterator(MyComputedBlocks_local.end()));

            MyFarFieldMats.insert(MyFarFieldMats.end(), std::make_move_iterator(MyFarFieldMats_local.begin()), std::make_move_iterator(MyFarFieldMats_local.end()));

            MyNearFieldMats.insert(MyNearFieldMats.end(), std::make_move_iterator(MyNearFieldMats_local.begin()), std::make_move_iterator(MyNearFieldMats_local.end()));

            false_positive += false_positive_local;
        }
    }

    int local_offset_s = cluster_tree_s->get_local_offset();
    int local_size_s   = cluster_tree_s->get_local_size();

    // Build vectors of pointers for diagonal blocks
    for (int i = 0; i < MyComputedBlocks.size(); i++) {
        if (local_offset_s <= MyComputedBlocks[i]->get_offset_j() && MyComputedBlocks[i]->get_offset_j() < local_offset_s + local_size_s) {
            MyDiagComputedBlocks.push_back(MyComputedBlocks[i].get());
            // if (MyComputedBlocks[i]->get_offset_j() == MyComputedBlocks[i]->get_offset_i())
            //     MyStrictlyDiagFarFieldMats.push_back(MyComputedBlocks[i]);
        }
    }

    for (int i = 0; i < MyFarFieldMats.size(); i++) {
        if (local_offset_s <= MyFarFieldMats[i]->get_offset_j() && MyFarFieldMats[i]->get_offset_j() < local_offset_s + local_size_s) {
            MyDiagFarFieldMats.push_back(MyFarFieldMats[i]);
            if (MyFarFieldMats[i]->get_offset_j() == MyFarFieldMats[i]->get_offset_i())
                MyStrictlyDiagFarFieldMats.push_back(MyFarFieldMats[i]);
        }
    }
    for (int i = 0; i < MyNearFieldMats.size(); i++) {
        if (local_offset_s <= MyNearFieldMats[i]->get_offset_j() && MyNearFieldMats[i]->get_offset_j() < local_offset_s + local_size_s) {
            MyDiagNearFieldMats.push_back(MyNearFieldMats[i]);
            if (MyNearFieldMats[i]->get_offset_j() == MyNearFieldMats[i]->get_offset_i())
                MyStrictlyDiagNearFieldMats.push_back(MyNearFieldMats[i]);
        }
    }

    std::sort(MyComputedBlocks.begin(), MyComputedBlocks.end(), [](const std::unique_ptr<IMatrix<T>> &a, const std::unique_ptr<IMatrix<T>> &b) {
        if (a->get_offset_i() == b->get_offset_i()) {
            return a->get_offset_j() < b->get_offset_j();
        } else {
            return a->get_offset_i() < b->get_offset_i();
        }
    });
}

template <typename T>
bool HMatrix<T>::ComputeAdmissibleBlock(VirtualGenerator<T> &mat, Block &task, const double *const xt, const double *const xs, std::vector<std::unique_ptr<IMatrix<T>>> &MyComputedBlocks_local, std::vector<SubMatrix<T> *> &MyNearFieldMats_local, std::vector<LowRankMatrix<T> *> &MyFarFieldMats_local, int &false_positive_local) {
    if (task.IsAdmissible()) { // When called recursively, it may not be admissible
        AddFarFieldMat(mat, task, xt, xs, MyComputedBlocks_local, MyFarFieldMats_local, reqrank);
        if (MyFarFieldMats_local.back()->rank_of() != -1) {
            return false;
        } else {
            MyFarFieldMats_local.pop_back();
            MyComputedBlocks_local.pop_back();
            false_positive_local += 1;
        }
    }
    // We could compute a dense block if its size is small enough, we focus on improving compression for now
    // else if (task.get_size()<maxblocksize){
    //     AddNearFieldMat(mat,task,MyComputedBlocks_local, MyNearFieldMats_local);
    //     return false;
    // }

    std::size_t bsize       = task.get_size();
    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (s.IsLeaf()) {
        if (t.IsLeaf()) {
            return true;
        } else {
            std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
            for (int p = 0; p < t.get_nb_sons(); p++) {
                task.build_son(t.get_son(p), s);

                Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task.get_son(p), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
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
                Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task.get_son(p), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
                    }
                }
                return false;
            }
        } else {
            if (t.get_size() > s.get_size()) {
                std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    task.build_son(t.get_son(p), s);
                    Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task.get_son(p), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                    task.clear_sons();
                    return true;
                } else {
                    for (int p = 0; p < t.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
                        }
                    }
                    return false;
                }
            } else {
                std::vector<bool> Blocks_not_pushed(s.get_nb_sons());
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    task.build_son(t, s.get_son(p));
                    Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task.get_son(p), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                    task.clear_sons();
                    return true;
                } else {
                    for (int p = 0; p < s.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
                        }
                    }
                    return false;
                }
            }
        }
    }
}

template <typename T>
bool HMatrix<T>::ComputeAdmissibleBlocksSym(VirtualGenerator<T> &mat, Block &task, const double *const xt, const double *const xs, std::vector<std::unique_ptr<IMatrix<T>>> &MyComputedBlocks_local, std::vector<SubMatrix<T> *> &MyNearFieldMats_local, std::vector<LowRankMatrix<T> *> &MyFarFieldMats_local, int &false_positive_local) {

    if (task.IsAdmissible()) {

        AddFarFieldMat(mat, task, xt, xs, MyComputedBlocks_local, MyFarFieldMats_local, reqrank);
        if (MyFarFieldMats_local.back()->rank_of() != -1) {
            return false;
        } else {
            MyFarFieldMats_local.pop_back();
            MyComputedBlocks_local.pop_back();
            false_positive_local += 1;
            // AddNearFieldMat(mat, task,MyComputedBlocks_local, MyNearFieldMats_local);
            // return false;
        }
    }
    // We could compute a dense block if its size is small enough, we focus on improving compression for now
    // else if (task.get_size()<maxblocksize){
    //     AddNearFieldMat(mat,task,MyComputedBlocks_local, MyNearFieldMats_local);
    //     return false;
    // }

    std::size_t bsize       = task.get_size();
    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (s.IsLeaf()) {
        if (t.IsLeaf()) {
            return true;
        } else {
            std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
            for (int p = 0; p < t.get_nb_sons(); p++) {
                task.build_son(t.get_son(p), s);
                Blocks_not_pushed[p] = ComputeAdmissibleBlocksSym(mat, task.get_son(p), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
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
                Blocks_not_pushed[p] = ComputeAdmissibleBlocksSym(mat, task.get_son(p), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
            }

            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
                    }
                }
                return false;
            }
        } else {
            std::vector<bool> Blocks_not_pushed(t.get_nb_sons() * s.get_nb_sons());
            for (int l = 0; l < s.get_nb_sons(); l++) {
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    task.build_son(t.get_son(p), s.get_son(l));
                    Blocks_not_pushed[p + l * t.get_nb_sons()] = ComputeAdmissibleBlocksSym(mat, task.get_son(p + l * t.get_nb_sons()), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
                }
            }
            if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
                task.clear_sons();
                return true;
            } else {
                for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                    if (Blocks_not_pushed[p]) {
                        AddNearFieldMat(mat, task.get_son(p), MyComputedBlocks_local, MyNearFieldMats_local);
                    }
                }
                return false;
            }
        }
    }
}

// Build a dense block
template <typename T>
void HMatrix<T>::AddNearFieldMat(VirtualGenerator<T> &mat, Block &task, std::vector<std::unique_ptr<IMatrix<T>>> &MyComputedBlocks_local, std::vector<SubMatrix<T> *> &MyNearFieldMats_local) {

    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (use_permutation && !delay_dense_computation) {
        MyComputedBlocks_local.emplace_back(new SubMatrix<T>(mat, mat.get_dimension() * t.get_size(), mat.get_dimension() * s.get_size(), cluster_tree_t->get_perm().data() + t.get_offset(), cluster_tree_s->get_perm().data() + s.get_offset(), t.get_offset(), s.get_offset()));
    } else if (!delay_dense_computation) {
        MyComputedBlocks_local.emplace_back(new SubMatrix<T>(mat, mat.get_dimension() * t.get_size(), mat.get_dimension() * s.get_size(), no_permutation_target.data() + t.get_offset(), no_permutation_source.data() + s.get_offset(), t.get_offset(), s.get_offset()));
    } else {
        std::cout << mat.get_dimension() * t.get_size() << " " << mat.get_dimension() * s.get_size() << std::endl;
        MyComputedBlocks_local.emplace_back(new SubMatrix<T>(*zerogenerator, mat.get_dimension() * t.get_size(), mat.get_dimension() * s.get_size(), cluster_tree_t->get_perm().data() + t.get_offset(), cluster_tree_s->get_perm().data() + s.get_offset(), t.get_offset(), s.get_offset()));
    }

    MyNearFieldMats_local.push_back(dynamic_cast<SubMatrix<T> *>(MyComputedBlocks_local.back().get()));
}

// Build a low rank block
template <typename T>
void HMatrix<T>::AddFarFieldMat(VirtualGenerator<T> &mat, Block &task, const double *const xt, const double *const xs, std::vector<std::unique_ptr<IMatrix<T>>> &MyComputedBlocks_local, std::vector<LowRankMatrix<T> *> &MyFarFieldMats_local, const int &reqrank) {

    const VirtualCluster &t = task.get_target_cluster();
    const VirtualCluster &s = task.get_source_cluster();

    if (use_permutation) {
        MyComputedBlocks_local.emplace_back(new LowRankMatrix<T>(mat.get_dimension(), std::vector<int>(cluster_tree_t->get_perm_start() + t.get_offset(), cluster_tree_t->get_perm_start() + t.get_offset() + t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start() + s.get_offset(), cluster_tree_s->get_perm_start() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset(), reqrank, this->epsilon));
    }

    else {
        MyComputedBlocks_local.emplace_back(new LowRankMatrix<T>(mat.get_dimension(), std::vector<int>(no_permutation_target.data() + t.get_offset(), no_permutation_target.data() + t.get_offset() + t.get_size()), std::vector<int>(no_permutation_source.data() + s.get_offset(), no_permutation_source.data() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset(), reqrank, this->epsilon));
    }

    MyFarFieldMats_local.push_back(dynamic_cast<LowRankMatrix<T> *>(MyComputedBlocks_local.back().get()));
    MyFarFieldMats_local.back()->build(mat, *LowRankGenerator, t, xt, s, xs);
}

// Compute infos
template <typename T>
void HMatrix<T>::ComputeInfos(const std::vector<double> &mytime) {
    // 0 : block tree ; 1 : compute blocks ;
    std::vector<double> maxtime(2), meantime(2);
    // 0 : dense mat ; 1 : lr mat ; 2 : rank ; 3 : local_size
    std::vector<std::size_t> maxinfos(4, 0), mininfos(4, std::max(nc, nr));
    std::vector<double> meaninfos(4, 0);
    // Infos
    for (int i = 0; i < MyNearFieldMats.size(); i++) {
        std::size_t size = MyNearFieldMats[i]->nb_rows() * MyNearFieldMats[i]->nb_cols();
        maxinfos[0]      = std::max(maxinfos[0], size);
        mininfos[0]      = std::min(mininfos[0], size);
        meaninfos[0] += size;
    }
    for (int i = 0; i < MyFarFieldMats.size(); i++) {
        std::size_t size = MyFarFieldMats[i]->nb_rows() * MyFarFieldMats[i]->nb_cols();
        std::size_t rank = MyFarFieldMats[i]->rank_of();
        maxinfos[1]      = std::max(maxinfos[1], size);
        mininfos[1]      = std::min(mininfos[1], size);
        meaninfos[1] += size;
        maxinfos[2] = std::max(maxinfos[2], rank);
        mininfos[2] = std::min(mininfos[2], rank);
        meaninfos[2] += rank;
    }
    maxinfos[3]  = local_size;
    mininfos[3]  = local_size;
    meaninfos[3] = local_size;

    if (rankWorld == 0) {
        MPI_Reduce(MPI_IN_PLACE, &(maxinfos[0]), 4, my_MPI_SIZE_T, MPI_MAX, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, &(mininfos[0]), 4, my_MPI_SIZE_T, MPI_MIN, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
    } else {
        MPI_Reduce(&(maxinfos[0]), &(maxinfos[0]), 4, my_MPI_SIZE_T, MPI_MAX, 0, comm);
        MPI_Reduce(&(mininfos[0]), &(mininfos[0]), 4, my_MPI_SIZE_T, MPI_MIN, 0, comm);
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
    infos["Dimension"]                = NbrToStr(this->dimension);
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
    infos["Compression_ratio"]        = NbrToStr(this->compression_ratio());
    infos["Space_saving"]             = NbrToStr(this->space_saving());
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

    infos["Eta"]                   = NbrToStr(eta);
    infos["Eps"]                   = NbrToStr(epsilon);
    infos["MinTargetDepth"]        = NbrToStr(mintargetdepth);
    infos["MinSourceDepth"]        = NbrToStr(minsourcedepth);
    infos["MinClusterSizeTarget"]  = NbrToStr(cluster_tree_t->get_minclustersize());
    infos["MinClusterSizeSource"]  = NbrToStr(cluster_tree_s->get_minclustersize());
    infos["MinClusterDepthTarget"] = NbrToStr(cluster_tree_t->get_min_depth());
    infos["MaxClusterDepthTarget"] = NbrToStr(cluster_tree_t->get_max_depth());
    infos["MinClusterDepthSource"] = NbrToStr(cluster_tree_s->get_min_depth());
    infos["MaxClusterDepthSource"] = NbrToStr(cluster_tree_s->get_max_depth());
    infos["MaxBlockSize"]          = NbrToStr(maxblocksize);
}

template <typename T>
void HMatrix<T>::mymvprod_global_to_local(const T *const in, T *const out, const int &mu) const {

    std::fill(out, out + local_size * mu, 0);
    int incx(1), incy(1), local_size_rhs(local_size * mu);
    T da(1);

    // Contribution champ lointain
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<T> temp(local_size * mu, 0);
        // To localize the rhs with multiple rhs, it is transpose. So instead of A*B, we do transpose(B)*transpose(A)
        char transb = 'T';
        // In case of a hermitian matrix, the rhs is conjugate transpose
        if (symmetry == 'H') {
            transb = 'C';
        }

        // Contribution champ lointain
#if _OPENMP
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < MyComputedBlocks.size(); b++) {
            int offset_i = MyComputedBlocks[b]->get_offset_i();
            int offset_j = MyComputedBlocks[b]->get_offset_j();
            if (!(symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
                MyComputedBlocks[b]->add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
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
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < MyDiagComputedBlocks.size(); b++) {
                int offset_i = MyDiagComputedBlocks[b]->get_offset_j();
                int offset_j = MyDiagComputedBlocks[b]->get_offset_i();

                if (offset_i != offset_j) { // remove strictly diagonal blocks
                    MyDiagComputedBlocks[b]->add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
                }
            }

#if _OPENMP
#    pragma omp for schedule(guided) nowait
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
        Blas<T>::axpy(&local_size_rhs, &da, temp.data(), &incx, out, &incy);
    }
}

template <typename T>
void HMatrix<T>::mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu) const {
    std::fill(out, out + this->nc * mu, 0);
    int incx(1), incy(1);
    int global_size_rhs = this->nc * mu;
    T da(1);

    // Contribution champ lointain
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<T> temp(this->nc * mu, 0);
#if _OPENMP
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < MyComputedBlocks.size(); b++) {
            int offset_i = MyComputedBlocks[b]->get_offset_i();
            int offset_j = MyComputedBlocks[b]->get_offset_j();
            if (!(symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
                MyComputedBlocks[b]->add_mvprod_row_major(in + (offset_i - local_offset) * mu, temp.data() + offset_j * mu, mu, 'T', 'T');
            }
        }

#if _OPENMP
#    pragma omp critical
#endif
        Blas<T>::axpy(&(global_size_rhs), &da, temp.data(), &incx, out, &incy);
    }

    MPI_Allreduce(MPI_IN_PLACE, out, this->nc * mu, wrapper_mpi<T>::mpi_type(), MPI_SUM, comm);
}

template <typename T>
void HMatrix<T>::local_to_global_target(const T *const in, T *const out, const int &mu) const {
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

template <typename T>
void HMatrix<T>::local_to_global_source(const T *const in, T *const out, const int &mu) const {
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

template <typename T>
void HMatrix<T>::mymvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
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

template <typename T>
void HMatrix<T>::mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

    if (this->symmetry == 'S' || this->symmetry == 'H') {
        this->mymvprod_local_to_local(in, out, mu, work);
        return;
    }

    double time      = MPI_Wtime();
    bool need_delete = false;
    if (work == nullptr) {
        work        = new T[(this->nc + local_size_source * sizeWorld) * mu];
        need_delete = true;
    }

    std::fill(out, out + local_size_source * mu, 0);
    int incx(1), incy(1);
    int global_size_rhs = this->nc * mu;
    T da(1);

    std::fill(work, work + this->nc * mu, 0);
    T *rbuf = work + this->nc * mu;

    // Contribution champ lointain
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<T> temp(this->nc * mu, 0);
#if _OPENMP
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < MyComputedBlocks.size(); b++) {
            int offset_i = MyComputedBlocks[b]->get_offset_i();
            int offset_j = MyComputedBlocks[b]->get_offset_j();
            if (!(symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
                MyComputedBlocks[b]->add_mvprod_row_major(in + (offset_i - local_offset) * mu, temp.data() + offset_j * mu, mu, 'T', 'T');
            }
        }

#if _OPENMP
#    pragma omp critical
#endif
        Blas<T>::axpy(&(global_size_rhs), &da, temp.data(), &incx, work, &incy);
    }

    std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
    std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        scounts[i] = (cluster_tree_s->get_masteroffset(i).second) * mu;
        rcounts[i] = (local_size_source)*mu;
        if (i > 0) {
            sdispls[i] = sdispls[i - 1] + scounts[i - 1];
            rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
        }
    }

    MPI_Alltoallv(work, &(scounts[0]), &(sdispls[0]), wrapper_mpi<T>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<T>::mpi_type(), comm);

    for (int i = 0; i < sizeWorld; i++)
        std::transform(out, out + local_size_source * mu, rbuf + rdispls[i], out, std::plus<T>());

    if (need_delete) {
        delete[] work;
        work = nullptr;
    }
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T>
void HMatrix<T>::mvprod_global_to_global(const T *const in, T *const out, const int &mu) const {
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

template <typename T>
void HMatrix<T>::mvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
    double time      = MPI_Wtime();
    bool need_delete = false;
    if (work == nullptr) {
        work        = new T[this->nc * mu];
        need_delete = true;
    }

    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

    if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
        throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }
    if (mu == 1) {
        std::vector<T> in_perm(local_size_source), out_perm(local_size);

        // local permutation
        if (use_permutation) {
            // permutation
            this->local_source_to_local_cluster(in, in_perm.data());

            // prod
            mymvprod_local_to_local(in_perm.data(), out_perm.data(), 1, work);

            // permutation
            this->local_cluster_to_local_target(out_perm.data(), out, comm);

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

template <typename T>
void HMatrix<T>::mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu) const {
    double time = MPI_Wtime();
    if (this->symmetry == 'S') {
        this->mvprod_global_to_global(in, out, mu);
        return;
    } else if (this->symmetry == 'H') {
        std::vector<T> in_conj(in, in + nr * mu);
        conj_if_complex(in_conj.data(), nr * mu);
        this->mvprod_global_to_global(in_conj.data(), out, mu);
        conj_if_complex(out, mu * nc);
        return;
    }
    if (mu == 1) {

        if (use_permutation) {
            std::vector<T> in_perm(nr), out_perm(nc);

            // permutation
            this->target_to_cluster_permutation(in, in_perm.data());

            mymvprod_transp_local_to_global(in_perm.data() + local_offset, out_perm.data(), 1);

            // permutation
            this->cluster_to_source_permutation(out_perm.data(), out);
        } else {
            mymvprod_transp_local_to_global(in + local_offset, out, 1);
        }

    } else {

        std::vector<T> out_perm(mu * nc);
        std::vector<T> in_perm(local_size * mu + mu * nc);
        std::vector<T> buffer(nr);

        for (int i = 0; i < mu; i++) {
            // Permutation
            if (use_permutation) {
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

        if (symmetry == 'H') {
            conj_if_complex(in_perm.data(), local_size * mu);
        }

        mymvprod_transp_local_to_global(in_perm.data(), in_perm.data() + local_size * mu, mu);

        for (int i = 0; i < mu; i++) {
            if (use_permutation) {
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

        if (symmetry == 'H') {
            conj_if_complex(out, nc * mu);
        }
    }
    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T>
void HMatrix<T>::mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
    double time           = MPI_Wtime();
    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
    if (this->symmetry == 'S') {
        this->mvprod_local_to_local(in, out, mu);
        return;
    } else if (this->symmetry == 'H') {
        std::vector<T> in_conj(in, in + local_size * mu);
        conj_if_complex(in_conj.data(), local_size * mu);
        this->mvprod_local_to_local(in_conj.data(), out, mu);
        conj_if_complex(out, mu * local_size_source);
        return;
    }
    bool need_delete = false;
    if (work == nullptr) {
        work        = new T[(this->nc + sizeWorld * this->get_source_cluster()->get_local_size()) * mu];
        need_delete = true;
    }

    if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
        throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
    }

    if (mu == 1) {
        std::vector<T> in_perm(local_size), out_perm(local_size_source);

        // local permutation
        if (use_permutation) {
            this->local_target_to_local_cluster(in, in_perm.data());

            // prod
            mymvprod_transp_local_to_local(in_perm.data(), out_perm.data(), 1, work);

            // permutation
            this->local_cluster_to_local_source(out_perm.data(), out, comm);

        } else {
            mymvprod_transp_local_to_local(in, out, 1, work);
        }

    } else {

        std::vector<T> in_perm(local_size * mu);
        std::vector<T> out_perm(local_size_source * mu);
        std::vector<T> buffer(std::max(local_size_source, local_size));

        for (int i = 0; i < mu; i++) {
            // local permutation
            if (use_permutation) {
                this->local_target_to_local_cluster(in + i * local_size, buffer.data());

                // Transpose
                for (int j = 0; j < local_size; j++) {
                    in_perm[i + j * mu] = buffer[j];
                }
            } else {
                // Transpose
                for (int j = 0; j < local_size; j++) {
                    in_perm[i + j * mu] = in[j + i * local_size];
                }
            }
        }

        if (symmetry == 'H') {
            conj_if_complex(in_perm.data(), local_size_source * mu);
        }

        mymvprod_transp_local_to_local(in_perm.data(), out_perm.data(), mu, work);

        for (int i = 0; i < mu; i++) {
            if (use_permutation) {
                // Tranpose
                for (int j = 0; j < local_size_source; j++) {
                    buffer[j] = out_perm[i + j * mu];
                }

                // local permutation
                this->local_cluster_to_local_source(buffer.data(), out + i * local_size_source);
            } else {
                // Tranpose
                for (int j = 0; j < local_size_source; j++) {
                    out[j + i * local_size_source] = out_perm[i + j * mu];
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

template <typename T>
void HMatrix<T>::mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const {
    std::fill(out, out + local_size * mu, 0);

    // Contribution champ lointain
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<T> temp(local_size * mu, 0);
        // To localize the rhs with multiple rhs, it is transpose. So instead of A*B, we do transpose(B)*transpose(A)
        char transb = 'T';
        // In case of a hermitian matrix, the rhs is conjugate transpose
        if (symmetry == 'H') {
            transb = 'C';
        }
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

template <typename T>
void HMatrix<T>::source_to_cluster_permutation(const T *const in, T *const out) const {
    global_to_cluster(cluster_tree_s.get(), in, out);
}

template <typename T>
void HMatrix<T>::target_to_cluster_permutation(const T *const in, T *const out) const {
    global_to_cluster(cluster_tree_t.get(), in, out);
}

template <typename T>
void HMatrix<T>::cluster_to_target_permutation(const T *const in, T *const out) const {
    cluster_to_global(cluster_tree_t.get(), in, out);
}

template <typename T>
void HMatrix<T>::cluster_to_source_permutation(const T *const in, T *const out) const {
    cluster_to_global(cluster_tree_s.get(), in, out);
}

template <typename T>
void HMatrix<T>::local_target_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
    local_to_local_cluster(cluster_tree_t.get(), in, out, comm);
}

template <typename T>
void HMatrix<T>::local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
    local_to_local_cluster(cluster_tree_s.get(), in, out, comm);
}

template <typename T>
void HMatrix<T>::local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm) const {
    local_cluster_to_local(cluster_tree_t.get(), in, out, comm);
}

template <typename T>
void HMatrix<T>::local_cluster_to_local_source(const T *const in, T *const out, MPI_Comm comm) const {
    local_cluster_to_local(cluster_tree_s.get(), in, out, comm);
}

template <typename T>
std::vector<T> HMatrix<T>::operator*(const std::vector<T> &x) const {
    assert(x.size() == nc);
    std::vector<T> result(nr, 0);
    mvprod_global_to_global(x.data(), result.data(), 1);
    return result;
}

template <typename T>
double HMatrix<T>::compression_ratio() const {

    double my_compressed_size = 0.;
    double uncompressed_size  = ((long int)this->nr) * this->nc;
    double nr_b, nc_b, rank;

    for (int j = 0; j < MyFarFieldMats.size(); j++) {
        nr_b = MyFarFieldMats[j]->nb_rows();
        nc_b = MyFarFieldMats[j]->nb_cols();
        rank = MyFarFieldMats[j]->rank_of();
        my_compressed_size += rank * (nr_b + nc_b);
    }

    for (int j = 0; j < MyNearFieldMats.size(); j++) {
        nr_b = MyNearFieldMats[j]->nb_rows();
        nc_b = MyNearFieldMats[j]->nb_cols();
        if (MyNearFieldMats[j]->get_offset_i() == MyNearFieldMats[j]->get_offset_j() && this->get_symmetry_type() != 'N' && nr_b == nc_b) {
            my_compressed_size += (nr_b * (nc_b + 1)) / 2;
        } else {
            my_compressed_size += nr_b * nc_b;
        }
    }

    double compressed_size = 0;
    MPI_Allreduce(&my_compressed_size, &compressed_size, 1, MPI_DOUBLE, MPI_SUM, comm);

    return uncompressed_size / compressed_size;
}

template <typename T>
double HMatrix<T>::space_saving() const {

    double my_compressed_size = 0.;
    double uncompressed_size  = ((long int)this->nr) * this->nc;
    double nr_b, nc_b, rank;

    for (int j = 0; j < MyFarFieldMats.size(); j++) {
        nr_b = MyFarFieldMats[j]->nb_rows();
        nc_b = MyFarFieldMats[j]->nb_cols();
        rank = MyFarFieldMats[j]->rank_of();
        my_compressed_size += rank * (nr_b + nc_b);
    }

    for (int j = 0; j < MyNearFieldMats.size(); j++) {
        nr_b = MyNearFieldMats[j]->nb_rows();
        nc_b = MyNearFieldMats[j]->nb_cols();
        if (MyNearFieldMats[j]->get_offset_i() == MyNearFieldMats[j]->get_offset_j() && this->get_symmetry_type() != 'N' && nr_b == nc_b) {
            my_compressed_size += (nr_b * (nc_b + 1)) / 2;
        } else {
            my_compressed_size += nr_b * nc_b;
        }
    }

    double compressed_size = 0;
    MPI_Allreduce(&my_compressed_size, &compressed_size, 1, MPI_DOUBLE, MPI_SUM, comm);

    return 1 - compressed_size / uncompressed_size;
}

template <typename T>
void HMatrix<T>::print_infos() const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

    if (rankWorld == 0) {
        for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
            std::cout << it->first << "\t" << it->second << std::endl;
        }
        std::cout << std::endl;
    }
}

template <typename T>
void HMatrix<T>::save_infos(const std::string &outputname, std::ios_base::openmode mode, const std::string &sep) const {
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
            std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
        }
    }
}

template <typename T>
void HMatrix<T>::save_plot(const std::string &outputname) const {

    std::ofstream outputfile((outputname + "_" + NbrToStr(rankWorld) + ".csv").c_str());

    if (outputfile) {
        outputfile << nr << "," << nc << std::endl;
        for (typename std::vector<SubMatrix<T> *>::const_iterator it = MyNearFieldMats.begin(); it != MyNearFieldMats.end(); ++it) {
            outputfile << (*it)->get_offset_i() << "," << (*it)->get_ir().size() << "," << (*it)->get_offset_j() << "," << (*it)->get_ic().size() << "," << -1 << std::endl;
        }
        for (typename std::vector<LowRankMatrix<T> *>::const_iterator it = MyFarFieldMats.begin(); it != MyFarFieldMats.end(); ++it) {
            outputfile << (*it)->get_offset_i() << "," << (*it)->get_ir().size() << "," << (*it)->get_offset_j() << "," << (*it)->get_ic().size() << "," << (*it)->rank_of() << std::endl;
        }
        outputfile.close();
    } else {
        std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
    }
}

template <typename T>
underlying_type<T> Frobenius_absolute_error(const HMatrix<T> &B, const VirtualGenerator<T> &A) {
    underlying_type<T> myerr = 0;
    for (int j = 0; j < B.MyFarFieldMats.size(); j++) {
        underlying_type<T> test = Frobenius_absolute_error(*(B.MyFarFieldMats[j]), A);
        myerr += std::pow(test, 2);
    }

    underlying_type<T> err = 0;
    MPI_Allreduce(&myerr, &err, 1, wrapper_mpi<T>::mpi_underlying_type(), MPI_SUM, B.comm);

    return std::sqrt(err);
}

template <typename T>
Matrix<T> HMatrix<T>::get_local_dense() const {
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
    for (int l = 0; l < MyFarFieldMats.size(); l++) {
        const LowRankMatrix<T> &lmat = *(MyFarFieldMats[l]);
        int local_nr                 = lmat.nb_rows();
        int local_nc                 = lmat.nb_cols();
        int offset_i                 = lmat.get_offset_i();
        int offset_j                 = lmat.get_offset_j();
        Matrix<T> FarFielBlock(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int k = 0; k < local_nc; k++) {
            std::copy_n(&(FarFielBlock(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
        }
    }
    return Dense;
}

template <typename T>
Matrix<T> HMatrix<T>::get_local_dense_perm() const {
    Matrix<T> Dense(local_size, nc);
    copy_local_dense_perm(Dense.data());
    return Dense;
}

template <typename T>
void HMatrix<T>::copy_local_dense_perm(T *ptr) const {
    if (!(cluster_tree_t->IsLocal())) {
        throw std::logic_error("[Htool error] Permutation is not local, get_local_dense_perm cannot be used"); // LCOV_EXCL_LINE
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

    for (int l = 0; l < MyFarFieldMats.size(); l++) {
        const LowRankMatrix<T> &lmat = *(MyFarFieldMats[l]);
        int local_nr                 = lmat.nb_rows();
        int local_nc                 = lmat.nb_cols();
        int offset_i                 = lmat.get_offset_i();
        int offset_j                 = lmat.get_offset_j();
        Matrix<T> FarFielBlock(local_nr, local_nc);
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

template <typename T>
std::vector<T> HMatrix<T>::get_local_diagonal(bool permutation) const {
    std::vector<T> diagonal(local_size, 0);
    copy_local_diagonal(diagonal.data(), permutation);
    return diagonal;
}

template <typename T>
void HMatrix<T>::copy_local_diagonal(T *ptr, bool permutation) const {
    if (!(cluster_tree_t->IsLocal()) && permutation) {
        throw std::logic_error("[Htool error] Permutation is not local, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
    }
    if (cluster_tree_t != cluster_tree_s) {
        throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
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

template <typename T>
Matrix<T> HMatrix<T>::get_local_interaction(bool permutation) const {
    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
    Matrix<T> local_interaction(local_size, local_size_source);
    copy_local_interaction(local_interaction.data(), permutation);
    return local_interaction;
}

template <typename T>
Matrix<T> HMatrix<T>::get_local_diagonal_block(bool permutation) const {
    int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
    Matrix<T> diagonal_block(local_size, local_size_source);
    copy_local_diagonal_block(diagonal_block.data(), permutation);
    return diagonal_block;
}

template <typename T>
void HMatrix<T>::copy_local_interaction(T *ptr, bool permutation) const {
    if ((!(cluster_tree_t->IsLocal()) || !(cluster_tree_s->IsLocal())) && permutation) {
        throw std::logic_error("[Htool error] Permutation is not local, get_local_interaction cannot be used"); // LCOV_EXCL_LINE
    }

    int local_offset_source = cluster_tree_s->get_masteroffset(rankWorld).first;
    int local_size_source   = cluster_tree_s->get_masteroffset(rankWorld).second;
    // Internal dense blocks
    for (int i = 0; i < MyDiagNearFieldMats.size(); i++) {
        const SubMatrix<T> &submat = *(MyDiagNearFieldMats[i]);
        int local_nr               = submat.nb_rows();
        int local_nc               = submat.nb_cols();
        int offset_i               = submat.get_offset_i() - local_offset;
        int offset_j               = submat.get_offset_j() - local_offset_source;
        for (int i = 0; i < local_nc; i++) {
            std::copy_n(&(submat(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
        }
    }

    // Internal compressed block
    for (int i = 0; i < MyDiagFarFieldMats.size(); i++) {
        const LowRankMatrix<T> &lmat = *(MyDiagFarFieldMats[i]);
        int local_nr                 = lmat.nb_rows();
        int local_nc                 = lmat.nb_cols();
        int offset_i                 = lmat.get_offset_i() - local_offset;
        int offset_j                 = lmat.get_offset_j() - local_offset_source;
        ;
        Matrix<T> FarFielBlock(local_nr, local_nc);
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

template <typename T>
void HMatrix<T>::copy_local_diagonal_block(T *ptr, bool permutation) const {
    if (cluster_tree_t != cluster_tree_s) {
        throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal_block cannot be used"); // LCOV_EXCL_LINE
    }
    copy_local_interaction(ptr, permutation);
}

template <typename T>
std::pair<int, int> HMatrix<T>::get_max_size_blocks() const {
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

template <typename T>
void HMatrix<T>::apply_dirichlet(const std::vector<int> &boundary) {
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
