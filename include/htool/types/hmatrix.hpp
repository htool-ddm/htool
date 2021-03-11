#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if _OPENMP
#    include <omp.h>
#endif

#include "../blocks/blocks.hpp"
#include "../clustering/cluster.hpp"
#include "../misc/misc.hpp"
#include "../misc/parametres.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "matrix.hpp"
#include "multihmatrix.hpp"
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
template <typename T, template <typename, typename> class MultiLowRankMatrix, typename ClusterImpl, template <typename> class AdmissibleCondition>
class MultiHMatrix;
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class HMatrix;

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
double Frobenius_absolute_error(const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &B, const IMatrix<T> &A);

// Class
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class HMatrix : public Parametres {

  private:
    // Data members
    int nr;
    int nc;
    int reqrank;
    int local_size;
    int local_offset;
    char symmetry;
    char UPLO;

    std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_s;
    std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_t;

    std::unique_ptr<Block<ClusterImpl, AdmissibleCondition>> BlockTree;

    std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> MyFarFieldMats;
    std::vector<std::unique_ptr<SubMatrix<T>>> MyNearFieldMats;
    std::vector<LowRankMatrix<T, ClusterImpl> *> MyDiagFarFieldMats;
    std::vector<SubMatrix<T> *> MyDiagNearFieldMats;
    std::vector<LowRankMatrix<T, ClusterImpl> *> MyStrictlyDiagFarFieldMats;
    std::vector<SubMatrix<T> *> MyStrictlyDiagNearFieldMats;

    mutable std::map<std::string, std::string> infos;

    const MPI_Comm comm;
    int rankWorld, sizeWorld;

    // Internal methods
    void ComputeBlocks(IMatrix<T> &mat, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs);
    void ComputeSymBlocks(IMatrix<T> &mat, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs);
    bool UpdateBlocks(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &, std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &);
    bool UpdateSymBlocks(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &, std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &);
    void AddNearFieldMat(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, std::vector<std::unique_ptr<SubMatrix<T>>> &);
    void AddFarFieldMat(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs, std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &, const int &reqrank = -1);
    void ComputeInfos(const std::vector<double> &mytimes);

    // Check arguments
    void check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs, char symmetry, char UPLO) const;
    void check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<int> &tabs, char symmetry, char UPLO) const;
    void check_arguments_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, char symmetry, char UPLO) const;

    // Friends
    template <typename U, template <typename, typename> class MultiLowRankMatrix, typename ClusterImplU, template <typename> class AdmissibleConditionU>
    friend class MultiHMatrix;

  public:
    // Special constructor for hand-made build (for MultiHMatrix for example)
    HMatrix(int nr0, int nc0, const std::shared_ptr<Cluster<ClusterImpl>> &cluster_tree_t0, const std::shared_ptr<Cluster<ClusterImpl>> &cluster_tree_s0, char symmetry0 = 'N', char UPLO = 'N', const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(nr0), nc(nc0), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0), symmetry(symmetry0), comm(comm0){};

    // Build
    void build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs); // To be used with two different clusters

    // Full constructor
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Constructor without radius
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<int> &tabs, const std::vector<double> &gs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Constructor without mass
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Constructor without tab
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<double> &gs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Constructor without radius, tab and mass
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<R3> &xs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Symmetry build
    void build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt); // To be used with one different clusters

    // Full symmetry constructor
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

    // Symmetry constructor without radius
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<double> &gt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

    // Constructor without mass
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

    // Constructor without tab
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<double> &gt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

    // Constructor without radius, tab and mass
    HMatrix(IMatrix<T> &, const std::vector<R3> &xt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

    // Build with precomputed clusters
    void build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<int> &tabs); // To be used with two different clusters

    // Full constructor with precomputed clusters
    HMatrix(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::shared_ptr<Cluster<ClusterImpl>> &s, const std::vector<R3> &xs, const std::vector<int> &tabs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Constructor without tab and with precomputed clusters
    HMatrix(IMatrix<T> &, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::shared_ptr<Cluster<ClusterImpl>> &s, const std::vector<R3> &xs, const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with two different clusters

    // Symmetry build with precomputed cluster
    void build(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::vector<int> &tabt); // To be used with one different clusters

    // Full symmetry constructor with precomputed cluster
    HMatrix(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::vector<int> &tabt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

    // Constructor without tab and with precomputed cluster
    HMatrix(IMatrix<T> &, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, char Symmetry = 'N', char UPLO = 'N', const int &reqrank = -1, const MPI_Comm comm = MPI_COMM_WORLD); // To be used with one different clusters

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

    const Cluster<ClusterImpl> &get_cluster_tree_t() const { return *(cluster_tree_t.get()); }
    const Cluster<ClusterImpl> &get_cluster_tree_s() const { return *(cluster_tree_s.get()); }
    std::vector<std::pair<int, int>> get_MasterOffset_t() const { return cluster_tree_t->get_masteroffset(); }
    std::vector<std::pair<int, int>> get_MasterOffset_s() const { return cluster_tree_s->get_masteroffset(); }
    std::pair<int, int> get_MasterOffset_t(int i) const { return cluster_tree_t->get_masteroffset(i); }
    std::pair<int, int> get_MasterOffset_s(int i) const { return cluster_tree_s->get_masteroffset(i); }
    const std::vector<int> &get_permt() const { return cluster_tree_t->get_perm(); }
    const std::vector<int> &get_perms() const { return cluster_tree_s->get_perm(); }
    int get_permt(int i) const { return cluster_tree_t->get_perm(i); }
    int get_perms(int i) const { return cluster_tree_s->get_perm(i); }
    const std::vector<std::unique_ptr<SubMatrix<T>>> &get_MyNearFieldMats() const { return MyNearFieldMats; }
    const std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &get_MyFarFieldMats() const { return MyFarFieldMats; }
    const std::vector<SubMatrix<T> *> &get_MyDiagNearFieldMats() const { return MyDiagNearFieldMats; }
    const std::vector<LowRankMatrix<T, ClusterImpl> *> &get_MyDiagFarFieldMats() const { return MyDiagFarFieldMats; }
    const std::vector<SubMatrix<T> *> &get_MyStrictlyDiagNearFieldMats() const { return MyStrictlyDiagNearFieldMats; }
    const std::vector<LowRankMatrix<T, ClusterImpl> *> &get_MyStrictlyDiagFarFieldMats() const { return MyStrictlyDiagFarFieldMats; }

    // Infos
    const std::map<std::string, std::string> &get_infos() const { return infos; }
    std::string get_infos(const std::string &key) const { return infos[key]; }
    void add_info(const std::string &keyname, const std::string &value) const { infos[keyname] = value; }
    void print_infos() const;
    void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const;
    void save_plot(const std::string &outputname) const;
    double compression() const; // 1- !!!
    friend double Frobenius_absolute_error<T, LowRankMatrix, ClusterImpl>(const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &B, const IMatrix<T> &A);

    // Mat vec prod
    void mvprod_global(const T *const in, T *const out, const int &mu = 1) const;
    void mvprod_local(const T *const in, T *const out, T *const work, const int &mu) const;
    void mymvprod_local(const T *const in, T *const out, const int &mu) const;
    void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const;
    std::vector<T> operator*(const std::vector<T> &x) const;
    Matrix<T> operator*(const Matrix<T> &x) const;

    // Permutations
    template <typename U>
    void source_to_cluster_permutation(const U *const in, U *const out) const;
    template <typename U>
    void cluster_to_target_permutation(const U *const in, U *const out) const;

    // local to global
    void local_to_global(const T *const in, T *const out, const int &mu) const;

    // Convert
    Matrix<T> to_dense() const;
    Matrix<T> to_local_dense() const;
    Matrix<T> to_dense_perm() const;

    // Apply Dirichlet condition
    void apply_dirichlet(const std::vector<int> &boundary);
};

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs, char symmetry, char UPLO) const {
    if (!(mat.nb_rows() == tabt.size() && mat.nb_cols() == tabs.size()
          && mat.nb_rows() == ndofperelt * xt.size() && mat.nb_cols() == ndofperelt * xs.size()
          && mat.nb_rows() == ndofperelt * rt.size() && mat.nb_cols() == ndofperelt * rs.size()
          && mat.nb_rows() == ndofperelt * gt.size() && mat.nb_cols() == ndofperelt * gs.size()
          && (symmetry == 'N' || symmetry == 'H' || symmetry == 'S')
          && (UPLO == 'N' || UPLO == 'L' || UPLO == 'U')
          && ((symmetry == 'N' && UPLO == 'N') || (symmetry != 'N' && UPLO != 'N'))
          && ((symmetry == 'H' && is_complex<T>()) || symmetry != 'H'))) {
        throw std::invalid_argument("[Htool error] Invalid arguments for HMatrix");
    }
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::check_arguments(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<int> &tabs, char symmetry, char UPLO) const {
    if (!(mat.nb_rows() == tabt.size() && mat.nb_cols() == tabs.size()
          && mat.nb_rows() == ndofperelt * xt.size() && mat.nb_cols() == ndofperelt * xs.size()
          && (symmetry == 'N' || symmetry == 'H' || symmetry == 'S')
          && (UPLO == 'N' || UPLO == 'L' || UPLO == 'U')
          && ((symmetry == 'N' && UPLO == 'N') || (symmetry != 'N' && UPLO != 'N'))
          && ((symmetry == 'H' && is_complex<T>()) || symmetry != 'H'))) {
        throw std::invalid_argument("[Htool error] Invalid arguments for HMatrix");
    }
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::check_arguments_sym(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, char symmetry, char UPLO) const {
    this->check_arguments(mat, xt, rt, tabt, gt, xt, rt, tabt, gt, symmetry, UPLO);
}

// build
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(4), maxtime(4), meantime(4);

    // Construction arbre des paquets
    double time    = MPI_Wtime();
    cluster_tree_t = std::make_shared<ClusterImpl>(); // target
    cluster_tree_s = std::make_shared<ClusterImpl>(); // source
    cluster_tree_t->build(xt, rt, tabt, gt, -1, comm);
    cluster_tree_s->build(xs, rs, tabs, gs, -1, comm);

    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    mytimes[0] = MPI_Wtime() - time;

    // Construction arbre des blocs
    time = MPI_Wtime();
    this->BlockTree.reset(new Block<ClusterImpl, AdmissibleCondition>(*cluster_tree_t, *cluster_tree_s));
    this->BlockTree->build(UPLO, comm);
    mytimes[1] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, tabt, xs, tabs);
    mytimes[2] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

// Full constructor
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    this->check_arguments(mat, xt, rt, tabt, gt, xs, rs, tabs, gs, this->symmetry, this->UPLO);
    this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs);
}

// Constructor without rt and rs
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<int> &tabs, const std::vector<double> &gs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    this->check_arguments(mat, xt, std::vector<double>(xt.size(), 0), tabt, gt, xs, std::vector<double>(xs.size(), 0), tabs, gs, this->symmetry, this->UPLO);
    this->build(mat, xt, std::vector<double>(xt.size(), 0), tabt, gt, xs, std::vector<double>(xs.size(), 0), tabs, gs);
}

// Constructor without gt and gs
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    this->check_arguments(mat, xt, rt, tabt, std::vector<double>(xt.size(), 1), xs, rs, tabs, std::vector<double>(xs.size(), 1), this->symmetry, this->UPLO);
    this->build(mat, xt, rt, tabt, std::vector<double>(xt.size(), 1), xs, rs, tabs, std::vector<double>(xs.size(), 1));
}

// Constructor without tabt and tabs
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<double> &gs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    std::vector<int> tabt(ndofperelt * xt.size()), tabs(ndofperelt * xs.size());
    std::iota(tabt.begin(), tabt.end(), int(0));
    std::iota(tabs.begin(), tabs.end(), int(0));
    this->check_arguments(mat, xt, rt, tabt, gt, xs, rs, tabs, gs, this->symmetry, this->UPLO);
    this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs);
}

// Constructor without radius, mass and tab
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<R3> &xs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    std::vector<int> tabt(ndofperelt * xt.size()), tabs(ndofperelt * xs.size());
    std::iota(tabt.begin(), tabt.end(), int(0));
    std::iota(tabs.begin(), tabs.end(), int(0));
    this->check_arguments(mat, xt, std::vector<double>(xt.size(), 0), tabt, std::vector<double>(xt.size(), 1), xs, std::vector<double>(xs.size(), 0), tabs, std::vector<double>(xs.size(), 1), this->symmetry, this->UPLO);
    this->build(mat, xt, std::vector<double>(xt.size(), 0), tabt, std::vector<double>(xt.size(), 1), xs, std::vector<double>(xs.size(), 0), tabs, std::vector<double>(xs.size(), 1));
}

// Symmetry build
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(4), maxtime(4), meantime(4);

    // Construction arbre des paquets
    double time    = MPI_Wtime();
    cluster_tree_t = std::make_shared<ClusterImpl>();
    cluster_tree_s = cluster_tree_t;
    cluster_tree_t->build(xt, rt, tabt, gt, -1, comm);
    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    mytimes[0] = MPI_Wtime() - time;

    // Construction arbre des blocs
    time = MPI_Wtime();

    this->BlockTree.reset(new Block<ClusterImpl, AdmissibleCondition>(*cluster_tree_t, *cluster_tree_s));
    this->BlockTree->build(UPLO, comm);

    mytimes[1] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, tabt, xt, tabt);
    mytimes[2] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

// Full symmetry constructor
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, char symmetry0, char UPLO0, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    this->check_arguments_sym(mat, xt, rt, tabt, gt, this->symmetry, this->UPLO);
    this->build(mat, xt, rt, tabt, gt);
}

// Symmetry constructor without rt
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<double> &gt, char symmetry0, char UPLO0, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    this->check_arguments_sym(mat, xt, std::vector<double>(xt.size(), 0), tabt, gt, this->symmetry, this->UPLO);
    this->build(mat, xt, std::vector<double>(xt.size(), 0), tabt, gt);
}

// Symmetry constructor without tabt
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<double> &gt, char symmetry0, char UPLO0, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    std::vector<int> tabt(ndofperelt * xt.size());
    std::iota(tabt.begin(), tabt.end(), int(0));
    this->check_arguments_sym(mat, xt, rt, tabt, gt, this->symmetry, this->UPLO);
    this->build(mat, xt, rt, tabt, gt);
}

// Symmetry constructor without gt
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, char symmetry0, char UPLO0, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    this->check_arguments_sym(mat, xt, rt, tabt, std::vector<double>(xt.size(), 1), this->symmetry, this->UPLO);
    this->build(mat, xt, rt, tabt, std::vector<double>(xt.size(), 1));
}

// Symmetry constructor without rt, tabt and gt
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::vector<R3> &xt, char symmetry0, char UPLO0, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), comm(comm0) {
    std::vector<int> tabt(ndofperelt * xt.size());
    std::iota(tabt.begin(), tabt.end(), int(0));
    this->check_arguments_sym(mat, xt, std::vector<double>(xt.size(), 0), tabt, std::vector<double>(xt.size(), 1), this->symmetry, this->UPLO);
    this->build(mat, xt, std::vector<double>(xt.size(), 0), tabt, std::vector<double>(xt.size(), 1));
}

// build with input cluster
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::build(IMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::vector<R3> &xs, const std::vector<int> &tabs) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> mytimes(4), maxtime(4), meantime(4);

    // Construction arbre des paquets
    double time = MPI_Wtime();

    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    mytimes[0] = MPI_Wtime() - time;

    // Construction arbre des blocs
    time = MPI_Wtime();
    this->BlockTree.reset(new Block<ClusterImpl, AdmissibleCondition>(*cluster_tree_t, *cluster_tree_s));
    this->BlockTree->build(UPLO, comm);
    mytimes[1] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, tabt, xs, tabs);
    mytimes[2] = MPI_Wtime() - time;

    // Infos
    ComputeInfos(mytimes);
}

// Full constructor with precomputed clusters
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::vector<int> &tabt, const std::shared_ptr<Cluster<ClusterImpl>> &s, const std::vector<R3> &xs, const std::vector<int> &tabs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0), comm(comm0) {
    this->check_arguments(mat, xt, tabt, xs, tabs, this->symmetry, this->UPLO);
    this->build(mat, xt, tabt, xs, tabs);
}

// Constructor without tabt and tabs
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::shared_ptr<Cluster<ClusterImpl>> &s, const std::vector<R3> &xs, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry('N'), UPLO('N'), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0), comm(comm0) {
    std::vector<int> tabt(ndofperelt * xt.size()), tabs(ndofperelt * xs.size());
    std::iota(tabt.begin(), tabt.end(), int(0));
    std::iota(tabs.begin(), tabs.end(), int(0));
    this->check_arguments(mat, xt, tabt, xs, tabs, this->symmetry, this->UPLO);
    this->build(mat, xt, tabt, xs, tabs);
}

// Full symmetry constructor
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, const std::vector<int> &tabt, char symmetry0, char UPLO0, const int &reqrank0, MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0), comm(comm0) {
    this->check_arguments(mat, xt, tabt, xt, tabt, this->symmetry, this->UPLO);
    this->build(mat, xt, tabt, xt, tabt);
}

// Symmetry constructor without tabt
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::HMatrix(IMatrix<T> &mat, const std::shared_ptr<Cluster<ClusterImpl>> &t, const std::vector<R3> &xt, char symmetry0, char UPLO0, const int &reqrank0, const MPI_Comm comm0) : nr(mat.nb_rows()), nc(mat.nb_cols()), symmetry(symmetry0), UPLO(UPLO0), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0), comm(comm0) {
    std::vector<int> tabt(ndofperelt * xt.size());
    std::iota(tabt.begin(), tabt.end(), int(0));
    this->check_arguments(mat, xt, tabt, xt, tabt, this->symmetry, this->UPLO);
    this->build(mat, xt, tabt, xt, tabt);
}

// Compute blocks recursively
// TODO: recursivity -> stack for compute blocks
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::ComputeBlocks(IMatrix<T> &mat, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs) {
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp parallel
#endif
    {
        std::vector<std::unique_ptr<SubMatrix<T>>> MyNearFieldMats_local;
        std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> MyFarFieldMats_local;
        std::vector<Block<ClusterImpl, AdmissibleCondition> *> local_tasks = BlockTree->get_local_tasks();
#if _OPENMP && !defined(PYTHON_INTERFACE)
#    pragma omp for schedule(guided)
#endif
        for (int p = 0; p < local_tasks.size(); p++) {
            bool not_pushed;
            if (symmetry == 'H' || symmetry == 'S') {
                not_pushed = UpdateSymBlocks(mat, *(local_tasks[p]), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
            } else {
                not_pushed = UpdateBlocks(mat, *(local_tasks[p]), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
bool HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::UpdateBlocks(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local, std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &MyFarFieldMats_local) {
    if (task.IsAdmissible()) {
        AddFarFieldMat(mat, task, xt, tabt, xs, tabs, MyFarFieldMats_local, reqrank);
        if (MyFarFieldMats_local.back()->rank_of() != -1) {
            return false;
        } else {
            MyFarFieldMats_local.pop_back();
            // AddNearFieldMat(mat,task,MyNearFieldMats_local);
            // return false;
        }
    }
    // else {
    // 	AddNearFieldMat(mat,task,MyNearFieldMats_local);
    // 	return false;
    // }

    int bsize                     = task.get_size();
    const Cluster<ClusterImpl> &t = task.get_target_cluster();
    const Cluster<ClusterImpl> &s = task.get_source_cluster();

    if (s.IsLeaf()) {
        if (t.IsLeaf()) {
            return true;
        } else {

            std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
            for (int p = 0; p < t.get_nb_sons(); p++) {
                task.build_son(t.get_son(p), s);

                Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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
                Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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
                    Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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
                    Blocks_not_pushed[p] = UpdateBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
bool HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::UpdateSymBlocks(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local, std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &MyFarFieldMats_local) {

    if (task.IsAdmissible()) {

        AddFarFieldMat(mat, task, xt, tabt, xs, tabs, MyFarFieldMats_local, reqrank);
        if (MyFarFieldMats_local.back()->rank_of() != -1) {
            return false;
        } else {
            MyFarFieldMats_local.pop_back();
        }
    }

    int bsize                     = task.get_size();
    const Cluster<ClusterImpl> &t = task.get_target_cluster();
    const Cluster<ClusterImpl> &s = task.get_source_cluster();

    if (s.IsLeaf()) {
        if (t.IsLeaf()) {
            return true;
        } else {
            std::vector<bool> Blocks_not_pushed(t.get_nb_sons());
            for (int p = 0; p < t.get_nb_sons(); p++) {
                task.build_son(t.get_son(p), s);
                Blocks_not_pushed[p] = UpdateSymBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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
                Blocks_not_pushed[p] = UpdateSymBlocks(mat, task.get_son(p), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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
                    Blocks_not_pushed[p + l * t.get_nb_sons()] = UpdateSymBlocks(mat, task.get_son(p + l * t.get_nb_sons()), xt, tabt, xs, tabs, MyNearFieldMats_local, MyFarFieldMats_local);
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
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::AddNearFieldMat(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local) {

    const Cluster<ClusterImpl> &t = task.get_target_cluster();
    const Cluster<ClusterImpl> &s = task.get_source_cluster();

    MyNearFieldMats_local.emplace_back(new SubMatrix<T>(mat, std::vector<int>(cluster_tree_t->get_perm_start() + t.get_offset(), cluster_tree_t->get_perm_start() + t.get_offset() + t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start() + s.get_offset(), cluster_tree_s->get_perm_start() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset()));
}

// Build a low rank block
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::AddFarFieldMat(IMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const std::vector<R3> xt, const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int> tabs, std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>> &MyFarFieldMats_local, const int &reqrank) {

    const Cluster<ClusterImpl> &t = task.get_target_cluster();
    const Cluster<ClusterImpl> &s = task.get_source_cluster();

    MyFarFieldMats_local.emplace_back(new LowRankMatrix<T, ClusterImpl>(std::vector<int>(cluster_tree_t->get_perm_start() + t.get_offset(), cluster_tree_t->get_perm_start() + t.get_offset() + t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start() + s.get_offset(), cluster_tree_s->get_perm_start() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset(), reqrank));
    MyFarFieldMats_local.back()->build(mat, t, xt, tabt, s, xs, tabs);
}

// Compute infos
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::ComputeInfos(const std::vector<double> &mytime) {
    // 0 : cluster tree ; 1 : block tree ; 2 : compute blocks ;
    std::vector<double> maxtime(3), meantime(3);
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
    } else {
        MPI_Reduce(&(maxinfos[0]), &(maxinfos[0]), 4, MPI_INT, MPI_MAX, 0, comm);
        MPI_Reduce(&(mininfos[0]), &(mininfos[0]), 4, MPI_INT, MPI_MIN, 0, comm);
        MPI_Reduce(&(meaninfos[0]), &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
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
    MPI_Reduce(&(mytime[0]), &(maxtime[0]), 3, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&(mytime[0]), &(meantime[0]), 3, MPI_DOUBLE, MPI_SUM, 0, comm);

    meantime /= sizeWorld;

    infos["Cluster_mean"]    = NbrToStr(meantime[0]);
    infos["Cluster_max"]     = NbrToStr(maxtime[0]);
    infos["Block_tree_mean"] = NbrToStr(meantime[1]);
    infos["Block_tree_max"]  = NbrToStr(maxtime[1]);
    infos["Blocks_mean"]     = NbrToStr(meantime[2]);
    infos["Blocks_max"]      = NbrToStr(maxtime[2]);

    // Size
    infos["Source_size"]              = NbrToStr(this->nc);
    infos["Target_size"]              = NbrToStr(this->nr);
    infos["Dense_block_size_max"]     = NbrToStr(maxinfos[0]);
    infos["Dense_block_size_mean"]    = NbrToStr(meaninfos[0]);
    infos["Dense_block_size_min"]     = NbrToStr(mininfos[0]);
    infos["Low_rank_block_size_max"]  = NbrToStr(maxinfos[1]);
    infos["Low_rank_block_size_mean"] = NbrToStr(meaninfos[1]);
    infos["Low_rank_block_size_min"]  = NbrToStr(mininfos[1]);

    infos["Rank_max"]        = NbrToStr(maxinfos[2]);
    infos["Rank_mean"]       = NbrToStr(meaninfos[2]);
    infos["Rank_min"]        = NbrToStr(mininfos[2]);
    infos["Number_of_lrmat"] = NbrToStr(nlrmat);
    infos["Number_of_dmat"]  = NbrToStr(ndmat);
    infos["Compression"]     = NbrToStr(this->compression());
    infos["Local_size_max"]  = NbrToStr(maxinfos[3]);
    infos["Local_size_mean"] = NbrToStr(meaninfos[3]);
    infos["Local_size_min"]  = NbrToStr(mininfos[3]);

    infos["Number_of_MPI_tasks"] = NbrToStr(sizeWorld);
#if _OPENMP
    infos["Number_of_threads_per_tasks"] = NbrToStr(omp_get_max_threads());
    infos["Number_of_procs"]             = NbrToStr(sizeWorld * omp_get_max_threads());
#else
    infos["Number_of_procs"] = NbrToStr(sizeWorld);
#endif

    infos["Eta"]            = NbrToStr(GetEta());
    infos["Eps"]            = NbrToStr(GetEpsilon());
    infos["MinTargetDepth"] = NbrToStr(GetMinTargetDepth());
    infos["MinSourceDepth"] = NbrToStr(GetMinSourceDepth());
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::mymvprod_local(const T *const in, T *const out, const int &mu) const {

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
            const LowRankMatrix<T, ClusterImpl> &M = *(MyFarFieldMats[b]);
            int offset_i                           = M.get_offset_i();
            int offset_j                           = M.get_offset_j();

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
                const LowRankMatrix<T, ClusterImpl> &M = *(MyDiagFarFieldMats[b]);
                int offset_i                           = M.get_offset_j();
                int offset_j                           = M.get_offset_i();

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

// template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl, template<typename> class AdmissibleCondition>
// void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::local_to_global(const T* const in, T* const out, const int& mu) const{
// 	// Allgather
// 	std::vector<int> recvcounts(sizeWorld);
// 	std::vector<int>  displs(sizeWorld);
//
// 	displs[0] = 0;
//
// 	for (int i=0; i<sizeWorld; i++) {
// 		recvcounts[i] = (cluster_tree_t->get_masteroffset(i).second)*mu;
// 		if (i > 0)
// 			displs[i] = displs[i-1] + recvcounts[i-1];
// 	}
//
// 	MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out + (mu==1 ? 0 : mu*nc), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
//
//     //
//     if (mu!=1){
//         for (int i=0 ;i<mu;i++){
//             for (int j=0; j<sizeWorld;j++){
//                 std::copy_n(out+mu*nc+displs[j]+i*recvcounts[j]/mu,recvcounts[j]/mu,out+i*nc+displs[j]/mu);
//             }
//         }
//     }
// }

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::local_to_global(const T *const in, T *const out, const int &mu) const {
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

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::mvprod_local(const T *const in, T *const out, T *const work, const int &mu) const {
    double time = MPI_Wtime();

    this->local_to_global(in, work, mu);
    this->mymvprod_local(work, out, mu);

    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::mvprod_global(const T *const in, T *const out, const int &mu) const {
    double time = MPI_Wtime();

    if (mu == 1) {
        std::vector<T> out_perm(local_size);
        std::vector<T> buffer(std::max(nc, nr));

        // Permutation
        cluster_tree_s->global_to_cluster(in, buffer.data());

        //
        mymvprod_local(buffer.data(), out_perm.data(), 1);

        // Allgather
        std::vector<int> recvcounts(sizeWorld);
        std::vector<int> displs(sizeWorld);

        displs[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
            if (i > 0)
                displs[i] = displs[i - 1] + recvcounts[i - 1];
        }

        MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);

        // Permutation
        cluster_tree_t->cluster_to_global(buffer.data(), out);

    } else {

        std::vector<T> in_perm(std::max(nr, nc) * mu * 2);
        std::vector<T> out_perm(local_size * mu);
        std::vector<T> buffer(nc);

        for (int i = 0; i < mu; i++) {
            // Permutation
            cluster_tree_s->global_to_cluster(in + i * nc, buffer.data());

            // Transpose
            for (int j = 0; j < nc; j++) {
                in_perm[i + j * mu] = buffer[j];
            }
        }

        if (symmetry == 'H') {
            conj_if_complex(in_perm.data(), nc * mu);
        }

        mymvprod_local(in_perm.data(), in_perm.data() + nc * mu, mu);

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
            for (int j = 0; j < sizeWorld; j++) {
                std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, in_perm.data() + i * nr + displs[j] / mu);
            }

            // Permutation
            cluster_tree_t->cluster_to_global(in_perm.data() + i * nr, out + i * nr);
        }
    }
    // Timing
    infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
    infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const {
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
            const LowRankMatrix<T, ClusterImpl> &M = *(MyFarFieldMats[b]);
            int offset_i                           = M.get_offset_i();
            int offset_j                           = M.get_offset_j();
            int size_j                             = M.nb_cols();

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
                const LowRankMatrix<T, ClusterImpl> &M = *(MyDiagFarFieldMats[b]);
                int offset_i                           = M.get_offset_j();
                int offset_j                           = M.get_offset_i();
                int size_j                             = M.nb_rows();

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

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
template <typename U>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::source_to_cluster_permutation(const U *const in, U *const out) const {
    cluster_tree_s->global_to_cluster(in, out);
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
template <typename U>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::cluster_to_target_permutation(const U *const in, U *const out) const {
    cluster_tree_t->cluster_to_global(in, out);
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
std::vector<T> HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::operator*(const std::vector<T> &x) const {
    assert(x.size() == nc);
    std::vector<T> result(nr, 0);
    mvprod_global(x.data(), result.data(), 1);
    return result;
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
double HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::compression() const {

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
        if (MyNearFieldMats[j]->get_ir() == MyNearFieldMats[j]->get_ic() && this->get_symmetry_type() != 'N') {
            mycomp += nr_b * nc_b / (2 * size);
        } else {
            mycomp += nr_b * nc_b / size;
        }
    }

    double comp = 0;
    MPI_Allreduce(&mycomp, &comp, 1, MPI_DOUBLE, MPI_SUM, comm);

    return 1 - comp;
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::print_infos() const {
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

    if (rankWorld == 0) {
        for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
            std::cout << it->first << "\t" << it->second << std::endl;
        }
        std::cout << std::endl;
    }
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::save_infos(const std::string &outputname, std::ios_base::openmode mode, const std::string &sep) const {
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

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::save_plot(const std::string &outputname) const {

    std::ofstream outputfile((outputname + "_" + NbrToStr(rankWorld) + ".csv").c_str());

    if (outputfile) {
        outputfile << nr << "," << nc << std::endl;
        for (typename std::vector<std::unique_ptr<SubMatrix<T>>>::const_iterator it = MyNearFieldMats.begin(); it != MyNearFieldMats.end(); ++it) {
            outputfile << (*it)->get_offset_i() << "," << (*it)->get_ir().size() << "," << (*it)->get_offset_j() << "," << (*it)->get_ic().size() << "," << -1 << std::endl;
        }
        for (typename std::vector<std::unique_ptr<LowRankMatrix<T, ClusterImpl>>>::const_iterator it = MyFarFieldMats.begin(); it != MyFarFieldMats.end(); ++it) {
            outputfile << (*it)->get_offset_i() << "," << (*it)->get_ir().size() << "," << (*it)->get_offset_j() << "," << (*it)->get_ic().size() << "," << (*it)->rank_of() << std::endl;
        }
        outputfile.close();
    } else {
        std::cout << "Unable to create " << outputname << std::endl;
    }
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
double Frobenius_absolute_error(const HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition> &B, const IMatrix<T> &A) {
    double myerr = 0;
    for (int j = 0; j < B.MyFarFieldMats.size(); j++) {
        double test = Frobenius_absolute_error(*(B.MyFarFieldMats[j]), A);
        myerr += std::pow(test, 2);
    }

    double err = 0;
    MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

    return std::sqrt(err);
}
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
Matrix<T> HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::to_dense() const {
    Matrix<T> Dense(nr, nc);
    // Internal dense blocks
    for (int l = 0; l < MyNearFieldMats.size(); l++) {
        const SubMatrix<T> &submat = *(MyNearFieldMats[l]);
        int local_nr               = submat.nb_rows();
        int local_nc               = submat.nb_cols();
        int offset_i               = submat.get_offset_i();
        int offset_j               = submat.get_offset_j();
        for (int k = 0; k < local_nc; k++) {
            std::copy_n(&(submat(0, k)), local_nr, Dense.data() + offset_i + (offset_j + k) * nr);
        }
    }

    // Internal compressed block
    Matrix<T> FarFielBlock(local_size, local_size);
    for (int l = 0; l < MyFarFieldMats.size(); l++) {
        const LowRankMatrix<T, ClusterImpl> &lmat = *(MyFarFieldMats[l]);
        int local_nr                              = lmat.nb_rows();
        int local_nc                              = lmat.nb_cols();
        int offset_i                              = lmat.get_offset_i();
        int offset_j                              = lmat.get_offset_j();
        FarFielBlock.resize(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int k = 0; k < local_nc; k++) {
            std::copy_n(&(FarFielBlock(0, k)), local_nr, Dense.data() + offset_i + (offset_j + k) * nr);
        }
    }
    return Dense;
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
Matrix<T> HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::to_local_dense() const {
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
        const LowRankMatrix<T, ClusterImpl> &lmat = *(MyFarFieldMats[l]);
        int local_nr                              = lmat.nb_rows();
        int local_nc                              = lmat.nb_cols();
        int offset_i                              = lmat.get_offset_i();
        int offset_j                              = lmat.get_offset_j();
        FarFielBlock.resize(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int k = 0; k < local_nc; k++) {
            std::copy_n(&(FarFielBlock(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
        }
    }
    return Dense;
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
Matrix<T> HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::to_dense_perm() const {
    Matrix<T> Dense(nr, nc);
    // Internal dense blocks
    for (int l = 0; l < MyNearFieldMats.size(); l++) {
        const SubMatrix<T> &submat = *(MyNearFieldMats[l]);
        int local_nr               = submat.nb_rows();
        int local_nc               = submat.nb_cols();
        int offset_i               = submat.get_offset_i();
        int offset_j               = submat.get_offset_j();
        for (int k = 0; k < local_nc; k++)
            for (int j = 0; j < local_nr; j++)
                Dense(get_permt(j + offset_i), get_perms(k + offset_j)) = submat(j, k);
    }

    // Internal compressed block
    Matrix<T> FarFielBlock(local_size, local_size);
    for (int l = 0; l < MyFarFieldMats.size(); l++) {
        const LowRankMatrix<T, ClusterImpl> &lmat = *(MyFarFieldMats[l]);
        int local_nr                              = lmat.nb_rows();
        int local_nc                              = lmat.nb_cols();
        int offset_i                              = lmat.get_offset_i();
        int offset_j                              = lmat.get_offset_j();
        FarFielBlock.resize(local_nr, local_nc);
        lmat.get_whole_matrix(&(FarFielBlock(0, 0)));
        for (int k = 0; k < local_nc; k++)
            for (int j = 0; j < local_nr; j++)
                Dense(get_permt(j + offset_i), get_perms(k + offset_j)) = FarFielBlock(j, k);
    }
    return Dense;
}

template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void HMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::apply_dirichlet(const std::vector<int> &boundary) {
    // Renum
    std::vector<int> boundary_renum(boundary.size());
    cluster_tree_t->global_to_cluster(boundary.data(), boundary_renum.data());

    //
    for (int j = 0; j < MyStrictlyDiagNearFieldMats.size(); j++) {
        SubMatrix<T> &submat = *(MyStrictlyDiagNearFieldMats[j]);
        int local_nr         = submat.nb_rows();
        int local_nc         = submat.nb_cols();
        int offset_i         = submat.get_offset_i();
        int offset_j         = submat.get_offset_j();
        for (int i = offset_i; i < offset_i + std::min(local_nr, local_nc); i++) {
            if (boundary_renum[i])
                submat(i - offset_i, i - offset_i) = 1e30;
        }
    }
}

} // namespace htool
#endif
