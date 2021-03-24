#ifndef HTOOL_MULTIHMATRIX_HPP
#define HTOOL_MULTIHMATRIX_HPP

#include "../blocks/blocks.hpp"
#include "../clustering/cluster.hpp"
#include "../multilrmat/multilrmat.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "hmatrix.hpp"
#include "matrix.hpp"
#include "multimatrix.hpp"
#include <cassert>
#include <fstream>
#include <map>
#include <memory>
#include <mpi.h>

namespace htool {

// Friend functions --- forward declaration
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class HMatrix;

template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class MultiHMatrix;

template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
double Frobenius_absolute_error(const MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition> &B, const MultiIMatrix<T> &A, int l);

// Class
template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
class MultiHMatrix {

  private:
    // Data members
    int nr;
    int nc;
    int space_dim;
    int reqrank;
    int local_size;
    int local_offset;
    int nb_hmatrix;

    const MPI_Comm comm;
    int rankWorld, sizeWorld;

    // Parameters
    int ndofperelt;
    double epsilon;
    double eta;
    int minclustersize;
    int maxblocksize;
    int minsourcedepth;
    int mintargetdepth;

    std::vector<HMatrix<T, bareLowRankMatrix, ClusterImpl, AdmissibleCondition>> HMatrices;
    std::vector<Block<ClusterImpl, AdmissibleCondition> *> MyBlocks;

    std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_t;
    std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_s;

    std::unique_ptr<Block<ClusterImpl, AdmissibleCondition>> BlockTree;

  public:
    // Constructor
    MultiHMatrix(int space_dim0, double epsilon0 = 1e-6, double eta0 = 10, const int &reqrank0 = -1, const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(0), nc(0), space_dim(space_dim0), reqrank(reqrank0), nb_hmatrix(0), comm(comm0), ndofperelt(1), epsilon(epsilon0), eta(eta0), minclustersize(10), maxblocksize(1e6), minsourcedepth(0), mintargetdepth(0), cluster_tree_t(nullptr), cluster_tree_s(nullptr){};

    // Constructor with precomputed clusters
    MultiHMatrix(const std::shared_ptr<Cluster<ClusterImpl>> &cluster_tree_t0, const std::shared_ptr<Cluster<ClusterImpl>> &cluster_tree_s0, double epsilon0 = 1e-6, double eta0 = 10, const int &reqrank0 = -1, const MPI_Comm comm0 = MPI_COMM_WORLD) : nr(), nc(), space_dim(cluster_tree_t0->get_space_dim()), reqrank(reqrank0), nb_hmatrix(), comm(comm0), ndofperelt(1), epsilon(epsilon0), eta(eta0), minclustersize(10), maxblocksize(1e6), minsourcedepth(0), mintargetdepth(0), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0){}; // To be used with two different clusters

    // Build
    void build(MultiIMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt, const double *const xs, const double *const rs, const int *const tabs, const double *const gs);

    void build(MultiIMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<double> &rt, const std::vector<int> &tabt, const std::vector<double> &gt, const std::vector<R3> &xs, const std::vector<double> &rs, const std::vector<int> &tabs, const std::vector<double> &gs) {
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

    // // Symmetry build
    // void build_sym(MultiIMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt);

    // Build auto
    void build_auto(MultiIMatrix<T> &mat, const double *const xt, const double *const xs);

    void build_auto(MultiIMatrix<T> &mat, const std::vector<R3> &xt, const std::vector<R3> &xs) {
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
    // // Symmetry auto build
    // void build_auto_sym(IMatrix<T> &mat, const double *const xt);

    // Internal methods
    void ComputeBlocks(MultiIMatrix<T> &mat, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs);
    void AddNearFieldMat(MultiIMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, std::vector<std::unique_ptr<SubMatrix<T>>> &);
    bool AddFarFieldMat(MultiIMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<bareLowRankMatrix<T, ClusterImpl>>> &, const int &reqrank = -1);

    HMatrix<T, bareLowRankMatrix, ClusterImpl, AdmissibleCondition> &operator[](int j) { return HMatrices[j]; };
    const HMatrix<T, bareLowRankMatrix, ClusterImpl, AdmissibleCondition> &operator[](int j) const { return HMatrices[j]; };

    friend double Frobenius_absolute_error<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition>(const MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition> &B, const MultiIMatrix<T> &A, int l);

    // Getters
    int nb_rows() const { return nr; }
    int nb_cols() const { return nc; }
    const MPI_Comm &get_comm() const { return comm; }
    int get_nlrmat(int i) const {
        int res = HMatrices[i].MyFarFieldMats.size();
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm);
        return res;
    }
    int get_ndmat(int i) const {
        int res = HMatrices[i].MyNearFieldMats.size();
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm);
        return res;
    }
    int nb_hmats() const { return nb_hmatrix; }

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
    void set_minclustersize(unsigned int minclustersize0) { this->minclustersize = minclustersize0; };
    void set_minsourcedepth(unsigned int minsourcedepth0) { this->minsourcedepth = minsourcedepth0; };
    void set_mintargetdepth(unsigned int mintargetdepth0) { this->mintargetdepth = mintargetdepth0; };
    void set_maxblocksize(unsigned int maxblocksize0) { this->maxblocksize = maxblocksize0; };

    // Mat vec prod
    void mvprod_global(int i, const T *const in, T *const out, const int &mu = 1) const {
        HMatrices[i].mvprod_global(in, out, mu);
    }
};

// build
template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition>::build(MultiIMatrix<T> &mat, const double *const xt, const double *const rt, const int *const tabt, const double *const gt, const double *const xs, const double *const rs, const int *const tabs, const double *const gs) {

    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);

    std::vector<double> mytimes(3), maxtime(3), meantime(3);

    this->nc         = mat.nb_cols();
    this->nr         = mat.nb_rows();
    this->nb_hmatrix = mat.nb_matrix();

    // Construction arbre des paquets
    double time = MPI_Wtime();
    if (cluster_tree_t == nullptr) {
        cluster_tree_t = std::make_shared<ClusterImpl>(this->space_dim); // target
        cluster_tree_t->set_ndofperelt(ndofperelt);
        cluster_tree_t->set_minclustersize(minclustersize);
        cluster_tree_t->build_global(this->nr, xt, rt, tabt, gt, -1, comm);
    }
    if (cluster_tree_s == nullptr) {
        cluster_tree_s = std::make_shared<ClusterImpl>(this->space_dim); // source
        cluster_tree_s->set_minclustersize(minclustersize);
        cluster_tree_s->set_ndofperelt(ndofperelt);
        cluster_tree_s->build_global(this->nc, xs, rs, tabs, gs, -1, comm);
    }

    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();

    mytimes[0] = MPI_Wtime() - time;

    // Hmatrices
    for (int l = 0; l < nb_hmatrix; l++) {
        HMatrices.emplace_back(space_dim, nr, nc, cluster_tree_t, cluster_tree_s, 'N', 'N', comm);
    }

    // Construction arbre des blocs
    time = MPI_Wtime();
    this->BlockTree.reset(new Block<ClusterImpl, AdmissibleCondition>(*cluster_tree_t, *cluster_tree_s));
    this->BlockTree->set_mintargetdepth(this->mintargetdepth);
    this->BlockTree->set_minsourcedepth(this->minsourcedepth);
    this->BlockTree->set_maxblocksize(this->maxblocksize);
    this->BlockTree->set_eta(this->eta);
    bool force_sym = false;
    this->BlockTree->build('N', force_sym, comm);
    mytimes[1] = MPI_Wtime() - time;

    // Assemblage des sous-matrices
    time = MPI_Wtime();
    ComputeBlocks(mat, xt, tabt, xs, tabs);
    mytimes[2] = MPI_Wtime() - time;

    // Distribute necessary data
    for (int l = 0; l < nb_hmatrix; l++) {
        HMatrices[l].sizeWorld      = sizeWorld;
        HMatrices[l].rankWorld      = rankWorld;
        HMatrices[l].reqrank        = this->reqrank;
        HMatrices[l].local_size     = local_size;
        HMatrices[l].local_offset   = local_offset;
        HMatrices[l].ndofperelt     = ndofperelt;
        HMatrices[l].epsilon        = epsilon;
        HMatrices[l].eta            = eta;
        HMatrices[l].minclustersize = minclustersize;
        HMatrices[l].maxblocksize   = maxblocksize;
        HMatrices[l].minsourcedepth = minsourcedepth;
        HMatrices[l].mintargetdepth = mintargetdepth;
    }

    // Infos
    for (int l = 0; l < nb_hmatrix; l++) {
        HMatrices[l].ComputeInfos(mytimes);
    }
}

// Build auto
template <typename T, template <typename, typename> class LowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void MultiHMatrix<T, LowRankMatrix, ClusterImpl, AdmissibleCondition>::build_auto(MultiIMatrix<T> &mat, const double *const xt, const double *const xs) {
    std::vector<int> tabt(this->ndofperelt * mat.nb_rows()), tabs(this->ndofperelt * mat.nb_cols());
    std::iota(tabt.begin(), tabt.end(), int(0));
    std::iota(tabs.begin(), tabs.end(), int(0));
    this->build(mat, xt, std::vector<double>(mat.nb_rows(), 0).data(), tabt.data(), std::vector<double>(mat.nb_rows(), 1).data(), xs, std::vector<double>(mat.nb_cols(), 0).data(), tabs.data(), std::vector<double>(mat.nb_cols(), 1).data());
}

template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition>::ComputeBlocks(MultiIMatrix<T> &mat, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs) {
#if _OPENMP
#    pragma omp parallel
#endif
    {
        std::vector<std::unique_ptr<SubMatrix<T>>> MyNearFieldMats_local;
        std::vector<std::unique_ptr<bareLowRankMatrix<T, ClusterImpl>>> MyFarFieldMats_local;
        std::vector<Block<ClusterImpl, AdmissibleCondition> *> local_tasks = BlockTree->get_local_tasks();
#if _OPENMP
#    pragma omp for schedule(guided)
#endif
        for (int p = 0; p < local_tasks.size(); p++) {

            if (local_tasks[p]->IsAdmissible()) {
                bool test = AddFarFieldMat(mat, *(local_tasks[p]), xt, tabt, xs, tabs, MyFarFieldMats_local, reqrank);

                if (test) {
                    AddNearFieldMat(mat, *(local_tasks[p]), MyNearFieldMats_local);
                }
            } else {
                AddNearFieldMat(mat, *(local_tasks[p]), MyNearFieldMats_local);
            }
        }

#if _OPENMP
#    pragma omp critical
#endif
        {
            for (int l = 0; l < nb_hmatrix; l++) {
                int count = l;
                while (count < MyFarFieldMats_local.size()) {
                    HMatrices[l].MyFarFieldMats.push_back(std::move(MyFarFieldMats_local[count]));
                    count += nb_hmatrix;
                }
                count = l;
                while (count < MyNearFieldMats_local.size()) {
                    HMatrices[l].MyNearFieldMats.push_back(std::move(MyNearFieldMats_local[count]));
                    count += nb_hmatrix;
                }
            }
        }
    }
}

// Build a dense block
template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
void MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition>::AddNearFieldMat(MultiIMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, std::vector<std::unique_ptr<SubMatrix<T>>> &MyNearFieldMats_local) {

    const Cluster<ClusterImpl> &t = task.get_target_cluster();
    const Cluster<ClusterImpl> &s = task.get_source_cluster();

    MultiSubMatrix<T> Local_MultiSubMatrix(mat, std::vector<int>(cluster_tree_t->get_perm_start() + t.get_offset(), cluster_tree_t->get_perm_start() + t.get_offset() + t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start() + s.get_offset(), cluster_tree_s->get_perm_start() + s.get_offset() + s.get_size()), t.get_offset(), s.get_offset());

    for (int l = 0; l < nb_hmatrix; l++) {
        SubMatrix<T> *submat = new SubMatrix<T>(Local_MultiSubMatrix[l]);
        MyNearFieldMats_local.emplace_back(submat);
    }
}

// Build a low rank block
template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
bool MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition>::AddFarFieldMat(MultiIMatrix<T> &mat, Block<ClusterImpl, AdmissibleCondition> &task, const double *const xt, const int *const tabt, const double *const xs, const int *const tabs, std::vector<std::unique_ptr<bareLowRankMatrix<T, ClusterImpl>>> &MyFarFieldMats_local, const int &reqrank) {

    const Cluster<ClusterImpl> &t = task.get_target_cluster();
    const Cluster<ClusterImpl> &s = task.get_source_cluster();

    MultiLowRankMatrix<T, ClusterImpl> Local_MultiLowRankMatrix(std::vector<int>(cluster_tree_t->get_perm_start() + t.get_offset(), cluster_tree_t->get_perm_start() + t.get_offset() + t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start() + s.get_offset(), cluster_tree_s->get_perm_start() + s.get_offset() + s.get_size()), mat.nb_matrix(), t.get_offset(), s.get_offset(), reqrank, epsilon);
    Local_MultiLowRankMatrix.build(mat, t, xt, tabt, s, xs, tabs);

    if (Local_MultiLowRankMatrix.rank_of() != -1) {
        for (int l = 0; l < nb_hmatrix; l++) {
            bareLowRankMatrix<T, ClusterImpl> *lrmat = new bareLowRankMatrix<T, ClusterImpl>(Local_MultiLowRankMatrix[l]);
            MyFarFieldMats_local.emplace_back(lrmat);
        }
        return 0;
    } else {
        return 1;
    }
}

template <typename T, template <typename, typename> class MultiLowRankMatrix, class ClusterImpl, template <typename> class AdmissibleCondition>
double Frobenius_absolute_error(const MultiHMatrix<T, MultiLowRankMatrix, ClusterImpl, AdmissibleCondition> &B, const MultiIMatrix<T> &A, int l) {
    double myerr                                                                          = 0;
    const std::vector<std::unique_ptr<bareLowRankMatrix<T, ClusterImpl>>> &MyFarFieldMats = B[l].get_MyFarFieldMats();
    for (int j = 0; j < MyFarFieldMats.size(); j++) {
        double test = Frobenius_absolute_error(*(MyFarFieldMats[j]), A, l);
        myerr += std::pow(test, 2);
    }

    double err = 0;
    MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

    return std::sqrt(err);
}

} // namespace htool
#endif