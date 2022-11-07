#ifndef HTOOL_LOCAL_HMATRIX_HPP
#define HTOOL_LOCAL_HMATRIX_HPP

#include "../interfaces/clustering/virtual_cluster.hpp"
#include "../interfaces/hmatrix/virtual_dense_blocks_generator.hpp"
#include "../interfaces/hmatrix/virtual_generator.hpp"
#include "../interfaces/virtual_local_operator.hpp"
#include "zero_generator.hpp"

namespace htool {
template <typename T>
class LocalHMatrix : public VirtualLocalOperator<T> {
  private:
    std::shared_ptr<VirtualCluster> cluster_tree_target_;
    std::shared_ptr<VirtualCluster> cluster_tree_source_;

    std::vector<Block<T> *> MyComputedBlocks;
    std::vector<Block<T> *> MyFarFieldMats;
    std::vector<Block<T> *> MyNearFieldMats;
    std::vector<Block<T> *> MyDiagFarFieldMats;
    std::vector<Block<T> *> MyDiagNearFieldMats;
    std::vector<Block<T> *> MyDiagComputedBlocks;
    std::vector<Block<T> *> MyStrictlyDiagFarFieldMats;
    std::vector<Block<T> *> MyStrictlyDiagNearFieldMats;

    // Parameters
    double epsilon_     = 1e-6;
    double eta_         = 10;
    int maxblocksize_   = 1e6;
    int minsourcedepth_ = 0;
    int mintargetdepth_ = 0;

    // Strategies
    std::shared_ptr<VirtualLowRankGenerator<T>> LowRankGenerator;
    std::shared_ptr<VirtualAdmissibilityCondition> AdmissibilityCondition;

    // Properties
    bool delay_dense_computation_ = false;
    bool use_permutation_         = true;
    char symmetry_                = 'N';
    char UPLO_                    = 'N';
    int reqrank_                  = -1;

  public:
    HMatrix(std::shared_ptr<VirtualCluster> cluster_tree_target, const std::shared_ptr<VirtualCluster> cluster_tree_source, double epsilon = 1e-6, double eta = 10, char Symmetry = 'N', char UPLO = 'N', int reqrank = -1) : cluster_tree_target_(cluster_tree_target), cluster_tree_source_(cluster_tree_source), epsilon_(epsilon0), eta_(eta0), symmetry_(Symmetry), UPLO_(UPLO), reqrank_(reqrank){};

    //
    void build(VirtualGenerator<T> &mat, const double *const xt, const double *const xs);
    void build(VirtualGenerator<T> &mat, const double *const xt);

    // Operations
    void mvprod(const T *const in, T *const out, int mu = 1) const override;
    void mvprod_transp(const T *const in, T *const out, int mu = 1) const override;

    // Getters for relative position to global
    const VirtualCluster *cluster_tree_target() const { return cluster_tree_target_.get(); }
    const VirtualCluster *cluster_tree_source() const { return cluster_tree_source_.get(); }

    // Getters/setters for parameters
    double &
    epsilon() { return epsilon_; }
    const double &epsilon() const { return epsilon_; }
    double &eta() { return eta_; }
    const double &eta() const { return eta_; }
    double &maxblocksize() { return maxblocksize_; }
    const double &maxblocksize() const { return maxblocksize_; }
    double &minsourcedepth() { return minsourcedepth_; }
    const double &minsourcedepth() const { return minsourcedepth_; }
    double &mintargetdepth() { return mintargetdepth_; }
    const double &mintargetdepth() const { return mintargetdepth_; }

    // Getters/setters for properties
    double &delay_dense_computation() { return delay_dense_computation_; }
    const double &delay_dense_computation() const { return delay_dense_computation_; }
    double &use_permutation() { return use_permutation_; }
    const double &use_permutation() const { return use_permutation_; }
    double &symmetry() { return symmetry_; }
    const double &symmetry() const { return symmetry_; }
    double &UPLO() { return UPLO_; }
    const double &UPLO() const { return UPLO_; }
    double &reqrank() { return reqrank_; }
    const double &reqrank() const { return reqrank_; }

    // Setters for strategies
    void set_compression(std::shared_ptr<VirtualLowRankGenerator<T>> ptr) { LowRankGenerator = ptr; };
    void set_admissibility_condition(std::shared_ptr<VirtualAdmissibilityCondition<T>> ptr) { AdmissibilityCondition = ptr; };
};

template <typename T>
void LocalHMatrix<T>::build(VirtualGenerator<T> &mat, const double *const xt, const double *const xs) {
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

    // Zero generator when we delay the dense computation
    if (delay_dense_computation) {
        zerogenerator = std::unique_ptr<ZeroGenerator<T>>(new ZeroGenerator<T>(mat.nb_rows(), mat.nb_cols(), mat.get_dimension()));
    }

    // Construction arbre des paquets
    local_size   = cluster_tree_t->get_local_size();
    local_offset = cluster_tree_t->get_local_offset();
}

template <typename T>
void LocalHMatrix<T>::mvprod(const T *const in, T *const out, int mu) const {
    int local_size   = cluster_tree_target_->get_size();
    int local_offset = cluster_tree_target_->get_offset();
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
        if (symmetry_ == 'H') {
            transb = 'C';
        }

        // Contribution champ lointain
#if _OPENMP
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < MyComputedBlocks.size(); b++) {
            int offset_i = MyComputedBlocks[b]->get_target_cluster().get_offset();
            int offset_j = MyComputedBlocks[b]->get_source_cluster().get_offset();
            if (!(symmetry_ != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
                MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
            }
        }

        // Symmetry part of the diagonal part
        if (symmetry_ != 'N') {
            transb      = 'N';
            char op_sym = 'T';
            if (symmetry_ == 'H') {
                op_sym = 'C';
            }

#if _OPENMP
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < MyDiagComputedBlocks.size(); b++) {
                int offset_i = MyDiagComputedBlocks[b]->get_source_cluster().get_offset();
                int offset_j = MyDiagComputedBlocks[b]->get_target_cluster().get_offset();

                if (offset_i != offset_j) { // remove strictly diagonal blocks
                    MyDiagComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
                }
            }

#if _OPENMP
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < MyStrictlyDiagNearFieldMats.size(); b++) {
                const Block<T> *M = MyStrictlyDiagNearFieldMats[b];
                int offset_i      = M->get_target_cluster().get_offset();
                int offset_j      = M->get_source_cluster().get_offset();
                ;
                M->get_dense_block_data()->add_mvprod_row_major_sym(in + offset_j * mu, temp.data() + (offset_i - local_offset) * mu, mu, this->UPLO, this->symmetry);
            }
        }

#if _OPENMP
#    pragma omp critical
#endif
        Blas<T>::axpy(&local_size_rhs, &da, temp.data(), &incx, out, &incy);
    }
}

} // namespace htool
#endif
