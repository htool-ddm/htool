#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if defined(_OPENMP)
#    include <omp.h>
#endif
#include "../basic_types/tree.hpp"
#include "../clustering/cluster_node.hpp"
#include "../misc/logger.hpp"
#include "hmatrix_tree_data.hpp"
#include "interfaces/virtual_admissibility_condition.hpp"
#include "interfaces/virtual_dense_blocks_generator.hpp"
#include "interfaces/virtual_generator.hpp"
#include "lrmat/lrmat.hpp"
#include <queue>

#include <mpi.h>
namespace htool {

// Class
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class HMatrix : public TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>> {
  public:
    enum class StorageType {
        Dense,
        LowRank,
        Hierarchical
    };

  private:
    // Data members
    const Cluster<CoordinatePrecision> *m_target_cluster, *m_source_cluster; // child's clusters are non owning
    char m_symmetry{'N'};
    char m_UPLO{'N'};

    std::unique_ptr<Matrix<CoefficientPrecision>> m_dense_data{nullptr};
    std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> m_low_rank_data{nullptr};

    // Cached leaves
    // std::vector<HMatrix *> m_dense_leaves{};
    // std::vector<HMatrix *> m_dense_leaves_in_diagonal_block{};
    // std::vector<HMatrix *> m_diagonal_dense_leaves{};
    // std::vector<HMatrix *> m_low_rank_leaves{};
    // std::vector<HMatrix *> m_low_rank_leaves_in_diagonal_block{};
    // std::vector<HMatrix *> m_diagonal_low_rank_leaves{};
    mutable std::vector<const HMatrix *> m_leaves{};
    mutable std::vector<const HMatrix *> m_leaves_for_symmetry{};
    mutable char m_symmetry_type_for_leaves{'N'};
    // std::vector<HMatrix *> m_leaves_in_diagonal_block{};

    StorageType m_storage_type{StorageType::Hierarchical};

    void set_leaves_in_cache() const {
        if (m_leaves.empty()) {
            std::stack<std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>> hmatrix_stack;
            hmatrix_stack.push(std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>(this, m_symmetry != 'N'));

            while (!hmatrix_stack.empty()) {
                auto &current_element = hmatrix_stack.top();
                hmatrix_stack.pop();
                const HMatrix<CoefficientPrecision, CoordinatePrecision> *current_hmatrix = current_element.first;
                bool has_symmetric_ancestor                                               = current_element.second;

                if (current_hmatrix->is_leaf()) {
                    m_leaves.push_back(current_hmatrix);

                    if (has_symmetric_ancestor && current_hmatrix->get_target_cluster().get_offset() != current_hmatrix->get_source_cluster().get_offset()) {
                        m_leaves_for_symmetry.push_back(current_hmatrix);
                    }
                }

                if (m_symmetry_type_for_leaves == 'N' && current_hmatrix->get_symmetry() != 'N') {
                    m_symmetry_type_for_leaves = current_hmatrix->get_symmetry();
                }

                for (const auto &child : current_hmatrix->get_children()) {
                    hmatrix_stack.push(std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>(child.get(), current_hmatrix->get_symmetry() != 'N' || has_symmetric_ancestor));
                }
            }
        }
    }

    void
    threaded_hierarchical_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    void threaded_hierarchical_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;

  public:
    // Root constructor
    HMatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(), m_target_cluster(&target_cluster), m_source_cluster(&source_cluster) {
    }

    // Child constructor
    HMatrix(const HMatrix &parent, const Cluster<CoordinatePrecision> *target_cluster, const Cluster<CoordinatePrecision> *source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(parent), m_target_cluster(target_cluster), m_source_cluster(source_cluster) {}

    HMatrix(const HMatrix &rhs) : TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(rhs), m_target_cluster(rhs.m_target_cluster), m_source_cluster(rhs.m_source_cluster), m_symmetry(rhs.m_symmetry), m_UPLO(rhs.m_UPLO), m_leaves(), m_leaves_for_symmetry(), m_symmetry_type_for_leaves(), m_storage_type(rhs.m_storage_type) {
        Logger::get_instance().log(LogLevel::INFO, "Deep copy of HMatrix");
        this->m_depth     = rhs.m_depth;
        this->m_is_root   = rhs.m_is_root;
        this->m_tree_data = std::make_shared<HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_tree_data);
        this->m_children.clear();
        for (auto &child : rhs.m_children) {
            this->m_children.emplace_back(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(*child));
        }
        if (rhs.m_dense_data) {
            m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(*rhs.m_dense_data);
        }
        if (rhs.m_low_rank_data) {
            m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_low_rank_data);
        }
    }
    HMatrix &operator=(const HMatrix &rhs) {
        Logger::get_instance().log(LogLevel::INFO, "Deep copy of HMatrix");
        if (&rhs == this) {
            return *this;
        }
        this->m_depth     = rhs.m_depth;
        this->m_is_root   = rhs.m_is_root;
        this->m_tree_data = std::make_shared<HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_tree_data);
        this->m_children.clear();
        for (auto &child : rhs.m_children) {
            this->m_children.emplace_back(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(*child));
        }
        m_target_cluster = rhs.m_target_cluster;
        m_source_cluster = rhs.m_source_cluster;
        m_symmetry       = rhs.m_symmetry;
        m_UPLO           = rhs.m_UPLO;
        m_storage_type   = rhs.m_storage_type;

        if (rhs.m_dense_data) {
            m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(*rhs.m_dense_data);
        }
        if (rhs.m_low_rank_data) {
            m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_low_rank_data);
        }
        m_leaves.clear();
        m_leaves_for_symmetry.clear();
        return *this;
    }

    HMatrix(HMatrix &&) noexcept            = default;
    HMatrix &operator=(HMatrix &&) noexcept = default;
    virtual ~HMatrix()                      = default;

    // HMatrix getters
    const Cluster<CoordinatePrecision> &get_target_cluster() const { return *m_target_cluster; }
    const Cluster<CoordinatePrecision> &get_source_cluster() const { return *m_source_cluster; }
    int nb_cols() const { return m_source_cluster->get_size(); }
    int nb_rows() const { return m_target_cluster->get_size(); }

    HMatrix<CoefficientPrecision, CoordinatePrecision> *get_child_or_this(const Cluster<CoordinatePrecision> &required_target_cluster, const Cluster<CoordinatePrecision> &required_source_cluster) {
        if (*m_target_cluster == required_target_cluster and *m_source_cluster == required_source_cluster) {
            return this;
        }
        for (auto &child : this->m_children) {
            if (child->get_target_cluster() == required_target_cluster and child->get_source_cluster() == required_source_cluster) {
                return child.get();
            }
        }
        return nullptr;
    }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> *get_child_or_this(const Cluster<CoordinatePrecision> &required_target_cluster, const Cluster<CoordinatePrecision> &required_source_cluster) const {
        if (*m_target_cluster == required_target_cluster and *m_source_cluster == required_source_cluster) {
            return this;
        }
        for (auto &child : this->m_children) {
            if (child->get_target_cluster() == required_target_cluster and child->get_source_cluster() == required_source_cluster) {
                return child.get();
            }
        }
        return nullptr;
    }

    int get_rank() const {
        return m_storage_type == StorageType::LowRank ? m_low_rank_data->rank_of() : -1;
    }
    const std::vector<const HMatrix *> &get_leaves() const {
        set_leaves_in_cache();
        return m_leaves;
    }
    const std::vector<const HMatrix *> &get_leaves_for_symmetry() const {
        set_leaves_in_cache();
        return m_leaves_for_symmetry;
    }
    const Matrix<CoefficientPrecision> *get_dense_data() const { return m_dense_data.get(); }
    Matrix<CoefficientPrecision> *get_dense_data() { return m_dense_data.get(); }
    const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() const { return m_low_rank_data.get(); }
    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() { return m_low_rank_data.get(); }
    char get_symmetry() const { return m_symmetry; }
    char get_UPLO() const { return m_UPLO; }
    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *get_hmatrix_tree_data() const { return this->m_tree_data.get(); }
    const HMatrix<CoefficientPrecision> *get_sub_hmatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) const {
        std::queue<const HMatrix<CoefficientPrecision> *> hmatrix_queue;
        hmatrix_queue.push(this);

        while (!hmatrix_queue.empty()) {
            const HMatrix<CoefficientPrecision> *current_hmatrix = hmatrix_queue.front();
            hmatrix_queue.pop();

            if (target_cluster == current_hmatrix->get_target_cluster() && source_cluster == current_hmatrix->get_source_cluster()) {
                return current_hmatrix;
            }

            const auto &children = current_hmatrix->get_children();
            for (auto &child : children) {
                hmatrix_queue.push(child.get());
            }
        }
        return nullptr;
    }
    HMatrix<CoefficientPrecision> *get_sub_hmatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) {
        std::queue<HMatrix<CoefficientPrecision> *> hmatrix_queue;
        hmatrix_queue.push(this);

        while (!hmatrix_queue.empty()) {
            HMatrix<CoefficientPrecision> *current_hmatrix = hmatrix_queue.front();
            hmatrix_queue.pop();

            if (target_cluster == current_hmatrix->get_target_cluster() && source_cluster == current_hmatrix->get_source_cluster()) {
                return current_hmatrix;
            }

            auto &children = current_hmatrix->get_children();
            for (auto &child : children) {
                hmatrix_queue.push(child.get());
            }
        }
        return nullptr;
    }
    StorageType get_storage_type() const { return m_storage_type; }

    // HMatrix node setters
    void set_symmetry(char symmetry) { m_symmetry = symmetry; }
    void set_UPLO(char UPLO) { m_UPLO = UPLO; }
    void set_target_cluster(const Cluster<CoordinatePrecision> *new_target_cluster) { m_target_cluster = new_target_cluster; }

    // Test properties
    bool is_dense() const { return m_storage_type == StorageType::Dense; }
    bool is_low_rank() const { return m_storage_type == StorageType::LowRank; }
    bool is_hierarchical() const { return m_storage_type == StorageType::Hierarchical; }

    // HMatrix Tree setters
    void set_eta(CoordinatePrecision eta) { this->m_tree_data->m_eta = eta; }
    void set_epsilon(underlying_type<CoefficientPrecision> epsilon) { this->m_tree_data->m_epsilon = epsilon; }
    void set_low_rank_generator(std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> ptr) { this->m_tree_data->m_low_rank_generator = ptr; }
    void set_admissibility_condition(std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> ptr) { this->m_tree_data->m_admissibility_condition = ptr; }
    void set_maximal_block_size(int maxblock_size) { this->m_tree_data->m_maxblocksize = maxblock_size; }
    void set_minimal_target_depth(unsigned int minimal_target_depth) { this->m_tree_data->m_minimal_target_depth = minimal_target_depth; }
    void set_minimal_source_depth(unsigned int minimal_source_depth) { this->m_tree_data->m_minimal_source_depth = minimal_source_depth; }

    // HMatrix Tree setters
    char get_symmetry_for_leaves() const { return m_symmetry_type_for_leaves; }

    // Infos
    // const std::map<std::string, std::string> &get_infos() const { return infos; }
    // std::string get_infos(const std::string &key) const { return infos[key]; }
    // void add_info(const std::string &keyname, const std::string &value) const { infos[keyname] = value; }
    // void print_infos() const;
    // void save_infos(const std::string &outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string &sep = " = ") const;
    // underlying_type<CoefficientPrecision> compressed_size() const;
    // double compression_ratio() const;
    // double space_saving() const;
    // friend underlying_type<CoefficientPrecision> Frobenius_absolute_error<CoefficientPrecision>(const HMatrix<CoefficientPrecision,CoordinatePrecision> &B, const VirtualGenerator<CoefficientPrecision> &A);

    // // // Output structure
    // void save_plot(const std::string &outputname) const;
    // std::vector<DisplayBlock> get_output() const;

    // Data computation
    void compute_dense_data(const VirtualGenerator<CoefficientPrecision> &generator) {
        m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(m_target_cluster->get_size(), m_source_cluster->get_size());
        generator.copy_submatrix(m_target_cluster->get_size(), m_source_cluster->get_size(), m_target_cluster->get_offset(), m_source_cluster->get_offset(), m_dense_data->data());
        m_storage_type = StorageType::Dense;
    }

    void compute_low_rank_data(const VirtualGenerator<CoefficientPrecision> &generator, const VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision> &low_rank_generator, int reqrank, underlying_type<CoefficientPrecision> epsilon) {
        m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(generator, low_rank_generator, *m_target_cluster, *m_source_cluster, reqrank, epsilon);
        m_storage_type  = StorageType::LowRank;
    }
    void clear_low_rank_data() { m_low_rank_data.reset(); }

    void set_dense_data(Matrix<CoefficientPrecision> &dense_matrix) {
        this->delete_children();
        m_leaves.clear();
        m_leaves_for_symmetry.clear();
        m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>();
        m_dense_data->assign(dense_matrix.nb_rows(), dense_matrix.nb_cols(), dense_matrix.release(), true);
        m_storage_type = StorageType::Dense;
    }

    // Linear algebra
    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;

    // void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const;
    // void add_matrix_product(CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const;

    // void mvprod_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    // void mvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    // void mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu = 1) const;
    // void mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;

    // void mymvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    // void mymvprod_global_to_local(const T *const in, T *const out, const int &mu = 1) const;
    // void mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const;
    // void mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu = 1) const;

    // void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const;
    // std::vector<CoefficientPrecision> operator*(const std::vector<CoefficientPrecision> &x) const;
    // Matrix<CoefficientPrecision> operator*(const Matrix<CoefficientPrecision> &x) const;

    // // Permutations
    // void source_to_cluster_permutation(const T *const in, T *const out) const;
    // void local_target_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    // void local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    // void local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;
    // void local_cluster_to_local_source(const T *const in, T *const out, MPI_Comm comm = MPI_COMM_WORLD) const;

    // // local to global
    // void local_to_global_source(const T *const in, T *const out, const int &mu) const;
    // void local_to_global_target(const T *const in, T *const out, const int &mu) const;

    // // Convert
    // Matrix<CoefficientPrecision> get_local_dense() const;
    // Matrix<CoefficientPrecision> get_local_dense_perm() const;
    // void copy_to_dense(CoefficientPrecision *) const;

    // // Apply Dirichlet condition
    // void apply_dirichlet(const std::vector<int> &boundary);
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {
    switch (m_storage_type) {
    case StorageType::Dense:
        if (m_symmetry == 'N') {
            m_dense_data->add_vector_product(trans, alpha, in, beta, out);
        } else {
            m_dense_data->add_vector_product_symmetric(trans, alpha, in, beta, out, m_UPLO, m_symmetry);
        }
        break;
    case StorageType::LowRank:
        m_low_rank_data->add_vector_product(trans, alpha, in, beta, out);
        break;
    default:
        threaded_hierarchical_add_vector_product(trans, alpha, in, beta, out);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {
    switch (m_storage_type) {
    case StorageType::Dense:
        if (m_symmetry == 'N') {
            m_dense_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu);
        } else {
            m_dense_data->add_matrix_product_symmetric_row_major(trans, alpha, in, beta, out, mu, m_UPLO, m_symmetry);
        }
        break;
    case StorageType::LowRank:
        m_low_rank_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu);
        break;
    default:
        threaded_hierarchical_add_matrix_product_row_major(trans, alpha, in, beta, out, mu);
        break;
    }
}

// // build
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::build(VirtualGenerator<CoefficientPrecision> &mat) {

//     // std::vector<double> mytimes(3), maxtime(3), meantime(3);

//     // // Default compression: sympartialACA
//     // if (m_block_tree_properties->m_low_rank_generator == nullptr) {
//     //     m_block_tree_properties->m_low_rank_generator = std::make_shared<sympartialACA<CoefficientPrecision>>();
//     // }

//     // Build block tree
//     bool not_pushed = false;
//     // if (m_blockm_block_tree_properties->m_UPLO == 'U' || m_block_tree_properties->m_UPLO == 'L') {
//     //     not_pushed = build_symmetric_block_tree();
//     // } else {
//     not_pushed = build_block_tree();
//     // }
//     // for (const std::unique_ptr<HMatrix<CoefficientPrecision,CoordinatePrecision>> &child : m_children) {
//     //     std::cout << child->get_target_cluster_tree().get_offset() << " " << child->get_target_cluster_tree().get_size() << " " << child->get_source_cluster_tree().get_offset() << " " << child->get_source_cluster_tree().get_size() << "\n";
//     // }

//     reset_root_of_block_tree();
//     if (not_pushed) {
//         m_block_tree_properties->m_tasks.push_back(this);
//     }

//     // for (const std::unique_ptr<HMatrix<CoefficientPrecision,CoordinatePrecision>> &child : m_children) {
//     //     std::cout << child->get_target_cluster_tree().get_offset() << " " << child->get_target_cluster_tree().get_size() << " " << child->get_source_cluster_tree().get_offset() << " " << child->get_source_cluster_tree().get_size() << "\n";
//     // }

//     // // Sort local tasks
//     // std::sort(m_block_tree_properties->m_tasks.begin(), m_block_tree_properties->m_tasks.end(), [](HMatrix *hmatrix_A, HMatrix *hmatrix_B) { return (hmatrix_A->m_source_cluster_tree->get_offset() == hmatrix_B->m_source_cluster_tree->get_offset()) ? (hmatrix_A->m_target_cluster_tree->get_offset() < hmatrix_B->m_target_cluster_tree->get_offset()) : hmatrix_A->m_source_cluster_tree->get_offset() < hmatrix_B->m_source_cluster_tree->get_offset(); });

//     // Compute blocks
//     compute_blocks(mat);

//     // // Infos
//     // ComputeInfos(mytimes);
// }

// // Symmetry build
// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::build(VirtualGenerator<CoefficientPrecision> &mat, const double *const xt) {

//     MPI_Comm_size(comm, &sizeWorld);
//     MPI_Comm_rank(comm, &rankWorld);
//     std::vector<double> mytimes(3), maxtime(3), meantime(3);

//     this->nc        = mat.nb_cols();
//     this->nr        = mat.nb_rows();
//     this->dimension = mam_target_cluster_tree->get_dimension();

//     // Default compression: sympartialACA
//     if (this->LowRankGenerator == nullptr) {
//         this->LowRankGenerator = std::make_shared<sympartialACA<CoefficientPrecision>>();
//     }

//     // Default admissibility condition
//     if (this->AdmissibilityCondition == nullptr) {
//         this->AdmissibilityCondition = std::make_shared<RjasanowSteinbach>();
//     }

//     // Zero generator when we delay the dense computation
//     if (delay_dense_computation) {
//         zerogenerator = std::unique_ptr<ZeroGenerator<CoefficientPrecision>>(new ZeroGenerator<CoefficientPrecision>(mat.nb_rows(), mat.nb_cols(), mam_target_cluster_tree->get_dimension()));
//     }

//     // Construction arbre des paquets
//     local_size   = cluster_tree_t->get_local_size();
//     local_offset = cluster_tree_t->get_local_offset();

//     // Construction arbre des blocs
//     double time = MPI_Wtime();

//     if (this->OffDiagonalApproximation != nullptr) {
//         this->BlockTree.reset(new Block<CoefficientPrecision>(this->AdmissibilityCondition->get(), cluster_tree_t->get_local_cluster(), cluster_tree_s->get_local_cluster()));
//     } else {
//         this->BlockTree.reset(new Block<CoefficientPrecision>(this->AdmissibilityCondition->get(), *cluster_tree_t, *cluster_tree_s));
//     }
//     this->BlockTree->set_mintargetdepth(m_mintargetdepth);
//     this->BlockTree->set_minsourcedepth(m_minsourcedepth);
//     this->BlockTree->set_maxblocksize(this->maxblocksize);
//     this->BlockTree->set_eta(this->eta);
//     bool force_sym = true;
//     this->BlockTree->build(UPLO, force_sym, comm);

//     mytimes[0] = MPI_Wtime() - time;

//     // Assemblage des sous-matrices
//     time = MPI_Wtime();
//     ComputeBlocks(mat, xt, xt);
//     mytimes[1] = MPI_Wtime() - time;

//     // Infos
//     ComputeInfos(mytimes);
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::build_dense_blocks(VirtualDenseBlocksGenerator<CoefficientPrecision> &dense_block_generator) {

//     int number_of_dense_blocks = m_block_tree_properties->m_dense_leaves.size();
//     auto &dense_blocks         = m_block_tree_properties->m_dense_leaves;

//     std::vector<int> row_sizes(number_of_dense_blocks, 0), col_sizes(number_of_dense_blocks, 0);
//     std::vector<const int *> rows(number_of_dense_blocks, nullptr), cols(number_of_dense_blocks, nullptr);
//     std::vector<T *> ptr(number_of_dense_blocks, nullptr);

//     for (int i = 0; i < number_of_dense_blocks; i++) {
//         row_sizes[i] = dense_blocks[i]->get_target_cluster()->get_size();
//         col_sizes[i] = dense_blocks[i]->get_source_cluster()->get_size();
//         rows[i]      = dense_blocks[i]->get_target_cluster()->get_perm_data();
//         cols[i]      = dense_blocks[i]->get_source_cluster()->get_perm_data();
//         ptr[i]       = dense_blocks[i]->get_dense_block_data()->data();
//     }
//     dense_block_generator.copy_dense_blocks(row_sizes, col_sizes, rows, cols, ptr);
// }

// // // Compute blocks recursively
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::compute_blocks(const VirtualGenerator<CoefficientPrecision> &mat) {
// #if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> local_dense_leaves{};
//         std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> local_low_rank_leaves{};
//         std::vector<HMatrix *> &local_tasks = m_block_tree_properties->m_tasks;

//         int false_positive_local = 0;
// #if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
// #    pragma omp for schedule(guided)
// #endif
//         for (int p = 0; p < m_block_tree_properties->m_tasks.size(); p++) {
//             if (!local_tasks[p]->is_admissible()) {
//                 local_tasks[p]->add_dense_leaf(mat, local_dense_leaves);
//             } else {
//                 // bool not_pushed;
//                 local_tasks[p]->add_low_rank_leaf(mat, local_low_rank_leaves);
//                 if (local_tasks[p]->get_low_rank_data()->rank_of() == -1) {
//                     local_tasks[p]->m_low_rank_data.reset();
//                     local_low_rank_leaves.pop_back();
//                     local_tasks[p]->add_dense_leaf(mat, local_dense_leaves);
//                 }
//                 // if (m_symmetry == 'H' || m_symmetry == 'S') {
//                 //     not_pushed = ComputeAdmissibleBlocksSym(mat, *(local_tasks[p]), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 // } else {
//                 //     not_pushed = ComputeAdmissibleBlock(mat, *(local_tasks[p]), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 // }

//                 // if (not_pushed) {
//                 //     local_tasks[p]->add_dense_leaf(mat, local_dense_leaves);
//                 // }
//             }
//         }
// #if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
// #    pragma omp critical
// #endif
//         {
//             m_block_tree_properties->m_low_rank_leaves.insert(m_block_tree_properties->m_low_rank_leaves.end(), std::make_move_iterator(local_low_rank_leaves.begin()), std::make_move_iterator(local_low_rank_leaves.end()));

//             m_block_tree_properties->m_dense_leaves.insert(m_block_tree_properties->m_dense_leaves.end(), std::make_move_iterator(local_dense_leaves.begin()), std::make_move_iterator(local_dense_leaves.end()));

//             m_block_tree_properties->m_false_positive += false_positive_local;
//         }
//     }

//     m_block_tree_properties->m_leaves.insert(m_block_tree_properties->m_leaves.end(), m_block_tree_properties->m_dense_leaves.begin(), m_block_tree_properties->m_dense_leaves.end());
//     m_block_tree_properties->m_leaves.insert(m_block_tree_properties->m_leaves.end(), m_block_tree_properties->m_low_rank_leaves.begin(), m_block_tree_properties->m_low_rank_leaves.end());

//     if (m_block_tree_properties->m_block_diagonal_hmatrix != nullptr) {
//         int local_offset_s = m_block_tree_properties->m_block_diagonal_hmatrix->get_source_cluster_tree().get_offset();
//         int local_size_s   = m_block_tree_properties->m_block_diagonal_hmatrix->get_source_cluster_tree().get_size();

//         // Build vectors of pointers for diagonal blocks
//         for (auto leaf : m_block_tree_properties->m_leaves) {
//             if (local_offset_s <= leaf->get_source_cluster_tree().get_offset() && leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
//                 m_block_tree_properties->m_leaves_in_diagonal_block.push_back(leaf);
//             }
//         }
//         for (auto low_rank_leaf : m_block_tree_properties->m_low_rank_leaves) {
//             if (local_offset_s <= low_rank_leaf->get_source_cluster_tree().get_offset() && low_rank_leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
//                 m_block_tree_properties->m_low_rank_leaves_in_diagonal_block.push_back(low_rank_leaf);
//                 if (low_rank_leaf->get_source_cluster_tree().get_offset() == low_rank_leaf->get_target_cluster_tree().get_offset()) {
//                     m_block_tree_properties->m_diagonal_low_rank_leaves.push_back(low_rank_leaf);
//                 }
//             }
//         }
//         for (auto dense_leaf : m_block_tree_properties->m_dense_leaves) {
//             if (local_offset_s <= dense_leaf->get_source_cluster_tree().get_offset() && dense_leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
//                 m_block_tree_properties->m_dense_leaves_in_diagonal_block.push_back(dense_leaf);
//                 if (dense_leaf->get_source_cluster_tree().get_offset() == dense_leaf->get_target_cluster_tree().get_offset()) {
//                     m_block_tree_properties->m_diagonal_dense_leaves.push_back(dense_leaf);
//                 }
//             }
//         }
//     }
// }

// template <typename CoefficientPrecision>
// bool HMatrix<CoefficientPrecision,CoordinatePrecision>::ComputeAdmissibleBlock(VirtualGenerator<CoefficientPrecision> &mat, Block<CoefficientPrecision> &task, const double *const xt, const double *const xs, std::vector<Block<CoefficientPrecision> *> &MyComputedBlocks_local, std::vector<Block<CoefficientPrecision> *> &MyNearFieldMats_local, std::vector<Block<CoefficientPrecision> *> &MyFarFieldMats_local, int &false_positive_local) {
//     if (task.IsAdmissible()) { // When called recursively, it may not be admissible
//         AddFarFieldMat(mat, task, xt, xs, MyComputedBlocks_local, MyFarFieldMats_local, reqrank);
//         if (MyFarFieldMats_local.back()->get_rank_of() != -1) {
//             return false;
//         } else {
//             MyComputedBlocks_local.back()->clear_data();
//             MyFarFieldMats_local.pop_back();
//             MyComputedBlocks_local.pop_back();
//             false_positive_local += 1;
//         }
//     }
//     // We could compute a dense block if its size is small enough, we focus on improving compression for now
//     // else if (task->get_size()<maxblocksize){
//     //     AddNearFieldMat(mat,task,MyComputedBlocks_local, MyNearFieldMats_local);
//     //     return false;
//     // }

//     std::size_t bsize       = task->get_size();
//     const VirtualCluster &t = m_target_cluster;
//     const VirtualCluster &s = m_source_cluster.

//     if (s.IsLeaf()) {
//         if (t.IsLeaf()) {
//             return true;
//         } else {
//             std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children());
//             for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                 task.build_son(m_target_cluster_tree->get_children()[p], s);

//                 Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         }
//     } else {
//         if (t.IsLeaf()) {
//             std::vector<bool> Blocks_not_pushed(m_source_cluster_tree->number_of_children());
//             for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                 task.build_son(t, m_source_cluster_tree->get_children()[p]);
//                 Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         } else {
//             if (m_target_cluster_tree->get_size() > m_source_cluster_tree->get_size()) {
//                 std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children());
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     task.build_son(m_target_cluster_tree->get_children()[p], s);
//                     Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 }

//                 if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                     task.clear_sons();
//                     return true;
//                 } else {
//                     for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                         if (Blocks_not_pushed[p]) {
//                             AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                         }
//                     }
//                     return false;
//                 }
//             } else {
//                 std::vector<bool> Blocks_not_pushed(m_source_cluster_tree->number_of_children());
//                 for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                     task.build_son(t, m_source_cluster_tree->get_children()[p]);
//                     Blocks_not_pushed[p] = ComputeAdmissibleBlock(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 }

//                 if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                     task.clear_sons();
//                     return true;
//                 } else {
//                     for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                         if (Blocks_not_pushed[p]) {
//                             AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                         }
//                     }
//                     return false;
//                 }
//             }
//         }
//     }
// }

// template <typename CoefficientPrecision>
// bool HMatrix<CoefficientPrecision,CoordinatePrecision>::ComputeAdmissibleBlocksSym(VirtualGenerator<CoefficientPrecision> &mat, Block<CoefficientPrecision> &task, const double *const xt, const double *const xs, std::vector<Block<CoefficientPrecision> *> &MyComputedBlocks_local, std::vector<Block<CoefficientPrecision> *> &MyNearFieldMats_local, std::vector<Block<CoefficientPrecision> *> &MyFarFieldMats_local, int &false_positive_local) {

//     if (task.IsAdmissible()) {

//         AddFarFieldMat(mat, task, xt, xs, MyComputedBlocks_local, MyFarFieldMats_local, reqrank);
//         if (MyFarFieldMats_local.back()->get_rank_of() != -1) {
//             return false;
//         } else {
//             MyComputedBlocks_local.back()->clear_data();
//             MyFarFieldMats_local.pop_back();
//             MyComputedBlocks_local.pop_back();
//             false_positive_local += 1;
//             // AddNearFieldMat(mat, task,MyComputedBlocks_local, MyNearFieldMats_local);
//             // return false;
//         }
//     }
//     // We could compute a dense block if its size is small enough, we focus on improving compression for now
//     // else if (task->get_size()<maxblocksize){
//     //     AddNearFieldMat(mat,task,MyComputedBlocks_local, MyNearFieldMats_local);
//     //     return false;
//     // }

//     std::size_t bsize       = task->get_size();
//     const VirtualCluster &t = m_target_cluster;
//     const VirtualCluster &s = m_source_cluster.

//     if (s.IsLeaf()) {
//         if (t.IsLeaf()) {
//             return true;
//         } else {
//             std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children());
//             for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                 task.build_son(m_target_cluster_tree->get_children()[p], s);
//                 Blocks_not_pushed[p] = ComputeAdmissibleBlocksSym(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         }
//     } else {
//         if (t.IsLeaf()) {
//             std::vector<bool> Blocks_not_pushed(m_source_cluster_tree->number_of_children());
//             for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                 task.build_son(t, m_source_cluster_tree->get_children()[p]);
//                 Blocks_not_pushed[p] = ComputeAdmissibleBlocksSym(mat, task->get_children()[p], xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//             }

//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < m_source_cluster_tree->number_of_children(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         } else {
//             std::vector<bool> Blocks_not_pushed(m_target_cluster_tree->number_of_children() * m_source_cluster_tree->number_of_children());
//             for (int l = 0; l < m_source_cluster_tree->number_of_children(); l++) {
//                 for (int p = 0; p < m_target_cluster_tree->number_of_children(); p++) {
//                     task.build_son(m_target_cluster_tree->get_children()[p], m_source_cluster_tree->get_son(l));
//                     Blocks_not_pushed[p + l * m_target_cluster_tree->number_of_children()] = ComputeAdmissibleBlocksSym(mat, task->get_son(p + l * m_target_cluster_tree->number_of_children()), xt, xs, MyComputedBlocks_local, MyNearFieldMats_local, MyFarFieldMats_local, false_positive_local);
//                 }
//             }
//             if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })) {
//                 task.clear_sons();
//                 return true;
//             } else {
//                 for (int p = 0; p < Blocks_not_pushed.size(); p++) {
//                     if (Blocks_not_pushed[p]) {
//                         AddNearFieldMat(mat, task->get_children()[p], MyComputedBlocks_local, MyNearFieldMats_local);
//                     }
//                 }
//                 return false;
//             }
//         }
//     }
// }

// // Build a dense block
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_dense_leaf(const VirtualGenerator<CoefficientPrecision> &mat, std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &local_dense_leaves) {

//     if (!m_block_tree_properties->m_delay_dense_computation) {
//         m_dense_data = std::unique_ptr<Matrix<CoefficientPrecision>>(new Matrix<CoefficientPrecision>(m_target_cluster_tree->get_size(), m_source_cluster_tree->get_size()));

//         mat.copy_submatrix(m_target_cluster_tree->get_size(), m_source_cluster_tree->get_size(), m_target_cluster_tree->get_offset(), m_source_cluster_tree->get_offset(), m_dense_data->data());
//     }
//     local_dense_leaves.push_back(this);
//     m_storage_type = StorageType::Dense;
// }

// // // Build a low rank block
// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_low_rank_leaf(const VirtualGenerator<CoefficientPrecision> &mat, std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &local_low_rank_leaves) {

//     m_low_rank_data = std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(new LowRankMatrix<CoefficientPrecision, CoordinatePrecision>(mat, *m_block_tree_properties->m_low_rank_generator, *m_target_cluster_tree, *m_source_cluster_tree, m_block_tree_properties->m_reqrank, m_block_tree_properties->m_epsilon));

//     local_low_rank_leaves.push_back(this);
//     m_storage_type = StorageType::LowRank;
// }

// // Compute infos
// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::ComputeInfos(const std::vector<double> &mytime) {
//     // 0 : block tree ; 1 : compute blocks ;
//     std::vector<double> maxtime(2), meantime(2);
//     // 0 : dense mat ; 1 : lr mat ; 2 : rank ; 3 : local_size
//     std::vector<std::size_t> maxinfos(4, 0), mininfos(4, std::max(nc, nr));
//     std::vector<double> meaninfos(4, 0);
//     // Infos
//     for (int i = 0; i < MyNearFieldMats.size(); i++) {
//         std::size_t size = MyNearFieldMats[i]->get_target_cluster()->get_size() * MyNearFieldMats[i]->get_source_cluster()->get_size();
//         maxinfos[0]      = std::max(maxinfos[0], size);
//         mininfos[0]      = std::min(mininfos[0], size);
//         meaninfos[0] += size;
//     }
//     for (int i = 0; i < MyFarFieldMats.size(); i++) {
//         std::size_t size = MyFarFieldMats[i]->get_target_cluster()->get_size() * MyFarFieldMats[i]->get_source_cluster()->get_size();
//         std::size_t rank = MyFarFieldMats[i]->get_rank_of();
//         maxinfos[1]      = std::max(maxinfos[1], size);
//         mininfos[1]      = std::min(mininfos[1], size);
//         meaninfos[1] += size;
//         maxinfos[2] = std::max(maxinfos[2], rank);
//         mininfos[2] = std::min(mininfos[2], rank);
//         meaninfos[2] += rank;
//     }
//     maxinfos[3]  = local_size;
//     mininfos[3]  = local_size;
//     meaninfos[3] = local_size;

//     if (rankWorld == 0) {
//         MPI_Reduce(MPI_IN_PLACE, &(maxinfos[0]), 4, my_MPI_SIZE_T, MPI_MAX, 0, comm);
//         MPI_Reduce(MPI_IN_PLACE, &(mininfos[0]), 4, my_MPI_SIZE_T, MPI_MIN, 0, comm);
//         MPI_Reduce(MPI_IN_PLACE, &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
//         MPI_Reduce(MPI_IN_PLACE, &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
//     } else {
//         MPI_Reduce(&(maxinfos[0]), &(maxinfos[0]), 4, my_MPI_SIZE_T, MPI_MAX, 0, comm);
//         MPI_Reduce(&(mininfos[0]), &(mininfos[0]), 4, my_MPI_SIZE_T, MPI_MIN, 0, comm);
//         MPI_Reduce(&(meaninfos[0]), &(meaninfos[0]), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
//         MPI_Reduce(&(false_positive), &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
//     }

//     int nlrmat   = this->get_nlrmat();
//     int ndmat    = this->get_ndmat();
//     meaninfos[0] = (ndmat == 0 ? 0 : meaninfos[0] / ndmat);
//     meaninfos[1] = (nlrmat == 0 ? 0 : meaninfos[1] / nlrmat);
//     meaninfos[2] = (nlrmat == 0 ? 0 : meaninfos[2] / nlrmat);
//     meaninfos[3] = meaninfos[3] / sizeWorld;
//     mininfos[0]  = (ndmat == 0 ? 0 : mininfos[0]);
//     mininfos[1]  = (nlrmat == 0 ? 0 : mininfos[1]);
//     mininfos[2]  = (nlrmat == 0 ? 0 : mininfos[2]);

//     // timing
//     MPI_Reduce(&(mytime[0]), &(maxtime[0]), 2, MPI_DOUBLE, MPI_MAX, 0, comm);
//     MPI_Reduce(&(mytime[0]), &(meantime[0]), 2, MPI_DOUBLE, MPI_SUM, 0, comm);

//     meantime /= sizeWorld;

//     infos["Block_tree_mean"] = NbrToStr(meantime[0]);
//     infos["Block_tree_max"]  = NbrToStr(maxtime[0]);
//     infos["Blocks_mean"]     = NbrToStr(meantime[1]);
//     infos["Blocks_max"]      = NbrToStr(maxtime[1]);

//     // Size
//     infos["Source_size"]              = NbrToStr(this->nc);
//     infos["Target_size"]              = NbrToStr(this->nr);
//     infos["Dimension"]                = NbrToStr(this->dimension);
//     infos["Dense_block_size_max"]     = NbrToStr(maxinfos[0]);
//     infos["Dense_block_size_mean"]    = NbrToStr(meaninfos[0]);
//     infos["Dense_block_size_min"]     = NbrToStr(mininfos[0]);
//     infos["Low_rank_block_size_max"]  = NbrToStr(maxinfos[1]);
//     infos["Low_rank_block_size_mean"] = NbrToStr(meaninfos[1]);
//     infos["Low_rank_block_size_min"]  = NbrToStr(mininfos[1]);

//     infos["Rank_max"]                 = NbrToStr(maxinfos[2]);
//     infos["Rank_mean"]                = NbrToStr(meaninfos[2]);
//     infos["Rank_min"]                 = NbrToStr(mininfos[2]);
//     infos["Number_of_lrmat"]          = NbrToStr(nlrmat);
//     infos["Number_of_dmat"]           = NbrToStr(ndmat);
//     infos["Number_of_false_positive"] = NbrToStr(false_positive);
//     infos["Local_compressed_size"]    = NbrToStr(this->local_compressed_size());
//     infos["Compression_ratio"]        = NbrToStr(this->compression_ratio());
//     infos["Space_saving"]             = NbrToStr(this->space_saving());
//     infos["Local_size_max"]           = NbrToStr(maxinfos[3]);
//     infos["Local_size_mean"]          = NbrToStr(meaninfos[3]);
//     infos["Local_size_min"]           = NbrToStr(mininfos[3]);

//     infos["Number_of_MPI_tasks"] = NbrToStr(sizeWorld);
// #if defined(_OPENMP)
//     infos["Number_of_threads_per_tasks"] = NbrToStr(omp_get_max_threads());
//     infos["Number_of_procs"]             = NbrToStr(sizeWorld * omp_get_max_threads());
// #else
//     infos["Number_of_procs"] = NbrToStr(sizeWorld);
// #endif

//     infos["Eta"]                   = NbrToStr(eta);
//     infos["Eps"]                   = NbrToStr(epsilon);
//     infos["MinTargetDepth"]        = NbrToStr(mintargetdepth);
//     infos["MinSourceDepth"]        = NbrToStr(minsourcedepth);
//     infos["MinClusterSizeTarget"]  = NbrToStr(cluster_tree_t->get_minclustersize());
//     infos["MinClusterSizeSource"]  = NbrToStr(cluster_tree_s->get_minclustersize());
//     infos["MinClusterDepthTarget"] = NbrToStr(cluster_tree_t->get_min_depth());
//     infos["MaxClusterDepthTarget"] = NbrToStr(cluster_tree_t->get_max_depth());
//     infos["MinClusterDepthSource"] = NbrToStr(cluster_tree_s->get_min_depth());
//     infos["MaxClusterDepthSource"] = NbrToStr(cluster_tree_s->get_max_depth());
//     infos["MaxBlockSize"]          = NbrToStr(maxblocksize);
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {

//     int target_size   = m_target_cluster_tree->get_size();
//     int source_size   = m_source_cluster_tree->get_size();
//     int source_offset = m_source_cluster_tree->get_offset();
//     int target_offset = m_target_cluster_tree->get_offset();

//     int incx(1), incy(1);
//     CoefficientPrecision da(1);
//     const auto &blocks                   = m_block_tree_properties->m_tasks;
//     const auto &blocks_in_diagonal_block = m_block_tree_properties->m_leaves_in_diagonal_block;
//     const auto &diagonal_dense_blocks    = m_block_tree_properties->m_diagonal_dense_leaves;
//     CoefficientPrecision local_beta      = 0;

//     char symmetry_trans;
//     if (m_block_tree_properties->m_symmetry == 'S') {
//         if (trans == 'N') {
//             symmetry_trans = 'T';
//         } else {
//             symmetry_trans = 'N';
//         }
//     } else if (m_block_tree_properties->m_symmetry == 'H') {
//         if (trans == 'N') {
//             symmetry_trans = 'C';
//         } else {
//             symmetry_trans = 'N';
//         }
//     }

//     std::transform(out, out + target_size, out, [beta](CoefficientPrecision a) { return a * beta; });

// // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(target_size, 0);
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//         for (int b = 0; b < blocks.size(); b++) {
//             int offset_i = blocks[b]->get_target_cluster_tree().get_offset() - target_offset;
//             int offset_j = blocks[b]->get_source_cluster_tree().get_offset() - source_offset;
//             std::cout << offset_i << " " << offset_j << "\n";
//             if (m_block_tree_properties->m_symmetry == 'N' || offset_i != offset_j) { // remove strictly diagonal blocks
//                 blocks[b]->is_dense() ? blocks[b]->get_dense_data()->add_vector_product(trans, alpha, in + offset_j, local_beta, temp.data() + offset_i) : blocks[b]->get_low_rank_data()->add_vector_product(trans, alpha, in + offset_j, local_beta, temp.data() + offset_i);
//                 ;
//             }
//         }

//         // Symmetry part of the diagonal part
//         if (m_block_tree_properties->m_symmetry != 'N') {
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//             for (int b = 0; b < blocks_in_diagonal_block.size(); b++) {
//                 int offset_i = blocks_in_diagonal_block[b]->get_source_cluster_tree().get_offset();
//                 int offset_j = blocks_in_diagonal_block[b]->get_target_cluster_tree().get_offset();

//                 if (offset_i != offset_j) { // remove strictly diagonal blocks
//                     blocks[b]->is_dense() ? blocks_in_diagonal_block[b]->get_dense_data()->add_vector_product(symmetry_trans, alpha, in + offset_j - target_offset, local_beta, temp.data() + (offset_i - source_offset)) : blocks_in_diagonal_block[b]->get_low_rank_data()->add_vector_product(symmetry_trans, alpha, in + offset_j - target_offset, local_beta, temp.data() + (offset_i - source_offset));
//                 }
//             }

// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//             for (int b = 0; b < diagonal_dense_blocks.size(); b++) {
//                 int offset_i = diagonal_dense_blocks[b]->get_target_cluster_tree().get_offset();
//                 int offset_j = diagonal_dense_blocks[b]->get_source_cluster_tree().get_offset();

//                 diagonal_dense_blocks[b]->get_dense_data()->add_vector_product_symmetric(trans, alpha, in + offset_j - source_offset, local_beta, temp.data() + (offset_i - target_offset), m_block_tree_properties->m_UPLO, m_block_tree_properties->m_symmetry);
//             }
//         }

// #if defined(_OPENMP)
// #    pragma omp critical
// #endif

//         Blas<CoefficientPrecision>::axpy(&target_size, &da, temp.data(), &incx, out, &incy);
//     }
// }

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::threaded_hierarchical_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {

    set_leaves_in_cache();

    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    int out_size(m_target_cluster->get_size());
    auto get_output_cluster{&HMatrix::get_target_cluster};
    auto get_input_cluster{&HMatrix::get_source_cluster};
    int local_input_offset  = m_source_cluster->get_offset();
    int local_output_offset = m_target_cluster->get_offset();
    char trans_sym          = (m_symmetry_type_for_leaves == 'S') ? 'T' : 'C';

    if (trans != 'N') {
        out_size            = m_source_cluster->get_size();
        get_input_cluster   = &HMatrix::get_target_cluster;
        get_output_cluster  = &HMatrix::get_source_cluster;
        local_input_offset  = m_target_cluster->get_offset();
        local_output_offset = m_source_cluster->get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        // TODO: use blas
        std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
    }

// Contribution champ lointain
#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<CoefficientPrecision> temp(out_size, 0);
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < m_leaves.size(); b++) {
            int input_offset  = (m_leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (m_leaves[b]->*get_output_cluster)().get_offset();
            m_leaves[b]->add_vector_product(trans, 1, in + input_offset - local_input_offset, 1, temp.data() + (output_offset - local_output_offset));
        }

        // Symmetry part of the diagonal part
        if (m_symmetry_type_for_leaves != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < m_leaves_for_symmetry.size(); b++) {
                int input_offset  = (m_leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (m_leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                m_leaves_for_symmetry[b]->add_vector_product(trans_sym, 1, in + output_offset - local_input_offset, 1, temp.data() + (input_offset - local_output_offset));
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrix<CoefficientPrecision, CoordinatePrecision>::threaded_hierarchical_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {

    set_leaves_in_cache();

    if ((trans == 'T' && m_symmetry_type_for_leaves == 'H')
        || (trans == 'C' && m_symmetry_type_for_leaves == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (" + std::string(1, trans) + " with " + m_symmetry_type_for_leaves + ")"); // LCOV_EXCL_LINE
        // throw std::invalid_argument("[Htool error] Operation is not supported (" + std::string(1, trans) + " with " + m_symmetry_type_for_leaves + ")");                  // LCOV_EXCL_LINE
    }

    int out_size(m_target_cluster->get_size() * mu);
    auto get_output_cluster{&HMatrix::get_target_cluster};
    auto get_input_cluster{&HMatrix::get_source_cluster};
    int local_output_offset = m_target_cluster->get_offset();
    int local_input_offset  = m_source_cluster->get_offset();
    char trans_sym          = (m_symmetry_type_for_leaves == 'S') ? 'T' : 'C';

    if (trans != 'N') {
        out_size            = m_source_cluster->get_size() * mu;
        get_input_cluster   = &HMatrix::get_target_cluster;
        get_output_cluster  = &HMatrix::get_source_cluster;
        local_input_offset  = m_target_cluster->get_offset();
        local_output_offset = m_source_cluster->get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        // TODO: use blas
        std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
    }

// Contribution champ lointain
#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<CoefficientPrecision> temp(out_size, 0);
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < m_leaves.size(); b++) {
            int input_offset  = (m_leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (m_leaves[b]->*get_output_cluster)().get_offset();
            m_leaves[b]->add_matrix_product_row_major(trans, 1, in + (input_offset - local_input_offset) * mu, 1, temp.data() + (output_offset - local_output_offset) * mu, mu);
        }

        // Symmetry part of the diagonal part
        if (m_symmetry_type_for_leaves != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < m_leaves_for_symmetry.size(); b++) {
                int input_offset  = (m_leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (m_leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                m_leaves_for_symmetry[b]->add_matrix_product_row_major(trans_sym, 1, in + (output_offset - local_input_offset) * mu, 1, temp.data() + (input_offset - local_output_offset) * mu, mu);
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
}

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu) const {
//     std::fill(out, out + this->nc * mu, 0);
//     int incx(1), incy(1);
//     int global_size_rhs = this->nc * mu;
//     T da(1);

//     // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(this->nc * mu, 0);
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//         for (int b = 0; b < MyComputedBlocks.size(); b++) {
//             int offset_i = MyComputedBlocks[b]->get_target_cluster()->get_offset();
//             int offset_j = MyComputedBlocks[b]->get_source_cluster()->get_offset();
//             if (!(m_symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
//                 MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_i - local_offset) * mu, temp.data() + offset_j * mu, mu, 'T', 'T');
//             }
//         }

// #if defined(_OPENMP)
// #    pragma omp critical
// #endif
//         Blas<CoefficientPrecision>::axpy(&(global_size_rhs), &da, temp.data(), &incx, out, &incy);
//     }

//     MPI_Allreduce(MPI_IN_PLACE, out, this->nc * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_to_global_target(const T *const in, T *const out, const int &mu) const {
//     // Allgather
//     std::vector<int> recvcounts(sizeWorld);
//     std::vector<int> displs(sizeWorld);

//     displs[0] = 0;

//     for (int i = 0; i < sizeWorld; i++) {
//         recvcounts[i] = (cluster_tree_t->get_masteroffset(i).second) * mu;
//         if (i > 0)
//             displs[i] = displs[i - 1] + recvcounts[i - 1];
//     }

//     MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_to_global_source(const T *const in, T *const out, const int &mu) const {
//     // Allgather
//     std::vector<int> recvcounts(sizeWorld);
//     std::vector<int> displs(sizeWorld);

//     displs[0] = 0;

//     for (int i = 0; i < sizeWorld; i++) {
//         recvcounts[i] = (cluster_tree_s->get_masteroffset(i).second) * mu;
//         if (i > 0)
//             displs[i] = displs[i - 1] + recvcounts[i - 1];
//     }
//     MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mymvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     double time      = MPI_Wtime();
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[this->nc * mu];
//         need_delete = true;
//     }
//     this->local_to_global_source(in, work, mu);
//     this->mymvprod_global_to_local(work, out, mu);

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

//     if (this->m_symmetry == 'S' || this->m_symmetry == 'H') {
//         this->mymvprod_local_to_local(in, out, mu, work);
//         return;
//     }

//     double time      = MPI_Wtime();
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[(this->nc + local_size_source * sizeWorld) * mu];
//         need_delete = true;
//     }

//     std::fill(out, out + local_size_source * mu, 0);
//     int incx(1), incy(1);
//     int global_size_rhs = this->nc * mu;
//     T da(1);

//     std::fill(work, work + this->nc * mu, 0);
//     T *rbuf = work + this->nc * mu;

//     // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(this->nc * mu, 0);
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided) nowait
// #endif
//         for (int b = 0; b < MyComputedBlocks.size(); b++) {
//             int offset_i = MyComputedBlocks[b]->get_target_cluster()->get_offset();
//             int offset_j = MyComputedBlocks[b]->get_source_cluster()->get_offset();
//             if (!(m_symmetry != 'N') || offset_i != offset_j) { // remove strictly diagonal blocks
//                 MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_i - local_offset) * mu, temp.data() + offset_j * mu, mu, 'T', 'T');
//             }
//         }

// #if defined(_OPENMP)
// #    pragma omp critical
// #endif
//         Blas<CoefficientPrecision>::axpy(&(global_size_rhs), &da, temp.data(), &incx, work, &incy);
//     }

//     std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
//     std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);

//     sdispls[0] = 0;
//     rdispls[0] = 0;

//     for (int i = 0; i < sizeWorld; i++) {
//         scounts[i] = (cluster_tree_s->get_masteroffset(i).second) * mu;
//         rcounts[i] = (local_size_source)*mu;
//         if (i > 0) {
//             sdispls[i] = sdispls[i - 1] + scounts[i - 1];
//             rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
//         }
//     }

//     MPI_Alltoallv(work, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

//     for (int i = 0; i < sizeWorld; i++)
//         std::transform(out, out + local_size_source * mu, rbuf + rdispls[i], out, std::plus<CoefficientPrecision>());

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_global_to_global(const T *const in, T *const out, const int &mu) const {
//     double time = MPI_Wtime();

//     if (mu == 1) {
//         std::vector<CoefficientPrecision> out_perm(local_size);
//         std::vector<CoefficientPrecision> buffer(std::max(nc, nr));

//         // Permutation
//         if (use_permutation) {
//             this->source_to_cluster_permutation(in, buffer.data());
//             mymvprod_global_to_local(buffer.data(), out_perm.data(), 1);

//         } else {
//             mymvprod_global_to_local(in, out_perm.data(), 1);
//         }

//         // Allgather
//         std::vector<int> recvcounts(sizeWorld);
//         std::vector<int> displs(sizeWorld);

//         displs[0] = 0;

//         for (int i = 0; i < sizeWorld; i++) {
//             recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
//             if (i > 0)
//                 displs[i] = displs[i - 1] + recvcounts[i - 1];
//         }

//         if (use_permutation) {
//             MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

//             // Permutation
//             this->cluster_to_target_permutation(buffer.data(), out);
//         } else {
//             MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
//         }

//     } else {

//         std::vector<CoefficientPrecision> in_perm(std::max(nr, nc) * mu * 2);
//         std::vector<CoefficientPrecision> out_perm(local_size * mu);
//         std::vector<CoefficientPrecision> buffer(nc);

//         for (int i = 0; i < mu; i++) {
//             // Permutation
//             if (use_permutation) {
//                 this->source_to_cluster_permutation(in + i * nc, buffer.data());
//                 // Transpose
//                 for (int j = 0; j < nc; j++) {
//                     in_perm[i + j * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = 0; j < nc; j++) {
//                     in_perm[i + j * mu] = in[j + i * nc];
//                 }
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(in_perm.data(), nc * mu);
//         }

//         mymvprod_global_to_local(in_perm.data(), in_perm.data() + nc * mu, mu);

//         // Tranpose
//         for (int i = 0; i < mu; i++) {
//             for (int j = 0; j < local_size; j++) {
//                 out_perm[i * local_size + j] = in_perm[i + j * mu + nc * mu];
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(out_perm.data(), out_perm.size());
//         }

//         // Allgather
//         std::vector<int> recvcounts(sizeWorld);
//         std::vector<int> displs(sizeWorld);

//         displs[0] = 0;

//         for (int i = 0; i < sizeWorld; i++) {
//             recvcounts[i] = cluster_tree_t->get_masteroffset(i).second * mu;
//             if (i > 0)
//                 displs[i] = displs[i - 1] + recvcounts[i - 1];
//         }

//         MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), in_perm.data() + mu * nr, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 for (int j = 0; j < sizeWorld; j++) {
//                     std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, in_perm.data() + i * nr + displs[j] / mu);
//                 }

//                 // Permutation
//                 this->cluster_to_target_permutation(in_perm.data() + i * nr, out + i * nr);
//             } else {
//                 for (int j = 0; j < sizeWorld; j++) {
//                     std::copy_n(in_perm.data() + mu * nr + displs[j] + i * recvcounts[j] / mu, recvcounts[j] / mu, out + i * nr + displs[j] / mu);
//                 }
//             }
//         }
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     double time      = MPI_Wtime();
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[this->nc * mu];
//         need_delete = true;
//     }

//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;

//     if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
//         throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
//     }
//     if (mu == 1) {
//         std::vector<CoefficientPrecision> in_perm(local_size_source), out_perm(local_size);

//         // local permutation
//         if (use_permutation) {
//             // permutation
//             this->local_source_to_local_cluster(in, in_perm.data());

//             // prod
//             mymvprod_local_to_local(in_perm.data(), out_perm.data(), 1, work);

//             // permutation
//             this->local_cluster_to_local_target(out_perm.data(), out, comm);

//         } else {
//             mymvprod_local_to_local(in, out, 1, work);
//         }

//     } else {

//         std::vector<CoefficientPrecision> in_perm(local_size_source * mu);
//         std::vector<CoefficientPrecision> out_perm(local_size * mu);
//         std::vector<CoefficientPrecision> buffer(std::max(local_size_source, local_size));

//         for (int i = 0; i < mu; i++) {
//             // local permutation
//             if (use_permutation) {
//                 this->local_source_to_local_cluster(in + i * local_size_source, buffer.data());

//                 // Transpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     in_perm[i + j * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     in_perm[i + j * mu] = in[j + i * local_size_source];
//                 }
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(in_perm.data(), local_size_source * mu);
//         }

//         mymvprod_local_to_local(in_perm.data(), out_perm.data(), mu, work);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 // Tranpose
//                 for (int j = 0; j < local_size; j++) {
//                     buffer[j] = out_perm[i + j * mu];
//                 }

//                 // local permutation
//                 this->local_cluster_to_local_target(buffer.data(), out + i * local_size);
//             } else {
//                 // Tranpose
//                 for (int j = 0; j < local_size; j++) {
//                     out[j + i * local_size] = out_perm[i + j * mu];
//                 }
//             }
//         }

//         if (m_symmetry == 'H') {
//             conj_if_complex(out, out_perm.size());
//         }
//     }

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu) const {
//     double time = MPI_Wtime();
//     if (this->m_symmetry == 'S') {
//         this->mvprod_global_to_global(in, out, mu);
//         return;
//     } else if (this->m_symmetry == 'H') {
//         std::vector<CoefficientPrecision> in_conj(in, in + nr * mu);
//         conj_if_complex(in_conj.data(), nr * mu);
//         this->mvprod_global_to_global(in_conj.data(), out, mu);
//         conj_if_complex(out, mu * nc);
//         return;
//     }
//     if (mu == 1) {

//         if (use_permutation) {
//             std::vector<CoefficientPrecision> in_perm(nr), out_perm(nc);

//             // permutation
//             this->target_to_cluster_permutation(in, in_perm.data());

//             mymvprod_transp_local_to_global(in_perm.data() + local_offset, out_perm.data(), 1);

//             // permutation
//             this->cluster_to_source_permutation(out_perm.data(), out);
//         } else {
//             mymvprod_transp_local_to_global(in + local_offset, out, 1);
//         }

//     } else {

//         std::vector<CoefficientPrecision> out_perm(mu * nc);
//         std::vector<CoefficientPrecision> in_perm(local_size * mu + mu * nc);
//         std::vector<CoefficientPrecision> buffer(nr);

//         for (int i = 0; i < mu; i++) {
//             // Permutation
//             if (use_permutation) {
//                 this->target_to_cluster_permutation(in + i * nr, buffer.data());
//                 // Transpose
//                 for (int j = local_offset; j < local_offset + local_size; j++) {
//                     in_perm[i + (j - local_offset) * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = local_offset; j < local_offset + local_size; j++) {
//                     in_perm[i + (j - local_offset) * mu] = in[j + i * nr];
//                 }
//             }
//         }

//         // It should never happen since we use mvprod_global_to_global in this case
//         // if (m_symmetry == 'H') {
//         //     conj_if_complex(in_perm.data(), local_size * mu);
//         // }

//         mymvprod_transp_local_to_global(in_perm.data(), in_perm.data() + local_size * mu, mu);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 // Transpose
//                 for (int j = 0; j < nc; j++) {
//                     out_perm[i * nc + j] = in_perm[i + j * mu + local_size * mu];
//                 }
//                 cluster_to_source_permutation(out_perm.data() + i * nc, out + i * nc);
//             } else {
//                 for (int j = 0; j < nc; j++) {
//                     out[i * nc + j] = in_perm[i + j * mu + local_size * mu];
//                 }
//             }
//         }

//         // It should never happen since we use mvprod_global_to_global in this case
//         // if (m_symmetry == 'H') {
//         //     conj_if_complex(out, nc * mu);
//         // }
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu, T *work) const {
//     double time           = MPI_Wtime();
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
//     if (this->m_symmetry == 'S') {
//         this->mvprod_local_to_local(in, out, mu);
//         return;
//     } else if (this->m_symmetry == 'H') {
//         std::vector<CoefficientPrecision> in_conj(in, in + local_size * mu);
//         conj_if_complex(in_conj.data(), local_size * mu);
//         this->mvprod_local_to_local(in_conj.data(), out, mu);
//         conj_if_complex(out, mu * local_size_source);
//         return;
//     }
//     bool need_delete = false;
//     if (work == nullptr) {
//         work        = new T[(this->nc + sizeWorld * this->get_source_cluster()->get_local_size()) * mu];
//         need_delete = true;
//     }

//     if (!(cluster_tree_s->IsLocal()) || !(cluster_tree_t->IsLocal())) {
//         throw std::logic_error("[Htool error] Permutation is not local, mvprod_local_to_local cannot be used"); // LCOV_EXCL_LINE
//     }

//     if (mu == 1) {
//         std::vector<CoefficientPrecision> in_perm(local_size), out_perm(local_size_source);

//         // local permutation
//         if (use_permutation) {
//             this->local_target_to_local_cluster(in, in_perm.data());

//             // prod
//             mymvprod_transp_local_to_local(in_perm.data(), out_perm.data(), 1, work);

//             // permutation
//             this->local_cluster_to_local_source(out_perm.data(), out, comm);

//         } else {
//             mymvprod_transp_local_to_local(in, out, 1, work);
//         }

//     } else {

//         std::vector<CoefficientPrecision> in_perm(local_size * mu);
//         std::vector<CoefficientPrecision> out_perm(local_size_source * mu);
//         std::vector<CoefficientPrecision> buffer(std::max(local_size_source, local_size));

//         for (int i = 0; i < mu; i++) {
//             // local permutation
//             if (use_permutation) {
//                 this->local_target_to_local_cluster(in + i * local_size, buffer.data());

//                 // Transpose
//                 for (int j = 0; j < local_size; j++) {
//                     in_perm[i + j * mu] = buffer[j];
//                 }
//             } else {
//                 // Transpose
//                 for (int j = 0; j < local_size; j++) {
//                     in_perm[i + j * mu] = in[j + i * local_size];
//                 }
//             }
//         }
//         // It should never happen since we use mvprod_global_to_global in this case
//         // if (m_symmetry == 'H') {
//         //     conj_if_complex(in_perm.data(), local_size_source * mu);
//         // }

//         mymvprod_transp_local_to_local(in_perm.data(), out_perm.data(), mu, work);

//         for (int i = 0; i < mu; i++) {
//             if (use_permutation) {
//                 // Tranpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     buffer[j] = out_perm[i + j * mu];
//                 }

//                 // local permutation
//                 this->local_cluster_to_local_source(buffer.data(), out + i * local_size_source);
//             } else {
//                 // Tranpose
//                 for (int j = 0; j < local_size_source; j++) {
//                     out[j + i * local_size_source] = out_perm[i + j * mu];
//                 }
//             }
//         }
//         // It should never happen since we use mvprod_global_to_global in this case
//         // if (m_symmetry == 'H') {
//         //     conj_if_complex(out, out_perm.size());
//         // }
//     }

//     if (need_delete) {
//         delete[] work;
//         work = nullptr;
//     }
//     // Timing
//     infos["nb_mat_vec_prod"]         = NbrToStr(1 + StrToNbr<int>(infos["nb_mat_vec_prod"]));
//     infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime() - time + StrToNbr<double>(infos["total_time_mat_vec_prod"]));
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const {
//     std::fill(out, out + local_size * mu, 0);

//     // Contribution champ lointain
// #if defined(_OPENMP)
// #    pragma omp parallel
// #endif
//     {
//         std::vector<CoefficientPrecision> temp(local_size * mu, 0);
//         // To localize the rhs with multiple rhs, it is transpose. So instead of A*B, we do transpose(B)*transpose(A)
//         char transb = 'T';
//         // In case of a hermitian matrix, the rhs is conjugate transpose
//         if (m_symmetry == 'H') {
//             transb = 'C';
//         }
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided)
// #endif
//         for (int b = 0; b < MyComputedBlocks.size(); b++) {
//             int offset_i = MyComputedBlocks[b]->get_target_cluster()->get_offset();
//             int offset_j = MyComputedBlocks[b]->get_source_cluster()->get_offset();
//             int size_j   = MyComputedBlocks[b]->get_source_cluster()->get_size();

//             if ((offset_j < offset + size && offset < offset_j + size_j) && (m_symmetry == 'N' || offset_i != offset_j)) {
//                 if (offset_j - offset < 0) {
//                     std::cout << "TEST "
//                               << " " << offset_j << " " << size_j << " "
//                               << " " << offset << " " << size << " "
//                               << " " << rankWorld << " "
//                               << offset_j - offset << std::endl;
//                 }
//                 MyComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb);
//             }
//         }

//         // Symmetry part of the diagonal part
//         if (m_symmetry != 'N') {
//             transb      = 'N';
//             char op_sym = 'T';
//             if (m_symmetry == 'H') {
//                 op_sym = 'C';
//             }
// #if defined(_OPENMP)
// #    pragma omp for schedule(guided)
// #endif
//             for (int b = 0; b < MyDiagComputedBlocks.size(); b++) {
//                 int offset_i = MyDiagComputedBlocks[b]->get_source_cluster()->get_offset();
//                 int offset_j = MyDiagComputedBlocks[b]->get_target_cluster()->get_offset();
//                 int size_j   = MyDiagComputedBlocks[b]->get_target_cluster()->get_size();

//                 if ((offset_j < offset + size && offset < offset_j + size_j) && offset_i != offset_j) { // remove strictly diagonal blocks
//                     MyDiagComputedBlocks[b]->get_block_data()->add_mvprod_row_major(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, transb, op_sym);
//                 }
//             }

// #if defined(_OPENMP)
// #    pragma omp for schedule(guided)
// #endif
//             for (int b = 0; b < MyStrictlyDiagNearFieldMats.size(); b++) {
//                 const Block<CoefficientPrecision> *M = MyStrictlyDiagNearFieldMats[b];
//                 int offset_i      = M->get_source_cluster()->get_offset();
//                 int offset_j      = M->get_target_cluster()->get_offset();
//                 int size_j        = M->get_source_cluster()->get_size();
//                 if (offset_j < offset + size && offset < offset_j + size_j) {
//                     M->get_dense_block_data()->add_mvprod_row_major_sym(in + (offset_j - offset + margin) * mu, temp.data() + (offset_i - local_offset) * mu, mu, this->UPLO, this->symmetry);
//                 }
//             }
//         }
// #if defined(_OPENMP)
// #    pragma omp critical
// #endif
//         std::transform(temp.begin(), temp.end(), out, out, std::plus<CoefficientPrecision>());
//     }

//     if (!(this->cluster_tree_s->get_local_offset() < offset + size && offset < this->cluster_tree_s->get_local_offset() + this->cluster_tree_s->get_local_size()) && this->OffDiagonalApproximation != nullptr) {
//         std::vector<CoefficientPrecision> off_diagonal_out(cluster_tree_t->get_local_size() * mu, 0);
//         int off_diagonal_offset = (offset < this->cluster_tree_s->get_local_offset()) ? offset : offset - this->cluster_tree_s->get_local_size();
//         if (mu > 1 && !this->OffDiagonalApproximation->IsUsingRowMajorStorage()) { // Need to transpose input and output for OffDiagonalApproximation
//             std::vector<CoefficientPrecision> off_diagonal_input_column_major(size * mu, 0);

//             for (int i = 0; i < mu; i++) {
//                 for (int j = 0; j < size; j++) {
//                     off_diagonal_input_column_major[j + i * size] = in[i + j * mu];
//                 }
//             }

//             if (m_symmetry == 'H') {
//                 conj_if_complex(off_diagonal_input_column_major.data(), size * mu);
//             }

//             this->OffDiagonalApproximation->mvprod_subrhs_to_local(off_diagonal_input_column_major.data(), off_diagonal_out.data(), mu, off_diagonal_offset, size);

//             if (m_symmetry == 'H') {
//                 conj_if_complex(off_diagonal_out.data(), off_diagonal_out.size());
//             }
//             for (int i = 0; i < mu; i++) {
//                 for (int j = 0; j < local_size; j++) {
//                     out[i + j * mu] += off_diagonal_out[i * local_size + j];
//                 }
//             }
//         } else {
//             this->OffDiagonalApproximation->mvprod_subrhs_to_local(in, off_diagonal_out.data(), mu, off_diagonal_offset, size);

//             int incx(1), incy(1), local_size_rhs(local_size * mu);
//             T da(1);

//             Blas<CoefficientPrecision>::axpy(&local_size_rhs, &da, off_diagonal_out.data(), &incx, out, &incy);
//         }
//     }
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::source_to_cluster_permutation(const T *const in, T *const out) const {
//     global_to_cluster(cluster_tree_m_source_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::target_to_cluster_permutation(const T *const in, T *const out) const {
//     global_to_cluster(cluster_tree_m_target_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::cluster_to_target_permutation(const T *const in, T *const out) const {
//     cluster_to_global(cluster_tree_m_target_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::cluster_to_source_permutation(const T *const in, T *const out) const {
//     cluster_to_global(cluster_tree_m_source_cluster_tree->get(), in, out);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_target_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
//     local_to_local_cluster(cluster_tree_m_target_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_source_to_local_cluster(const T *const in, T *const out, MPI_Comm comm) const {
//     local_to_local_cluster(cluster_tree_m_source_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_cluster_to_local_target(const T *const in, T *const out, MPI_Comm comm) const {
//     local_cluster_to_local(cluster_tree_m_target_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::local_cluster_to_local_source(const T *const in, T *const out, MPI_Comm comm) const {
//     local_cluster_to_local(cluster_tree_m_source_cluster_tree->get(), in, out, comm);
// }

// template <typename CoefficientPrecision>
// std::vector<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::operator*(const std::vector<CoefficientPrecision> &x) const {
//     assert(x.size() == nc);
//     std::vector<CoefficientPrecision> result(nr, 0);
//     mvprod_global_to_global(x.data(), result.data(), 1);
//     return result;
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// underlying_type<CoefficientPrecision> HMatrix<CoefficientPrecision, CoordinatePrecision>::compressed_size() const {

//     double my_compressed_size = 0.;
//     double nr_b, nc_b, rank;
//     const auto &low_rank_leaves = m_block_tree_properties->m_low_rank_leaves;
//     const auto &dense_leaves    = m_block_tree_properties->m_dense_leaves;
//     std::cout << dense_leaves.size() << " " << low_rank_leaves.size() << "\n";
//     for (int j = 0; j < low_rank_leaves.size(); j++) {
//         nr_b = low_rank_leaves[j]->get_target_cluster_tree().get_size();
//         nc_b = low_rank_leaves[j]->get_source_cluster_tree().get_size();
//         rank = low_rank_leaves[j]->get_low_rank_data()->rank_of();
//         my_compressed_size += rank * (nr_b + nc_b);
//     }

//     for (int j = 0; j < dense_leaves.size(); j++) {
//         nr_b = dense_leaves[j]->get_target_cluster_tree().get_size();
//         nc_b = dense_leaves[j]->get_source_cluster_tree().get_size();
//         if (dense_leaves[j]->get_target_cluster_tree().get_offset() == dense_leaves[j]->get_source_cluster_tree().get_offset() && m_block_tree_properties->m_symmetry != 'N' && nr_b == nc_b) {
//             my_compressed_size += (nr_b * (nc_b + 1)) / 2;
//         } else {
//             my_compressed_size += nr_b * nc_b;
//         }
//     }

//     return my_compressed_size;
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::print_infos() const {
//     int rankWorld;
//     MPI_Comm_rank(comm, &rankWorld);

//     if (rankWorld == 0) {
//         for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
//             std::cout << it->first << "\t" << it->second << std::endl;
//         }
//         std::cout << std::endl;
//     }
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::save_infos(const std::string &outputname, std::ios_base::openmode mode, const std::string &sep) const {
//     int rankWorld;
//     MPI_Comm_rank(comm, &rankWorld);

//     if (rankWorld == 0) {
//         std::ofstream outputfile(outputname, mode);
//         if (outputfile) {
//             for (std::map<std::string, std::string>::const_iterator it = infos.begin(); it != infos.end(); ++it) {
//                 outputfile << it->first << sep << it->second << std::endl;
//             }
//             outputfile.close();
//         } else {
//             std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
//         }
//     }
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::save_plot(const std::string &outputname) const {

//     std::ofstream outputfile((outputname + ".csv").c_str());

//     if (outputfile) {
//         const auto &output = get_output();
//         outputfile << m_block_tree_properties->m_root_cluster_tree_target->get_size() << "," << m_block_tree_properties->m_root_cluster_tree_source->get_size() << std::endl;
//         for (const auto &block : output) {
//             outputfile << block << "\n";
//         }
//         outputfile.close();
//     } else {
//         std::cout << "Unable to create " << outputname << std::endl; // LCOV_EXCL_LINE
//     }
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// std::vector<DisplayBlock> HMatrix<CoefficientPrecision, CoordinatePrecision>::get_output() const {

//     std::vector<DisplayBlock> output(m_block_tree_properties->m_tasks.size());
//     int index = 0;
//     for (const auto &task : m_block_tree_properties->m_tasks) {
//         output[index].target_offset = taskm_target_clusterree().get_offset() - m_block_tree_properties->m_root_cluster_tree_target->get_offset();
//         output[index].target_size   = taskm_target_clusterree().get_size();
//         output[index].source_offset = task->get_m_t.().get_offset() - m_block_tree_properties->m_root_cluster_tree_source->get_offset();
//         output[index].source_size   = task->get_m_t.().get_size();
//         output[index].rank          = task->is_dense() ? -1 : task->m_low_rank_data->rank_of();
//         index++;
//     }

//     return output;
// }

// template <typename CoefficientPrecision>
// underlying_type<CoefficientPrecision> Frobenius_absolute_error(const HMatrix<CoefficientPrecision,CoordinatePrecision> &B, const VirtualGenerator<CoefficientPrecision> &A) {
//     underlying_type<CoefficientPrecision> myerr = 0;
//     for (int j = 0; j < B.MyFarFieldMats.size(); j++) {
//         underlying_type<CoefficientPrecision> test = Frobenius_absolute_error<CoefficientPrecision>(*(B.MyFarFieldMats[j]), *(B.MyFarFieldMats[j]->get_low_rank_block_data()), A);
//         myerr += std::pow(test, 2);
//     }

//     underlying_type<CoefficientPrecision> err = 0;
//     MPI_Allreduce(&myerr, &err, 1, wrapper_mpi<CoefficientPrecision>::mpi_underlying_type(), MPI_SUM, B.comm);

//     return std::sqrt(err);
// }

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_dense() const {
//     Matrix<CoefficientPrecision> Dense(local_size, nc);
//     // Internal dense blocks
//     for (int l = 0; l < MyNearFieldMats.size(); l++) {
//         const Block<CoefficientPrecision> *submat = MyNearFieldMats[l];
//         int local_nr           = submat->get_target_cluster()->get_size();
//         int local_nc           = submat->get_source_cluster()->get_size();
//         int offset_i           = submat->get_target_cluster()->get_offset();
//         int offset_j           = submat->get_source_cluster()->get_offset();
//         for (int k = 0; k < local_nc; k++) {
//             std::copy_n(&(submat->get_dense_block_data()->operator()(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
//         }
//     }

//     // Internal compressed block
//     for (int l = 0; l < MyFarFieldMats.size(); l++) {
//         const Block<CoefficientPrecision> *lmat = MyFarFieldMats[l];
//         int local_nr         = lmat->get_target_cluster()->get_size();
//         int local_nc         = lmat->get_source_cluster()->get_size();
//         int offset_i         = lmat->get_target_cluster()->get_offset();
//         int offset_j         = lmat->get_source_cluster()->get_offset();
//         Matrix<CoefficientPrecision> FarFielBlock(local_nr, local_nc);
//         lmat->get_low_rank_block_data()->get_whole_matrix(&(FarFielBlock(0, 0)));
//         for (int k = 0; k < local_nc; k++) {
//             std::copy_n(&(FarFielBlock(0, k)), local_nr, Dense.data() + (offset_i - local_offset) + (offset_j + k) * local_size);
//         }
//     }
//     return Dense;
// }

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_dense_perm() const {
//     Matrix<CoefficientPrecision> Dense(local_size, nc);
//     copy_local_dense_perm(Dense.data());
//     return Dense;
// }

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_to_dense(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();
    int target_size   = hmatrix.get_target_cluster().get_size();

    for (auto leaf : hmatrix.get_leaves()) {
        int local_nr   = leaf->get_target_cluster().get_size();
        int local_nc   = leaf->get_source_cluster().get_size();
        int row_offset = leaf->get_target_cluster().get_offset() - target_offset;
        int col_offset = leaf->get_source_cluster().get_offset() - source_offset;
        if (leaf->is_dense()) {
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[j + row_offset + (k + col_offset) * target_size] = (*leaf->get_dense_data())(j, k);
                }
            }
        } else {

            Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[j + row_offset + (k + col_offset) * target_size] = low_rank_to_dense(j, k);
                }
            }
        }
    }

    // char symmetry_type = hmatrix.get_symmetry_for_leaves();
    // for (auto leaf : hmatrix.get_leaves_for_symmetry()) {
    //     int local_nr   = leaf->get_target_cluster().get_size();
    //     int local_nc   = leaf->get_source_cluster().get_size();
    //     int col_offset = leaf->get_target_cluster().get_offset() - source_offset;
    //     int row_offset = leaf->get_source_cluster().get_offset() - target_offset;
    //     if (leaf->is_dense()) {
    //         for (int j = 0; j < local_nr; j++) {
    //             for (int k = 0; k < local_nc; k++) {
    //                 ptr[k + row_offset + (j + col_offset) * target_size] = (*leaf->get_dense_data())(j, k);
    //             }
    //         }
    //     } else {

    //         Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
    //         leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
    //         for (int j = 0; j < local_nr; j++) {
    //             for (int k = 0; k < local_nc; k++) {
    //                 ptr[k + row_offset + (j + col_offset) * target_size] = low_rank_to_dense(j, k);
    //             }
    //         }
    //     }
    //     if (symmetry_type == 'H') {
    //         for (int k = 0; k < local_nc; k++) {
    //             for (int j = 0; j < local_nr; j++) {
    //                 ptr[k + row_offset + (j + col_offset) * target_size] = conj_if_complex(ptr[k + row_offset + (j + col_offset) * target_size]);
    //             }
    //         }
    //     }
    // }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_diagonal(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {
    if (hmatrix.get_target_cluster().get_offset() != hmatrix.get_source_cluster().get_offset() || hmatrix.get_target_cluster().get_size() != hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal cannot be used");                       // LCOV_EXCL_LINE
    }

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();

    for (auto leaf : hmatrix.get_leaves()) {
        int local_nr = leaf->get_target_cluster().get_size();
        int local_nc = leaf->get_source_cluster().get_size();
        int offset_i = leaf->get_target_cluster().get_offset() - target_offset;
        int offset_j = leaf->get_source_cluster().get_offset() - source_offset;
        if (leaf->is_dense() && offset_i == offset_j) {
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[k + offset_i] = (*leaf->get_dense_data())(k, k);
            }
        } else if (leaf->is_low_rank() && offset_i == offset_j) { // pretty rare...
            Matrix<CoefficientPrecision> low_rank_to_dense(local_nc, local_nr);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[k + offset_i] = (low_rank_to_dense)(k, k);
            }
        }
    }
}

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_interaction(bool permutation) const {
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
//     Matrix<CoefficientPrecision> local_interaction(local_size, local_size_source);
//     copy_local_interaction(local_interaction.data(), permutation);
//     return local_interaction;
// }

// template <typename CoefficientPrecision>
// Matrix<CoefficientPrecision> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_local_diagonal_block(bool permutation) const {
//     int local_size_source = cluster_tree_s->get_masteroffset(rankWorld).second;
//     Matrix<CoefficientPrecision> diagonal_block(local_size, local_size_source);
//     copy_local_diagonal_block(diagonal_block.data(), permutation);
//     return diagonal_block;
// }

// template <typename CoefficientPrecision, typename CoordinatePrecision>
// void HMatrix<CoefficientPrecision, CoordinatePrecision>::copy_local_interaction(T *ptr, bool permutation) const {
//     if ((!(cluster_tree_t->IsLocal()) || !(cluster_tree_s->IsLocal())) && permutation) {
//         throw std::logic_error("[Htool error] Permutation is not local, get_local_interaction cannot be used"); // LCOV_EXCL_LINE
//     }

//     int local_offset_source = cluster_tree_s->get_masteroffset(rankWorld).first;
//     int local_size_source   = cluster_tree_s->get_masteroffset(rankWorld).second;
//     // Internal dense blocks
//     for (int i = 0; i < MyDiagNearFieldMats.size(); i++) {
//         const Block<CoefficientPrecision> *submat = MyDiagNearFieldMats[i];
//         int local_nr                              = submat->get_target_cluster()->get_size();
//         int local_nc                              = submat->get_source_cluster()->get_size();
//         int offset_i                              = submat->get_target_cluster()->get_offset() - local_offset;
//         int offset_j                              = submat->get_source_cluster()->get_offset() - local_offset_source;
//         for (int i = 0; i < local_nc; i++) {
//             std::copy_n(&(submat->get_dense_block_data()->operator()(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
//         }
//     }

//     // Internal compressed block
//     for (int i = 0; i < MyDiagFarFieldMats.size(); i++) {
//         const Block<CoefficientPrecision> *lmat = MyDiagFarFieldMats[i];
//         int local_nr                            = lmat->get_target_cluster()->get_size();
//         int local_nc                            = lmat->get_source_cluster()->get_size();
//         int offset_i                            = lmat->get_target_cluster()->get_offset() - local_offset;
//         int offset_j                            = lmat->get_source_cluster()->get_offset() - local_offset_source;
//         ;
//         Matrix<CoefficientPrecision> FarFielBlock(local_nr, local_nc);
//         lmat->get_low_rank_block_data()->get_whole_matrix(&(FarFielBlock(0, 0)));
//         for (int i = 0; i < local_nc; i++) {
//             std::copy_n(&(FarFielBlock(0, i)), local_nr, ptr + offset_i + (offset_j + i) * local_size);
//         }
//     }

// // Asking for permutation while symmetry!=N means that the block is upper/lower triangular in Htool's numbering, but it is not true in User's numbering

// if (permutation && m_symmetry != 'N') {
//     if (UPLO == 'L' && m_symmetry == 'S') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = 0; j < i; j++) {
//                 ptr[j + i * local_size] = ptr[i + j * local_size];
//             }
//         }
//     }

//     if (UPLO == 'U' && m_symmetry == 'S') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = i + 1; j < local_size_source; j++) {
//                 ptr[j + i * local_size] = ptr[i + j * local_size];
//             }
//         }
//     }
//     if (UPLO == 'L' && m_symmetry == 'H') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = 0; j < i; j++) {
//                 ptr[j + i * local_size] = conj_if_complex(ptr[i + j * local_size]);
//             }
//         }
//     }

//     if (UPLO == 'U' && m_symmetry == 'H') {
//         for (int i = 0; i < local_size; i++) {
//             for (int j = i + 1; j < local_size_source; j++) {
//                 ptr[j + i * local_size] = conj_if_complex(ptr[i + j * local_size]);
//             }
//         }
//     }
// }
// // Permutations
// if (permutation) {
//     Matrix<CoefficientPrecision> diagonal_block_perm(local_size, local_size_source);
//     for (int i = 0; i < local_size; i++) {
//         for (int j = 0; j < local_size_source; j++) {
//             diagonal_block_perm(i, cluster_tree_s->get_global_perm(j + local_offset_source) - local_offset_source) = ptr[i + j * local_size];
//         }
//     }

//     for (int i = 0; i < local_size; i++) {
//         this->local_cluster_to_local_target(diagonal_block_perm.data() + i * local_size, ptr + i * local_size, comm);
//     }
// }
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::copy_local_diagonal_block(T *ptr, bool permutation) const {
//     if (cluster_tree_t != cluster_tree_s) {
//         throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal_block cannot be used"); // LCOV_EXCL_LINE
//     }
//     copy_local_interaction(ptr, permutation);
// }

// template <typename CoefficientPrecision>
// std::pair<int, int> HMatrix<CoefficientPrecision,CoordinatePrecision>::get_max_size_blocks() const {
//     int local_max_size_j = 0;
//     int local_max_size_i = 0;

//     for (int i = 0; i < MyFarFieldMats.size(); i++) {
//         if (local_max_size_j < MyFarFieldMats[i]->get_source_cluster()->get_size())
//             local_max_size_j = MyFarFieldMats[i]->get_source_cluster()->get_size();
//         if (local_max_size_i < MyFarFieldMats[i]->get_target_cluster()->get_size())
//             local_max_size_i = MyFarFieldMats[i]->get_target_cluster()->get_size();
//     }
//     for (int i = 0; i < MyNearFieldMats.size(); i++) {
//         if (local_max_size_j < MyNearFieldMats[i]->get_source_cluster()->get_size())
//             local_max_size_j = MyNearFieldMats[i]->get_source_cluster()->get_size();
//         if (local_max_size_i < MyNearFieldMats[i]->get_target_cluster()->get_size())
//             local_max_size_i = MyNearFieldMats[i]->get_target_cluster()->get_size();
//     }

//     return std::pair<int, int>(local_max_size_i, local_max_size_j);
// }

// template <typename CoefficientPrecision>
// void HMatrix<CoefficientPrecision,CoordinatePrecision>::apply_dirichlet(const std::vector<int> &boundary) {
//     // Renum
//     std::vector<int> boundary_renum(boundary.size());
//     this->source_to_cluster_permutation(boundary.data(), boundary_renum.data());

//     //
//     for (int j = 0; j < MyStrictlyDiagNearFieldMats.size(); j++) {
//         SubMatrix<CoefficientPrecision> &submat = *(MyStrictlyDiagNearFieldMats[j]);
//         int local_nr         = submat.nb_rows();
//         int local_nc         = submat.nb_cols();
//         int offset_i         = submam_target_cluster_tree->get_offset_i();
//         for (int i = offset_i; i < offset_i + std::min(local_nr, local_nc); i++) {
//             if (boundary_renum[i])
//                 submat(i - offset_i, i - offset_i) = 1e30;
//         }
//     }
// }

} // namespace htool
#endif
