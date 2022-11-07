#ifndef HTOOL_CLUSTERING_CLUSTER_HPP
#define HTOOL_CLUSTERING_CLUSTER_HPP

#include "../basic_types/matrix.hpp"
#include "../interfaces/virtual_cluster.hpp"
#include "../misc/user.hpp"
#include <functional>
#include <memory>
#include <mpi.h>
#include <stack>

namespace htool {
template <typename ClusteringType>
class Cluster : public VirtualCluster {
  protected:
    // Local information
    double m_rad;
    std::vector<double> m_ctr;
    std::vector<std::unique_ptr<Cluster>> m_sons; // Sons
    int m_depth;                                  // depth of the current cluster
    int m_rank;                                   // Rank for dofs of the current cluster
    int m_offset;                                 // Offset of the current cluster in the global numbering
    int m_size;
    int m_counter; // numbering of the nodes level-wise
    int m_nb_pt;
    bool m_local_permutation;

    // Global parameters
    int m_space_dim;                                 // dimension for geometric points
    unsigned int m_minclustersize{10};               // minimal number of geometric point in a cluster
    int m_max_depth;                                 // maximum depth of the tree
    int m_min_depth;                                 // minimum depth of the tree
    std::shared_ptr<std::vector<int>> m_permutation; // permutation from htool numbering to user numbering
    Cluster *const m_root;
    Cluster *m_local_cluster;
    std::vector<std::pair<int, int>> m_MasterOffset;
    MPI_Comm m_comm{MPI_COMM_WORLD};

    //
    ClusteringType clustering_type;

  public:
    // Root constructor
    Cluster(int space_dim = 3, MPI_Comm comm = MPI_COMM_WORLD) : m_rad(0), m_ctr(space_dim), m_depth(0), m_offset(0), m_size(0), m_counter(0), m_local_permutation(false), m_space_dim(space_dim), m_max_depth(0), m_min_depth(-1), m_permutation(std::make_shared<std::vector<int>>()), m_root(this), m_comm(comm) {}

    // Node constructor
    Cluster(Cluster *const root, int counter, const int &depth, std::shared_ptr<std::vector<int>> permutation) : m_rad(0), m_ctr(root->m_space_dim), m_depth(depth), m_offset(0), m_size(0), m_counter(counter), m_local_permutation(false), m_space_dim(root->m_space_dim), m_max_depth(-1), m_min_depth(-1), m_minclustersize(root->m_minclustersize), m_permutation(permutation), m_root(root), m_comm(root->m_comm) {}

    // global build cluster tree
    void build(int nb_pt0, const double *const x0, const double *const r0, const double *const g0, int nb_sons = -1) {
        this->m_nb_pt = nb_pt0;

        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(m_comm, &sizeWorld);
        MPI_Comm_rank(m_comm, &rankWorld);

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons)); // LCOV_EXCL_LINE

        // nb_sons=-1 is automatic mode
        if (nb_sons == -1) {
            nb_sons = 2;
        }
        // Initialisation
        this->m_rad   = 0;
        this->m_size  = nb_pt0;
        this->m_nb_pt = nb_pt0;
        this->m_rank  = -1;
        this->m_MasterOffset.resize(sizeWorld);
        this->m_local_permutation = false;
        this->m_depth             = 0; // ce constructeur est appele' juste pour la racine
        this->m_permutation->resize(nb_pt0);
        std::iota(this->m_permutation->begin(), this->m_permutation->end(), 0); // perm[i]=i

        // Recursion
        std::stack<Cluster *> s;
        std::stack<std::vector<int>> n;
        s.push(this);
        n.push(*(this->m_permutation));

        clustering_type.recursive_build(x0, r0, g0, nb_sons, m_comm, s, n);
    }

    void build(int nb_pt0, const double *const x0, int nb_sons = -1) {
        this->m_nb_pt = nb_pt0;
        this->build(nb_pt0, x0, std::vector<double>(nb_pt0, 0).data(), std::vector<double>(nb_pt0, 1).data(), nb_sons);
    }

    // local build cluster tree
    void build(int nb_pt0, const double *const x0, const double *const r0, const double *const g0, const int *const MasterOffset0, int nb_sons = -1) {
        this->m_nb_pt = nb_pt0;
        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(m_comm, &sizeWorld);
        MPI_Comm_rank(m_comm, &rankWorld);

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons)); // LCOV_EXCL_LINE

        // nb_sons=-1 is automatic mode
        if (nb_sons == -1) {
            nb_sons = 2;
        }

        // Initialisation of root
        this->m_rad   = 0;
        this->m_nb_pt = nb_pt0;
        this->m_size  = this->m_nb_pt;
        this->m_rank  = -1;
        this->m_MasterOffset.resize(sizeWorld);
        for (int p = 0; p < sizeWorld; p++) {
            this->m_MasterOffset[p].first  = MasterOffset0[2 * p];
            this->m_MasterOffset[p].second = MasterOffset0[2 * p + 1];
        }
        this->m_permutation->resize(this->m_nb_pt);

        this->m_depth             = 0; // ce constructeur est appele' juste pour la racine
        this->m_local_permutation = true;

        std::iota(this->m_permutation->begin(), this->m_permutation->end(), 0); // perm[i]=i
        // Build level of depth 1 with the given partition and prepare recursion
        std::stack<Cluster *> s;
        std::stack<std::vector<int>> n;

        for (int p = 0; p < sizeWorld; p++) {
            this->m_sons.emplace_back(new Cluster(this, p, this->m_depth + 1, this->m_permutation));
            this->m_sons[p]->set_offset(this->m_MasterOffset[p].first);
            this->m_sons[p]->set_size(this->m_MasterOffset[p].second);
            this->m_sons[p]->set_rank(p);

            if (rankWorld == this->m_sons[p]->get_counter()) {
                this->m_local_cluster = this->m_sons[p].get();
            }

            s.push(this->m_sons[p].get());
            n.push(std::vector<int>(this->m_permutation->begin() + this->m_sons[p]->get_offset(), this->m_permutation->begin() + this->m_sons[p]->get_offset() + this->m_sons[p]->get_size()));
        }

        // Recursion
        clustering_type.recursive_build(x0, r0, g0, nb_sons, m_comm, s, n);
    }

    void build(int nb_pt0, const double *const x0, const int *const MasterOffset0, int nb_sons = -1) {
        this->m_nb_pt = nb_pt0;
        this->build(nb_pt0, x0, std::vector<double>(nb_pt0, 0).data(), std::vector<double>(nb_pt0, 1).data(), MasterOffset0, nb_sons);
    }

    // Local build
    void build_local(int nb_pt0, const double *const x0, int size_partition, const int *const partition, int nb_sons = -1) {

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons)); // LCOV_EXCL_LINE

        // nb_sons=-1 is automatic mode
        if (nb_sons == -1) {
            nb_sons = 2;
        }

        // Initialisation of root
        this->m_rad   = 0;
        this->m_nb_pt = nb_pt0;
        this->m_size  = this->m_nb_pt;
        this->m_rank  = -1;
        this->m_MasterOffset.resize(1);
        this->m_MasterOffset[0].first  = 0;
        this->m_MasterOffset[0].second = m_nb_pt;
        this->m_permutation->resize(this->m_nb_pt);
        this->m_depth             = 0; // ce constructeur est appele' juste pour la racine
        this->m_local_permutation = true;
        this->m_local_cluster     = this->m_root;
        std::iota(this->m_permutation->begin(), this->m_permutation->end(), 0); // perm[i]=i

        // Recursive build
        std::stack<Cluster *> s;
        std::stack<std::vector<int>> n;
        for (int p = 0; p < size_partition; p++) {
            this->m_sons.emplace_back(new Cluster(this, p, this->m_depth + 1, this->m_permutation));
            this->m_sons[p]->set_offset(partition[2 * p]);
            this->m_sons[p]->set_size(partition[2 * p + 1]);
            this->m_sons[p]->set_rank(-1);

            s.push(this->m_sons[p].get());
            n.push(std::vector<int>(this->m_permutation->begin() + this->m_sons[p]->get_offset(), this->m_permutation->begin() + this->m_sons[p]->get_offset() + this->m_sons[p]->get_size()));
        }
    }

    //// Getters for current cluster
    double get_rad() const { return m_rad; }
    const std::vector<double> &get_ctr() const { return m_ctr; }
    const VirtualCluster &get_son(const int &j) const { return *(m_sons[j]); }
    VirtualCluster &get_son(const int &j) { return *(m_sons[j]); }
    Cluster *get_son_ptr(const int &j) { return m_sons[j].get(); }
    const Cluster *get_son_ptr(const int &j) const { return m_sons[j].get(); }
    int get_depth() const { return m_depth; }
    int get_rank() const { return m_rank; }
    int get_offset() const { return m_offset; }
    int get_size() const { return m_size; }
    const int *get_perm_data() const { return m_permutation->data() + m_offset; };
    int *get_perm_data() { return m_permutation->data() + m_offset; };
    int get_nb_sons() const { return m_sons.size(); }
    int get_counter() const { return m_counter; }
    bool is_leaf() const {
        return true ? (m_sons.size() == 0) : false;
    }

    // Getters for local cluster
    const VirtualCluster &get_local_cluster() const {
        return *(m_root->m_local_cluster);
    }
    std::shared_ptr<VirtualCluster> get_local_cluster_tree() {
        int rankWorld;
        MPI_Comm_rank(m_comm, &rankWorld);

        std::shared_ptr<Cluster> copy_local_cluster = std::make_shared<Cluster>();

        copy_local_cluster->m_MasterOffset.push_back(std::make_pair(this->m_MasterOffset[rankWorld].first, this->m_MasterOffset[rankWorld].second));

        copy_local_cluster->m_local_cluster = copy_local_cluster->m_root;

        copy_local_cluster->m_permutation = this->m_permutation;

        // Recursion
        std::stack<Cluster *> cluster_input;
        cluster_input.push(m_local_cluster);
        std::stack<Cluster *> cluster_output;
        cluster_output.push(copy_local_cluster->m_root);
        while (!cluster_input.empty()) {
            Cluster *curr_input  = cluster_input.top();
            Cluster *curr_output = cluster_output.top();

            cluster_input.pop();
            cluster_output.pop();

            curr_output->m_rank   = curr_input->m_rank;
            curr_output->m_ctr    = curr_input->m_ctr;
            curr_output->m_rad    = curr_input->m_rad;
            curr_output->m_offset = curr_input->m_offset;
            curr_output->m_size   = curr_input->m_size;

            int nb_sons = curr_input->m_sons.size();

            for (int p = 0; p < nb_sons; p++) {
                curr_output->m_sons.emplace_back(new Cluster(copy_local_cluster.get(), (curr_output->m_counter) * nb_sons + p, curr_output->m_depth + 1, this->m_permutation));

                cluster_input.push(curr_input->get_son_ptr(p));
                cluster_output.push(curr_output->get_son_ptr(p));
            }
        }

        return copy_local_cluster;
    }

    std::shared_ptr<VirtualCluster> get_cluster_tree() const {
        std::shared_ptr<Cluster> copy_cluster = std::make_shared<Cluster>();

        copy_cluster->m_MasterOffset = this->m_MasterOffset;

        copy_cluster->m_permutation = this->m_permutation;

        // Recursion
        std::stack<const Cluster *> cluster_input;
        cluster_input.push(this);
        std::stack<Cluster *> cluster_output;
        cluster_output.push(copy_cluster->m_root);
        while (!cluster_input.empty()) {
            const Cluster *curr_input = cluster_input.top();
            Cluster *curr_output      = cluster_output.top();

            cluster_input.pop();
            cluster_output.pop();

            curr_output->m_rank   = curr_input->m_rank;
            curr_output->m_ctr    = curr_input->m_ctr;
            curr_output->m_rad    = curr_input->m_rad;
            curr_output->m_offset = curr_input->m_offset;
            curr_output->m_size   = curr_input->m_size;

            int nb_sons = curr_input->m_sons.size();

            for (int p = 0; p < nb_sons; p++) {
                curr_output->m_sons.emplace_back(new Cluster(copy_cluster.get(), (curr_output->m_counter) * nb_sons + p, curr_output->m_depth + 1, this->m_permutation));

                cluster_input.push(curr_input->get_son_ptr(p));
                cluster_output.push(curr_output->get_son_ptr(p));
            }
        }

        return copy_cluster;
    }

    std::vector<int> get_local_perm() const {
        if (!m_local_permutation) {
            throw std::logic_error("[Htool error] Permutation is not local, get_local_perm cannot be used"); // LCOV_EXCL_LINE
        } else {
            return std::vector<int>(m_permutation->data() + m_root->m_local_cluster->get_offset(), m_permutation->data() + m_root->m_local_cluster->get_offset() + m_root->m_local_cluster->get_size());
        }
    }
    int get_local_offset() const { return m_root->m_local_cluster->get_offset(); }
    int get_local_size() const { return m_root->m_local_cluster->get_size(); }
    const std::vector<std::pair<int, int>> &get_masteroffset() const { return m_root->m_MasterOffset; }
    std::pair<int, int> get_local_masteroffset() const {
        int rankWorld;
        MPI_Comm_rank(m_comm, &rankWorld);
        return m_root->m_MasterOffset[rankWorld];
    }
    std::pair<int, int> get_masteroffset_on_rank(int i) const { return m_root->m_MasterOffset[i]; }
    bool is_local() const { return m_local_permutation; }

    //// Getters for global data
    int get_space_dim() const { return m_root->m_space_dim; }
    int get_minclustersize() const { return m_root->m_minclustersize; }
    int get_max_depth() const { return m_root->m_max_depth; }
    int get_min_depth() const { return m_root->m_min_depth; }
    const std::vector<int> &get_global_perm() const { return *m_permutation; };
    std::shared_ptr<std::vector<int>> get_perm_ptr() const { return m_permutation; };
    const int *get_global_perm_data() const { return m_permutation->data(); };
    int *get_global_perm_data() { return m_permutation->data(); };
    int get_global_perm(int i) const { return (*m_permutation)[i]; };
    const VirtualCluster *get_root() const {
        return m_root;
    }

    //// Setters
    void set_rank(int rank) { m_rank = rank; }
    void set_offset(int offset) { m_offset = offset; }
    void set_size(int size) { m_size = size; }
    void set_rad(double rad) { m_rad = rad; }
    void set_ctr(std::vector<double> ctr) { m_ctr = ctr; }
    void set_local_cluster(Cluster *local_cluster) { m_root->m_local_cluster = local_cluster; }
    void set_MasterOffset(int p, std::pair<int, int> pair) { m_root->m_MasterOffset[p] = pair; }

    void add_son(int counter0, const int &dep, std::shared_ptr<std::vector<int>> permutation0) {
        m_sons.emplace_back(new Cluster(m_root, counter0, dep, permutation0));
    }

    void set_max_depth(int max_depth) const { m_root->m_max_depth = max_depth; }
    void set_min_depth(int min_depth) const { m_root->m_min_depth = min_depth; }
    void set_minclustersize(unsigned int minclustersize) {
        if (minclustersize == 0) {
            throw std::invalid_argument("[Htool error] MinClusterSize parameter cannot be zero"); // LCOV_EXCL_LINE
        }
        m_minclustersize = minclustersize;
    }

    void clear_sons() {
        m_sons.clear();
        m_sons.resize(0);
    }

    // Output
    void print() const {
        int rankWorld;
        MPI_Comm_rank(m_comm, &rankWorld);
        if (rankWorld == 0) {
            if (!m_permutation->empty()) {
                std::cout << '[';
                for (std::vector<int>::const_iterator i = m_permutation->begin() + m_offset; i != m_permutation->begin() + m_offset + m_size; ++i)
                    std::cout << *i << ',';
                std::cout << "\b]" << std::endl;
                ;
            }

            for (auto &son : this->m_sons) {
                if (son != NULL)
                    (*son).print();
            }
        }
    }
    void save_geometry(const double *const x0, std::string filename, const std::vector<int> &depths) const {
        int rankWorld;
        MPI_Comm_rank(m_comm, &rankWorld);
        if (rankWorld == 0) {

            std::stack<VirtualCluster const *> s;
            s.push(this);
            std::ofstream output(filename + ".csv");

            std::vector<std::vector<int>> outputs(depths.size());
            std::vector<int> counters(depths.size(), 0);
            for (int p = 0; p < outputs.size(); p++) {
                outputs[p].resize(m_permutation->size());
            }

            // Permuted geometric points
            for (int d = 0; d < this->m_space_dim; d++) {
                output << "x_" << d << ",";
                for (int i = 0; i < m_permutation->size(); ++i) {
                    output << x0[this->m_space_dim * (*m_permutation)[i] + d];
                    if (i != m_permutation->size() - 1) {
                        output << ",";
                    }
                }
                output << "\n";
            }

            while (!s.empty()) {
                VirtualCluster const *curr = s.top();
                s.pop();
                std::vector<int>::const_iterator it = std::find(depths.begin(), depths.end(), curr->get_depth());

                if (it != depths.end()) {
                    int index = std::distance(depths.begin(), it);
                    std::fill_n(outputs[index].begin() + curr->get_offset(), curr->get_size(), counters[index]);
                    counters[index] += 1;
                }

                // Recursion
                if (!curr->is_leaf()) {

                    for (int p = 0; p < curr->get_nb_sons(); p++) {
                        s.push(&(curr->get_son(p)));
                    }
                }
            }

            for (int p = 0; p < depths.size(); p++) {
                output << depths[p] << ",";

                for (int i = 0; i < outputs[p].size(); ++i) {
                    output << outputs[p][i];
                    if (i != outputs[p].size() - 1) {
                        output << ',';
                    }
                }
                output << "\n";
            }
        }
    }

    void save_cluster(std::string filename) const {
        int rankWorld;
        MPI_Comm_rank(m_comm, &rankWorld);
        if (rankWorld == 0) {
            // Permutation
            std::ofstream output_permutation(filename + "_permutation.csv");

            for (int i = 0; i < this->m_permutation->size(); i++) {
                output_permutation << (*(this->m_permutation))[i];
                if (i != this->m_permutation->size() - 1) {
                    output_permutation << ",";
                }
            }

            // Tree
            std::stack<VirtualCluster const *> s;
            s.push(this);
            std::ofstream output_tree(filename + "_tree.csv");

            std::vector<std::vector<std::string>> outputs(this->m_max_depth + 1);

            while (!s.empty()) {
                VirtualCluster const *curr = s.top();
                s.pop();

                std::vector<std::string> infos_str = {NbrToStr(curr->get_nb_sons()), NbrToStr(curr->get_rank()), NbrToStr(curr->get_offset()), NbrToStr(curr->get_size()), NbrToStr(curr->get_rad())};
                for (int p = 0; p < this->m_space_dim; p++) {
                    infos_str.push_back(NbrToStr(curr->get_ctr()[p]));
                }

                std::string infos = join("|", infos_str);
                outputs[curr->get_depth()].push_back(infos);

                // Recursion
                if (!curr->is_leaf()) {
                    for (int p = curr->get_nb_sons() - 1; p != -1; p--) {
                        s.push(&(curr->get_son(p)));
                    }
                }
            }

            for (int p = 0; p < outputs.size(); p++) {
                for (int i = 0; i < outputs[p].size(); ++i) {
                    output_tree << outputs[p][i];
                    if (i != outputs[p].size() - 1) {
                        output_tree << ',';
                    }
                }
                output_tree << "\n";
            }
        }
    }

    void read_cluster(std::string file_permutation, std::string file_tree) {
        int rankWorld, sizeWorld;
        MPI_Comm_rank(m_comm, &rankWorld);
        MPI_Comm_size(m_comm, &sizeWorld);

        // Permutation
        std::ifstream input_permutation(file_permutation);
        if (!input_permutation) {
            std::cerr << "Cannot open file containing permutation" << std::endl;
            exit(1);
        }

        std::string line;
        std::getline(input_permutation, line);
        std::string delimiter     = ",";
        std::string sub_delimiter = "|";

        std::vector<std::string> permutation_str = split(line, delimiter.c_str());
        this->m_permutation->resize(permutation_str.size());

        for (int i = 0; i < permutation_str.size(); i++) {
            (*(this->m_permutation))[i] = StrToNbr<double>(permutation_str[i]);
        }

        // Tree
        std::stack<std::pair<Cluster *, int>> s;

        std::ifstream input_tree(file_tree);
        if (!input_tree) {
            std::cerr << "Cannot open file containing tree" << std::endl;
            exit(1);
        }
        std::vector<std::vector<std::string>> outputs;
        int count = 0;
        while (std::getline(input_tree, line)) {
            outputs.push_back(split(line, delimiter));
            count++;
        }

        // Initialisation root
        std::vector<int> counter_offset(outputs.size(), 0);
        std::vector<std::string> local_info_str = split(outputs[0][0], "|");

        int nb_sons     = StrToNbr<int>(local_info_str[0]);
        this->m_counter = 0;
        this->m_rank    = StrToNbr<int>(local_info_str[1]);
        this->m_offset  = StrToNbr<int>(local_info_str[2]);
        this->m_size    = StrToNbr<int>(local_info_str[3]);
        this->m_rad     = StrToNbr<double>(local_info_str[4]);
        for (int p = 0; p < m_space_dim; p++) {
            this->m_ctr[p] = StrToNbr<double>(local_info_str[5 + p]);
        }

        this->m_max_depth = outputs.size() + 1;
        if (sizeWorld == 1) {
            this->m_local_cluster = this->m_root;
            this->m_MasterOffset.push_back(std::pair<int, int>(0, this->m_size));
        } else {
            this->m_MasterOffset.resize(sizeWorld);
        }

        s.push(std::pair<Cluster *, int>(this, nb_sons));
        while (!s.empty()) {
            Cluster *curr = s.top().first;
            nb_sons       = s.top().second;
            std::vector<int> nb_sons_next(nb_sons, 0);
            s.pop();

            // Creating sons
            for (int p = 0; p < nb_sons; p++) {
                curr->add_son(counter_offset[curr->m_depth + 1] + p, curr->m_depth + 1, this->m_permutation);
                local_info_str            = split(outputs[curr->m_depth + 1][counter_offset[curr->m_depth + 1] + p], "|");
                nb_sons_next[p]           = StrToNbr<int>(local_info_str[0]);
                curr->m_sons[p]->m_rank   = StrToNbr<int>(local_info_str[1]);
                curr->m_sons[p]->m_offset = StrToNbr<int>(local_info_str[2]);
                curr->m_sons[p]->m_size   = StrToNbr<int>(local_info_str[3]);
                curr->m_sons[p]->m_rad    = StrToNbr<double>(local_info_str[4]);
                for (int l = 0; l < m_space_dim; l++) {
                    curr->m_sons[p]->m_ctr[l] = StrToNbr<double>(local_info_str[5 + l]);
                }

                if (sizeWorld > 1 && outputs[curr->m_depth + 1].size() == sizeWorld) {
                    this->m_MasterOffset[curr->m_sons[p]->get_counter()] = std::pair<int, int>(curr->m_sons[p]->get_offset(), curr->m_sons[p]->get_size());
                    if (rankWorld == curr->m_sons[p]->get_counter()) {
                        this->set_local_cluster(curr->get_son_ptr(p));
                    }
                }
            }

            // Recursion
            if (!curr->is_leaf()) {
                counter_offset[curr->m_depth + 1] += nb_sons;
                for (int p = curr->get_nb_sons() - 1; p != -1; p--) {
                    s.push(std::pair<Cluster *, int>(curr->get_son_ptr(p), nb_sons_next[p]));
                }
            } else {
                this->m_min_depth = std::min(this->m_min_depth, curr->m_depth);
            }
        }
    }
};

} // namespace htool
#endif
