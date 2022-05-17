#ifndef HTOOL_CLUSTERING_CLUSTER_HPP
#define HTOOL_CLUSTERING_CLUSTER_HPP

#include "../misc/user.hpp"
#include "../types/matrix.hpp"
#include "virtual_cluster.hpp"
#include <functional>
#include <memory>
#include <mpi.h>
#include <stack>

namespace htool {
template <typename ClusteringType>
class Cluster : public VirtualCluster {
  protected:
    // Data member
    std::vector<std::unique_ptr<Cluster>> sons; // Sons

    int rank;      // Rank for dofs of the current cluster
    int depth;     // depth of the current cluster
    int counter;   // numbering of the nodes level-wise
    int space_dim; // dimension for geometric points
    int nb_pt;

    double rad;
    std::vector<double> ctr;

    int max_depth;
    int min_depth;
    int offset;
    int size;

    unsigned int minclustersize;
    unsigned int ndofperelt;

    bool LocalPermutation;

    std::shared_ptr<std::vector<int>> permutation;

    Cluster *local_cluster;
    Cluster *const root;
    std::vector<std::pair<int, int>> MasterOffset;

    //
    ClusteringType clustering_type;

  public:
    // Root constructor
    Cluster(int space_dim0 = 3) : depth(0), counter(0), space_dim(space_dim0), nb_pt(0), rad(0), ctr(space_dim0), max_depth(0), min_depth(-1), offset(0), size(0), minclustersize(10), ndofperelt(1), LocalPermutation(false), permutation(std::make_shared<std::vector<int>>()), local_cluster(nullptr), root(this) {}

    // Node constructor
    Cluster(Cluster *const root0, int counter0, const int &dep, std::shared_ptr<std::vector<int>> permutation0) : counter(counter0), space_dim(root0->space_dim), nb_pt(0), rad(0.), ctr(root0->space_dim), max_depth(-1), min_depth(-1), offset(0), size(0), minclustersize(root0->minclustersize), ndofperelt(root0->ndofperelt), LocalPermutation(false), permutation(permutation0), root(root0) {
        depth = dep;
    }

    // global build cluster tree
    void build(int nb_pt0, const double *const x0, const double *const r0, const double *const g0, int nb_sons = -1, MPI_Comm comm = MPI_COMM_WORLD) {
        this->nb_pt = nb_pt0;

        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons)); // LCOV_EXCL_LINE

        // nb_sons=-1 is automatic mode
        if (nb_sons == -1) {
            nb_sons = 2;
        }
        // Initialisation
        this->rad   = 0;
        this->size  = nb_pt0;
        this->nb_pt = nb_pt0;
        this->rank  = -1;
        this->MasterOffset.resize(sizeWorld);
        this->LocalPermutation = false;
        this->depth            = 0; // ce constructeur est appele' juste pour la racine
        this->permutation->resize(nb_pt0);
        std::iota(this->permutation->begin(), this->permutation->end(), 0); // perm[i]=i

        // Recursion
        std::stack<Cluster *> s;
        std::stack<std::vector<int>> n;
        s.push(this);
        n.push(*(this->permutation));

        clustering_type.recursive_build(x0, r0, g0, nb_sons, comm, s, n);
    }

    void build(int nb_pt0, const double *const x0, int nb_sons = -1, MPI_Comm comm = MPI_COMM_WORLD) {
        this->nb_pt = nb_pt0;
        this->build(nb_pt0, x0, std::vector<double>(nb_pt0, 0).data(), std::vector<double>(nb_pt0, 1).data(), nb_sons, comm);
    }

    // local build cluster tree
    void build(int nb_pt0, const double *const x0, const double *const r0, const double *const g0, const int *const MasterOffset0, int nb_sons = -1, MPI_Comm comm = MPI_COMM_WORLD) {
        this->nb_pt = nb_pt0;
        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons)); // LCOV_EXCL_LINE

        // nb_sons=-1 is automatic mode
        if (nb_sons == -1) {
            nb_sons = 2;
        }

        // Initialisation of root
        this->rad   = 0;
        this->nb_pt = nb_pt0;
        this->size  = this->nb_pt;
        this->rank  = -1;
        this->MasterOffset.resize(sizeWorld);
        for (int p = 0; p < sizeWorld; p++) {
            this->MasterOffset[p].first  = MasterOffset0[2 * p];
            this->MasterOffset[p].second = MasterOffset0[2 * p + 1];
        }
        this->permutation->resize(this->nb_pt);

        this->depth            = 0; // ce constructeur est appele' juste pour la racine
        this->LocalPermutation = true;

        std::iota(this->permutation->begin(), this->permutation->end(), 0); // perm[i]=i
        // Build level of depth 1 with the given partition and prepare recursion
        std::stack<Cluster *> s;
        std::stack<std::vector<int>> n;

        for (int p = 0; p < sizeWorld; p++) {
            this->sons.emplace_back(new Cluster(this, p, this->depth + 1, this->permutation));
            this->sons[p]->set_offset(this->MasterOffset[p].first);
            this->sons[p]->set_size(this->MasterOffset[p].second);
            this->sons[p]->set_rank(p);

            if (rankWorld == this->sons[p]->get_counter()) {
                this->local_cluster = this->sons[p].get();
            }

            s.push(this->sons[p].get());
            n.push(std::vector<int>(this->permutation->begin() + this->sons[p]->get_offset(), this->permutation->begin() + this->sons[p]->get_offset() + this->sons[p]->get_size()));
        }

        // Recursion
        clustering_type.recursive_build(x0, r0, g0, nb_sons, comm, s, n);
    }

    void build(int nb_pt0, const double *const x0, const int *const MasterOffset0, int nb_sons = -1, MPI_Comm comm = MPI_COMM_WORLD) {
        this->nb_pt = nb_pt0;
        this->build(nb_pt0, x0, std::vector<double>(nb_pt0, 0).data(), std::vector<double>(nb_pt0, 1).data(), MasterOffset0, nb_sons, comm);
    }

    //// Getters for local data
    double get_rad() const { return rad; }
    const std::vector<double> &get_ctr() const { return ctr; }
    const VirtualCluster &get_son(const int &j) const { return *(sons[j]); }
    VirtualCluster &get_son(const int &j) { return *(sons[j]); }
    Cluster *get_son_ptr(const int &j) { return sons[j].get(); }
    int get_depth() const { return depth; }
    int get_rank() const { return rank; }
    int get_offset() const { return offset; }
    int get_size() const { return size; }
    const int *get_perm_data() const { return permutation->data() + offset; };
    int *get_perm_data() { return permutation->data() + offset; };
    int get_space_dim() const { return space_dim; }
    int get_minclustersize() const { return minclustersize; }
    int get_ndofperelt() const { return ndofperelt; }
    int get_nb_sons() const { return sons.size(); }
    int get_counter() const { return counter; }
    const VirtualCluster &get_local_cluster() const {
        return *(root->local_cluster);
    }

    std::shared_ptr<VirtualCluster> get_local_cluster_tree(MPI_Comm comm = MPI_COMM_WORLD) {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);

        std::shared_ptr<Cluster> copy_local_cluster = std::make_shared<Cluster>();

        copy_local_cluster->MasterOffset.push_back(std::make_pair(this->MasterOffset[rankWorld].first, this->MasterOffset[rankWorld].second));

        copy_local_cluster->local_cluster = copy_local_cluster->root;

        copy_local_cluster->permutation = this->permutation;

        // Recursion
        std::stack<Cluster *> cluster_input;
        cluster_input.push(local_cluster);
        std::stack<Cluster *> cluster_output;
        cluster_output.push(copy_local_cluster->root);
        while (!cluster_input.empty()) {
            Cluster *curr_input  = cluster_input.top();
            Cluster *curr_output = cluster_output.top();

            cluster_input.pop();
            cluster_output.pop();

            curr_output->rank   = curr_input->rank;
            curr_output->ctr    = curr_input->ctr;
            curr_output->rad    = curr_input->rad;
            curr_output->offset = curr_input->offset;
            curr_output->size   = curr_input->size;

            int nb_sons = curr_input->sons.size();

            for (int p = 0; p < nb_sons; p++) {
                curr_output->sons.emplace_back(new Cluster(copy_local_cluster.get(), (curr_output->counter) * nb_sons + p, curr_output->depth + 1, this->permutation));

                cluster_input.push(curr_input->get_son_ptr(p));
                cluster_output.push(curr_output->get_son_ptr(p));
            }
        }

        return copy_local_cluster;
    }

    std::vector<int> get_local_perm() const {
        if (!LocalPermutation) {
            throw std::logic_error("[Htool error] Permutation is not local, get_local_perm cannot be used"); // LCOV_EXCL_LINE
        } else {
            return std::vector<int>(permutation->data() + root->local_cluster->get_offset(), permutation->data() + root->local_cluster->get_offset() + root->local_cluster->get_size());
        }
    }

    //// Getters for global data
    int get_max_depth() const { return root->max_depth; }
    int get_min_depth() const { return root->min_depth; }
    const std::vector<int> &get_global_perm() const { return *permutation; };
    std::shared_ptr<std::vector<int>> get_perm_ptr() const { return permutation; };
    const int *get_global_perm_data() const { return permutation->data(); };
    int *get_global_perm_data() { return permutation->data(); };
    int get_global_perm(int i) const { return (*permutation)[i]; };
    const VirtualCluster *get_root() const {
        return root;
    }

    //// Getter for MasterOffsets
    int get_local_offset() const { return root->local_cluster->get_offset(); }
    int get_local_size() const { return root->local_cluster->get_size(); }
    const std::vector<std::pair<int, int>> &get_masteroffset() const { return root->MasterOffset; }
    std::pair<int, int> get_masteroffset(int i) const { return root->MasterOffset[i]; }

    bool IsLocal() const { return LocalPermutation; }
    //// Setters
    void set_rank(int rank0) { rank = rank0; }
    void set_offset(int offset0) { offset = offset0; }
    void set_size(int size0) { size = size0; }
    void set_rad(double rad0) { rad = rad0; }
    void set_ctr(std::vector<double> ctr0) { ctr = ctr0; }
    void set_local_cluster(Cluster *local_cluster0) { root->local_cluster = local_cluster0; }
    void set_MasterOffset(int p, std::pair<int, int> pair0) { this->root->MasterOffset[p] = pair0; }

    void add_son(int counter0, const int &dep, std::shared_ptr<std::vector<int>> permutation0) {
        sons.emplace_back(new Cluster(root, counter0, dep, permutation0));
    }

    void set_max_depth(int max_depth0) const { root->max_depth = max_depth0; }
    void set_min_depth(int min_depth0) const { root->min_depth = min_depth0; }

    void set_minclustersize(unsigned int minclustersize0) {
        if (minclustersize0 == 0) {
            throw std::invalid_argument("[Htool error] MinClusterSize parameter cannot be zero"); // LCOV_EXCL_LINE
        }
        minclustersize = minclustersize0;
    }
    void set_ndofperelt(unsigned int ndofperelt0) { ndofperelt = ndofperelt0; }

    bool IsLeaf() const {
        if (sons.size() == 0) {
            return true;
        }
        return false;
    }

    void clear_sons() {
        sons.clear();
        sons.resize(0);
    }

    // Output
    void print(MPI_Comm comm = MPI_COMM_WORLD) const {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);
        if (rankWorld == 0) {
            if (!permutation->empty()) {
                std::cout << '[';
                for (std::vector<int>::const_iterator i = permutation->begin() + offset; i != permutation->begin() + offset + size; ++i)
                    std::cout << *i << ',';
                std::cout << "\b]" << std::endl;
                ;
            }

            for (auto &son : this->sons) {
                if (son != NULL)
                    (*son).print();
            }
        }
    }
    void save_geometry(const double *const x0, std::string filename, const std::vector<int> &depths, MPI_Comm comm = MPI_COMM_WORLD) const {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);
        if (rankWorld == 0) {

            std::stack<VirtualCluster const *> s;
            s.push(this);
            std::ofstream output(filename + ".csv");

            std::vector<std::vector<int>> outputs(depths.size());
            std::vector<int> counters(depths.size(), 0);
            for (int p = 0; p < outputs.size(); p++) {
                outputs[p].resize(permutation->size());
            }

            // Permuted geometric points
            for (int d = 0; d < this->space_dim; d++) {
                output << "x_" << d << ",";
                for (int i = 0; i < permutation->size(); ++i) {
                    output << x0[this->space_dim * (*permutation)[i] + d];
                    if (i != permutation->size() - 1) {
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
                if (!curr->IsLeaf()) {

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

    void save_cluster(std::string filename, MPI_Comm comm = MPI_COMM_WORLD) const {
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);
        if (rankWorld == 0) {
            // Permutation
            std::ofstream output_permutation(filename + "_permutation.csv");

            for (int i = 0; i < this->permutation->size(); i++) {
                output_permutation << (*(this->permutation))[i];
                if (i != this->permutation->size() - 1) {
                    output_permutation << ",";
                }
            }

            // Tree
            std::stack<VirtualCluster const *> s;
            s.push(this);
            std::ofstream output_tree(filename + "_tree.csv");

            std::vector<std::vector<std::string>> outputs(this->max_depth + 1);

            while (!s.empty()) {
                VirtualCluster const *curr = s.top();
                s.pop();

                std::vector<std::string> infos_str = {NbrToStr(curr->get_nb_sons()), NbrToStr(curr->get_rank()), NbrToStr(curr->get_offset()), NbrToStr(curr->get_size()), NbrToStr(curr->get_rad())};
                for (int p = 0; p < this->space_dim; p++) {
                    infos_str.push_back(NbrToStr(curr->get_ctr()[p]));
                }

                std::string infos = join("|", infos_str);
                outputs[curr->get_depth()].push_back(infos);

                // Recursion
                if (!curr->IsLeaf()) {
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

    void read_cluster(std::string file_permutation, std::string file_tree, MPI_Comm comm = MPI_COMM_WORLD) {
        int rankWorld, sizeWorld;
        MPI_Comm_rank(comm, &rankWorld);
        MPI_Comm_size(comm, &sizeWorld);

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
        this->permutation->resize(permutation_str.size());

        for (int i = 0; i < permutation_str.size(); i++) {
            (*(this->permutation))[i] = StrToNbr<double>(permutation_str[i]);
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

        int nb_sons   = StrToNbr<int>(local_info_str[0]);
        this->counter = 0;
        this->rank    = StrToNbr<int>(local_info_str[1]);
        this->offset  = StrToNbr<int>(local_info_str[2]);
        this->size    = StrToNbr<int>(local_info_str[3]);
        this->rad     = StrToNbr<double>(local_info_str[4]);
        for (int p = 0; p < space_dim; p++) {
            this->ctr[p] = StrToNbr<double>(local_info_str[5 + p]);
        }

        this->max_depth = outputs.size() + 1;
        if (sizeWorld == 1) {
            this->local_cluster = this->root;
            this->MasterOffset.push_back(std::pair<int, int>(0, this->size));
        } else {
            this->MasterOffset.resize(sizeWorld);
        }

        s.push(std::pair<Cluster *, int>(this, nb_sons));
        while (!s.empty()) {
            Cluster *curr = s.top().first;
            nb_sons       = s.top().second;
            std::vector<int> nb_sons_next(nb_sons, 0);
            s.pop();

            // Creating sons
            for (int p = 0; p < nb_sons; p++) {
                curr->add_son(counter_offset[curr->depth + 1] + p, curr->depth + 1, this->permutation);
                local_info_str        = split(outputs[curr->depth + 1][counter_offset[curr->depth + 1] + p], "|");
                nb_sons_next[p]       = StrToNbr<int>(local_info_str[0]);
                curr->sons[p]->rank   = StrToNbr<int>(local_info_str[1]);
                curr->sons[p]->offset = StrToNbr<int>(local_info_str[2]);
                curr->sons[p]->size   = StrToNbr<int>(local_info_str[3]);
                curr->sons[p]->rad    = StrToNbr<double>(local_info_str[4]);
                for (int l = 0; l < space_dim; l++) {
                    curr->sons[p]->ctr[l] = StrToNbr<double>(local_info_str[5 + l]);
                }

                if (sizeWorld > 1 && outputs[curr->depth + 1].size() == sizeWorld) {
                    this->MasterOffset[curr->sons[p]->get_counter()] = std::pair<int, int>(curr->sons[p]->get_offset(), curr->sons[p]->get_size());
                    if (rankWorld == curr->sons[p]->get_counter()) {
                        this->set_local_cluster(curr->get_son_ptr(p));
                    }
                }
            }

            // Recursion
            if (!curr->IsLeaf()) {
                counter_offset[curr->depth + 1] += nb_sons;
                for (int p = curr->get_nb_sons() - 1; p != -1; p--) {
                    s.push(std::pair<Cluster *, int>(curr->get_son_ptr(p), nb_sons_next[p]));
                }
            } else {
                this->min_depth = std::min(this->min_depth, curr->depth);
            }
        }
    }
};

} // namespace htool
#endif
