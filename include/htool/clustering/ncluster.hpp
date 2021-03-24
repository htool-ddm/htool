#ifndef HTOOL_CLUSTERING_NCluster_HPP
#define HTOOL_CLUSTERING_NCluster_HPP

#include "../misc/evp.hpp"
#include "cluster.hpp"
#include "splitting.hpp"
#include <stack>

namespace htool {

template <SplittingTypes SplittingType>
class NCluster : public Cluster<NCluster<SplittingType>> {
  private:
    void recursive_build(const double *const x, const double *const r, const int *const tab, const double *const g, int nb_sons, MPI_Comm comm, std::stack<NCluster *> &s, std::stack<std::vector<int>> &n) {

        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        while (!s.empty()) {
            NCluster *curr       = s.top();
            std::vector<int> num = n.top();
            s.pop();
            n.pop();

            int curr_nb_sons = curr->depth == 0 ? sizeWorld : nb_sons;

            // Mass of the cluster
            int nb_pt = curr->size;
            double G  = 0;
            for (int j = 0; j < nb_pt; j++) {
                G += g[tab[num[j]]];
            }

            // Center of the cluster
            std::vector<double> xc(this->space_dim, 0);
            for (int j = 0; j < nb_pt; j++) {
                for (int p = 0; p < this->space_dim; p++) {
                    xc[p] += g[tab[num[j]]] * x[this->space_dim * tab[num[j]] + p];
                }
            }
            std::transform(xc.begin(), xc.end(), xc.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, 1. / G));

            curr->ctr = xc;

            // Radius and covariance matrix
            Matrix<double> cov(this->space_dim, this->space_dim);
            double rad = 0;
            for (int j = 0; j < nb_pt; j++) {
                std::vector<double> u(3, 0);
                for (int p = 0; p < this->space_dim; p++) {
                    u[p] = x[this->space_dim * tab[num[j]] + p] - xc[p];
                }

                rad = std::max(rad, norm2(u) + r[tab[num[j]]]);
                for (int p = 0; p < this->space_dim; p++) {
                    for (int q = 0; q < this->space_dim; q++) {
                        cov(p, q) += g[tab[num[j]]] * u[p] * u[q];
                    }
                }
            }
            curr->rad = rad;

            // Direction of largest extent
            std::vector<double> dir;
            if (this->space_dim == 2) {
                dir = solve_EVP_2(cov);
            } else if (this->space_dim == 3) {
                dir = solve_EVP_3(cov);
            } else {
                throw std::logic_error("[Htool error] clustering not define for spatial dimension !=2 and !=3");
            }

            // Creating sons
            curr->sons.resize(curr_nb_sons);
            for (int p = 0; p < curr_nb_sons; p++) {
                curr->sons[p] = new NCluster(this, (curr->counter) * curr_nb_sons + p, curr->depth + 1, this->permutation);
            }

            // Compute numbering
            std::vector<std::vector<int>> numbering = this->splitting(this->nb_pt, x, tab, num, curr, curr_nb_sons, dir);

            // Set offsets, size and rank of sons
            int count = 0;

            for (int p = 0; p < curr_nb_sons; p++) {
                curr->sons[p]->set_offset(curr->offset + count);
                curr->sons[p]->set_size(numbering[p].size());
                count += numbering[p].size();

                // level of parallelization
                if (curr->depth == 0) {
                    curr->sons[p]->set_rank(curr->sons[p]->get_counter());
                    if (rankWorld == curr->sons[p]->get_counter()) {
                        this->local_cluster = (curr->sons[p]);
                    }
                    this->MasterOffset[curr->sons[p]->get_counter()] = std::pair<int, int>(curr->sons[p]->get_offset(), curr->sons[p]->get_size());
                }
                // after level of parallelization
                else {
                    curr->sons[p]->set_rank(curr->rank);
                }
            }

            // Recursivite
            bool test_minclustersize = true;
            for (int p = 0; p < curr_nb_sons; p++) {
                test_minclustersize = test_minclustersize && (numbering[p].size() >= this->minclustersize);
            }
            if (test_minclustersize || curr->rank == -1) {
                for (int p = 0; p < curr_nb_sons; p++) {
                    s.push((curr->sons[p]));
                    n.push(numbering[p]);
                }
            } else {
                this->max_depth = std::max(this->max_depth, curr->depth);
                if (this->min_depth < 0) {
                    this->min_depth = curr->depth;
                } else {
                    this->min_depth = std::min(this->min_depth, curr->depth);
                }

                for (auto &son : curr->sons) {
                    delete son;
                    son = nullptr;
                }
                curr->sons.resize(0);
                std::copy_n(num.begin(), num.size(), this->permutation->begin() + curr->offset);
            }
        }
    }

  public:
    // Inherhits son constructor
    using Cluster<NCluster<SplittingType>>::Cluster;

    // build cluster tree
    // nb_sons=-1 means nb_sons = 2
    void build_global(int nb_pt0, const double *const x, const double *const r, const int *const tab, const double *const g, int nb_sons = -1, MPI_Comm comm = MPI_COMM_WORLD) {
        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons));

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
        this->sons.resize(sizeWorld);
        this->LocalPermutation = false;
        for (auto &son : this->sons) {
            son = nullptr;
        }
        this->depth = 0; // ce constructeur est appele' juste pour la racine

        this->permutation->resize(nb_pt0);
        std::iota(this->permutation->begin(), this->permutation->end(), 0); // perm[i]=i

        // Recursion
        std::stack<NCluster *> s;
        std::stack<std::vector<int>> n;
        s.push(this);
        n.push(*(this->permutation));

        this->recursive_build(x, r, tab, g, nb_sons, comm, s, n);
    }

    // build cluster tree from given partition
    void build_local(int nb_pt0, const double *const x, const double *const r, const int *const tab, const double *const g, const int *const MasterOffset0, int nb_sons = -1, MPI_Comm comm = MPI_COMM_WORLD) {

        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        // Impossible value for nb_sons
        if (nb_sons == 0 || nb_sons == 1)
            throw std::string("Impossible value for nb_sons:" + NbrToStr<int>(nb_sons));

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
        std::stack<NCluster *> s;
        std::stack<std::vector<int>> n;

        this->sons.resize(sizeWorld);
        for (int p = 0; p < sizeWorld; p++) {
            this->sons[p] = new NCluster(this, p, this->depth + 1, this->permutation);
            this->sons[p]->set_offset(this->MasterOffset[p].first);
            this->sons[p]->set_size(this->MasterOffset[p].second);
            this->sons[p]->set_rank(p);

            if (rankWorld == this->sons[p]->get_counter()) {
                this->local_cluster = this->sons[p];
            }

            s.push(this->sons[p]);
            n.push(std::vector<int>(this->permutation->begin() + this->sons[p]->get_offset(), this->permutation->begin() + this->sons[p]->get_offset() + this->sons[p]->get_size()));
        }

        // Recursion
        this->recursive_build(x, r, tab, g, nb_sons, comm, s, n);
    }
    std::vector<std::vector<int>> splitting(int nb_pt0, const double *const x, const int *const tab, std::vector<int> &num, Cluster<NCluster<SplittingType>> const *const curr_cluster, int nb_sons, const std::vector<double> &dir);
};

// Specialization of splitting
template <>
std::vector<std::vector<int>> NCluster<SplittingTypes::GeometricSplitting>::splitting(int nb_pt0, const double *const x, const int *const tab, std::vector<int> &num, Cluster<NCluster<SplittingTypes::GeometricSplitting>> const *const curr_cluster, int nb_sons, const std::vector<double> &dir) { return geometric_splitting(nb_pt0, x, tab, num, curr_cluster, nb_sons, dir); }

template <>
std::vector<std::vector<int>> NCluster<SplittingTypes::RegularSplitting>::splitting(int nb_pt0, const double *const x, const int *const tab, std::vector<int> &num, Cluster<NCluster<SplittingTypes::RegularSplitting>> const *const curr_cluster, int nb_sons, const std::vector<double> &dir) { return regular_splitting(nb_pt0, x, tab, num, curr_cluster, nb_sons, dir); }

// Typdef with specific splitting
typedef NCluster<SplittingTypes::GeometricSplitting> GeometricClustering;
typedef NCluster<SplittingTypes::RegularSplitting> RegularClustering;

} // namespace htool
#endif
