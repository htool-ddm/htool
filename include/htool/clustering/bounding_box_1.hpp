#ifndef HTOOL_CLUSTERING_BOUNDINGBOX_HPP
#define HTOOL_CLUSTERING_BOUNDINGBOX_HPP

#include "../misc/evp.hpp"
#include "cluster.hpp"
#include "splitting.hpp"
#include <limits>
#include <stack>

namespace htool {

template <SplittingTypes SplittingType>
class BoundingBox1 {
  public:
    void
    recursive_build(const double *const x, const double *const r, const double *const g, int nb_sons, MPI_Comm comm, std::stack<Cluster<BoundingBox1> *> &s, std::stack<std::vector<int>> &n) {

        // MPI parameters
        int rankWorld, sizeWorld;
        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        while (!s.empty()) {
            Cluster<BoundingBox1> *curr = s.top();
            std::vector<int> num        = n.top();
            s.pop();
            n.pop();

            int curr_nb_sons = curr->get_depth() == 0 ? sizeWorld : nb_sons;

            // Mass of the cluster
            int nb_pt = curr->get_size();
            double G  = 0;
            for (int j = 0; j < nb_pt; j++) {
                G += g[num[j]];
            }

            // Center of the cluster
            std::vector<double> xc(curr->get_space_dim(), 0);
            for (int j = 0; j < nb_pt; j++) {
                for (int p = 0; p < curr->get_space_dim(); p++) {
                    xc[p] += g[num[j]] * x[curr->get_space_dim() * num[j] + p];
                }
            }
            std::transform(xc.begin(), xc.end(), xc.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, 1. / G));

            curr->set_ctr(xc);

            // Radius and min max for each axis
            Matrix<double> cov(curr->get_space_dim(), curr->get_space_dim());
            double rad = 0;
            std::vector<double> min_point(curr->get_space_dim(), std::numeric_limits<double>::max());
            std::vector<double> max_point(curr->get_space_dim(), std::numeric_limits<double>::min());
            for (int j = 0; j < nb_pt; j++) {
                std::vector<double> u(3, 0);
                for (int p = 0; p < curr->get_space_dim(); p++) {
                    if (min_point[p] > x[curr->get_space_dim() * num[j] + p]) {
                        min_point[p] = x[curr->get_space_dim() * num[j] + p];
                    }
                    if (max_point[p] < x[curr->get_space_dim() * num[j] + p]) {
                        max_point[p] = x[curr->get_space_dim() * num[j] + p];
                    }
                    u[p] = x[curr->get_space_dim() * num[j] + p] - xc[p];
                }

                rad = std::max(rad, norm2(u) + r[j]);
            }
            curr->set_rad(rad);

            // Direction of largest extent
            double max_distance(std::numeric_limits<double>::min());
            int dir_axis = 0;
            for (int p = 0; p < curr->get_space_dim(); p++) {
                if (max_distance < max_point[p] - min_point[p]) {
                    max_distance = max_point[p] - min_point[p];
                    dir_axis     = p;
                }
            }
            std::vector<double> dir(curr->get_space_dim(), 0);
            dir[dir_axis] = 1;

            // Creating sons
            for (int p = 0; p < curr_nb_sons; p++) {
                curr->add_son(curr->get_counter() * curr_nb_sons + p, curr->get_depth() + 1, curr->get_perm_ptr());
            }

            // Compute numbering
            std::vector<std::vector<int>> numbering = splitting(x, num, curr, curr_nb_sons, dir);

            // Set offsets, size and rank of sons
            int count = 0;

            for (int p = 0; p < curr_nb_sons; p++) {
                curr->get_son_ptr(p)->set_offset(curr->get_offset() + count);
                curr->get_son_ptr(p)->set_size(numbering[p].size());
                count += numbering[p].size();

                // level of parallelization
                if (curr->get_depth() == 0) {
                    curr->get_son_ptr(p)->set_rank(curr->get_son_ptr(p)->get_counter());
                    if (rankWorld == curr->get_son_ptr(p)->get_counter()) {
                        curr->set_local_cluster(curr->get_son_ptr(p));
                    }
                    curr->set_MasterOffset(curr->get_son_ptr(p)->get_counter(), std::pair<int, int>(curr->get_son_ptr(p)->get_offset(), curr->get_son_ptr(p)->get_size()));
                }
                // after level of parallelization
                else {
                    curr->get_son_ptr(p)->set_rank(curr->get_rank());
                }
            }

            // Recursivite
            bool test_minclustersize = true;
            for (int p = 0; p < curr_nb_sons; p++) {
                test_minclustersize = test_minclustersize && (numbering[p].size() >= curr->get_minclustersize());
            }
            if (test_minclustersize || curr->get_rank() == -1) {
                for (int p = 0; p < curr_nb_sons; p++) {
                    s.push((curr->get_son_ptr(p)));
                    n.push(numbering[p]);
                }
            } else {
                curr->set_max_depth(std::max(curr->get_max_depth(), curr->get_depth()));
                if (curr->get_min_depth() < 0) {
                    curr->set_min_depth(curr->get_depth());
                } else {
                    curr->set_min_depth(std::min(curr->get_min_depth(), curr->get_depth()));
                }

                curr->clear_sons();
                std::copy_n(num.begin(), num.size(), curr->get_perm_start() + curr->get_offset());
            }
        }
    }

    std::vector<std::vector<int>> splitting(const double *const x, std::vector<int> &num, VirtualCluster const *const curr_cluster, int nb_sons, const std::vector<double> &dir);
};

// Specialization of splitting
template <>
inline std::vector<std::vector<int>> BoundingBox1<SplittingTypes::GeometricSplitting>::splitting(const double *const x, std::vector<int> &num, VirtualCluster const *const curr_cluster, int nb_sons, const std::vector<double> &dir) { return geometric_splitting(x, num, curr_cluster, nb_sons, dir); }

template <>
inline std::vector<std::vector<int>> BoundingBox1<SplittingTypes::RegularSplitting>::splitting(const double *const x, std::vector<int> &num, VirtualCluster const *const curr_cluster, int nb_sons, const std::vector<double> &dir) { return regular_splitting(x, num, curr_cluster, nb_sons, dir); }

// Typdef with specific splitting
typedef BoundingBox1<SplittingTypes::GeometricSplitting> BoundingBox1GeometricClustering;
typedef BoundingBox1<SplittingTypes::RegularSplitting> BoundingBox1RegularClustering;

} // namespace htool

#endif
