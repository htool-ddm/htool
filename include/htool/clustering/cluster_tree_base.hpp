#ifndef HTOOL_CLUSTER_TREE_BASE_HPP
#define HTOOL_CLUSTER_TREE_BASE_HPP

#include "cluster.hpp"
#include "cluster_tree.hpp"
#include <mpi.h>
#include <map>

namespace htool {
class Cluster_tree;

class Cluster_tree_base {
    friend class Cluster_tree;
private:
  // Data
  std::vector<int> perm;
  Cluster root;


public:
    // Full Constructor
    Cluster_tree_base(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,r0,tab0,g0,perm){}

    // Constructor without radius
    Cluster_tree_base(const std::vector<R3>& x0,const std::vector<int>& tab0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,tab0,g0,perm){}

    // Constructor without mass
    Cluster_tree_base(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,r0,tab0,perm){}

    // Constructor without tab
    Cluster_tree_base(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):perm(x0.size()),root(x0,r0,g0,perm){}

    // Constructor without radius and mass
    Cluster_tree_base(const std::vector<R3>& x0, const std::vector<int>& tab0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,tab0,perm){}

    // Constructor without radius, mass and tab
    Cluster_tree_base(const std::vector<R3>& x0, MPI_Comm comm0=MPI_COMM_WORLD):perm(x0.size()),root(x0,perm){}




};


} // namespace

#endif
