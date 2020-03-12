#ifndef HTOOL_MULTIHMATRIX_HPP
#define HTOOL_MULTIHMATRIX_HPP

#include <cassert>
#include <fstream>
#include <mpi.h>
#include <map>
#include <memory>
#include "matrix.hpp"
#include "../misc/parametres.hpp"
#include "../clustering/cluster_tree.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "hmatrix.hpp"
#include "multimatrix.hpp"


namespace htool {

// Class
template< template<typename> class MultiLowRankMatrix, typename T >
class MultiHMatrix: public Parametres{

private:
	// Data members
	int nr;
	int nc;
	int reqrank;
	int local_size;
	int local_offset;
    int nb_hmatrix;

	std::vector<Block*>		   Tasks;
	std::vector<Block*>		   MyBlocks;

    std::shared_ptr<Cluster_tree> cluster_tree_s;
	std::shared_ptr<Cluster_tree> cluster_tree_t;

public:

	// Full constructor
	MultiHMatrix(MultiIMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius, tab and mass
	MultiHMatrix(MultiIMatrix<T>&, const std::vector<R3>& xt, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

    // Destructor
	~MultiHMatrix() {
		for (int i=0; i<Tasks.size(); i++)
			delete Tasks[i];
	}

};

// // build
// template< template<typename> class MultiLowRankMatrix, typename T >
// void MultiHMatrix<MultiLowRankMatrix, T >::build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

// 	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

// 	MPI_Comm_dup(comm0,&comm);
// 	MPI_Comm_size(comm, &sizeWorld);
// 	MPI_Comm_rank(comm, &rankWorld);
// 	std::vector<double> mytimes(4), maxtime(4), meantime(4);

// 	// Construction arbre des paquets
// 	double time = MPI_Wtime();
// 	cluster_tree_t = std::make_shared<Cluster_tree>(xt,rt,tabt,gt,comm); // target
// 	cluster_tree_s = std::make_shared<Cluster_tree>(xs,rs,tabs,gs,comm); // source

// 	local_size   = cluster_tree_t->get_local_size();
// 	local_offset = cluster_tree_t->get_local_offset();

// 	mytimes[0] = MPI_Wtime() - time;

// 	// Construction arbre des blocs
// 	time = MPI_Wtime();
// 	Block* B = BuildBlockTree(cluster_tree_t->get_head(),cluster_tree_s->get_head());
// 	if (B != NULL) Tasks.push_back(B);
// 	mytimes[1] = MPI_Wtime() - time;

// 	// Repartition des blocs sur les processeurs
// 	time = MPI_Wtime();
// 	ScatterTasks();
// 	mytimes[2] = MPI_Wtime() - time;

// 	// Assemblage des sous-matrices
// 	time = MPI_Wtime();
// 	ComputeBlocks(mat,xt,tabt,xs,tabs);
// 	mytimes[3] = MPI_Wtime() - time;

// 	// Infos
// 	ComputeInfos(mytimes);
// }

// // Full constructor
// template< template<typename> class MultiLowRankMatrix, typename T >
// HMatrix<MultiLowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
// 	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
// }

} //namespace
#endif