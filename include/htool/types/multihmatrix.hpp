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
#include "../multilrmat/multilrmat.hpp"


namespace htool {

// Friend functions --- forward declaration
template< template<typename> class LowRankMatrix, typename T >
class HMatrix;

template<template<typename> class MultiLowRankMatrix,typename T >
class MultiHMatrix;

template< template<typename> class MultiLowRankMatrix, typename T >
double Frobenius_absolute_error(const MultiHMatrix<MultiLowRankMatrix,T>& B, const MultiIMatrix<T>& A, int l);

// Class
template<template<typename> class MultiLowRankMatrix,typename T >
class MultiHMatrix: public Parametres{

private:
	// Data members
	int nr;
	int nc;
	int reqrank;
	int local_size;
	int local_offset;
    int nb_hmatrix;

	MPI_Comm comm;
	int rankWorld,sizeWorld;

	std::vector<HMatrix<bareLowRankMatrix,T> >    HMatrices;
	// std::vector<Block*>		   Tasks;
	// std::vector<Block*>		   MyBlocks;

    std::shared_ptr<Cluster_tree> cluster_tree_s;
	std::shared_ptr<Cluster_tree> cluster_tree_t;


public:

	// Build
	void build(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor
	MultiHMatrix(MultiIMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius, tab and mass
	MultiHMatrix(MultiIMatrix<T>&, const std::vector<R3>& xt, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	
	// Internal methods
	void ComputeBlocks(MultiIMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	void AddNearFieldMat(MultiIMatrix<T>& mat, const Cluster& t, const Cluster& s, std::vector<SubMatrix<T>*>&);
	bool AddFarFieldMat(MultiIMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<bareLowRankMatrix<T>*>&, const int& reqrank=-1);

	HMatrix<bareLowRankMatrix,T>&  operator[](int j){return HMatrices[j];}; 
    const HMatrix<bareLowRankMatrix,T>&  operator[](int j) const {return HMatrices[j];}; 


	friend double Frobenius_absolute_error<MultiLowRankMatrix,T>(const MultiHMatrix<MultiLowRankMatrix,T>& B, const MultiIMatrix<T>& A, int l);

    // // Destructor
	// ~MultiHMatrix() {
	// 	for (int i=0; i<Tasks.size(); i++)
	// 		delete Tasks[i];
	// }

};

// build
template< template<typename> class MultiLowRankMatrix, typename T >
void MultiHMatrix<MultiLowRankMatrix, T >::build(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	MPI_Comm_dup(comm0,&comm);
	MPI_Comm_size(comm, &sizeWorld);
	MPI_Comm_rank(comm, &rankWorld);

	std::vector<double> mytimes(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	cluster_tree_t = std::make_shared<Cluster_tree>(xt,rt,tabt,gt,comm); // target
	cluster_tree_s = std::make_shared<Cluster_tree>(xs,rs,tabs,gs,comm); // source

	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();

	// Hmatrices
	for (int l=0;l<nb_hmatrix;l++){
		HMatrices.push_back(HMatrix<bareLowRankMatrix,T> (nr,nc,cluster_tree_t,cluster_tree_s));
	}

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = HMatrices[0].BuildBlockTree(cluster_tree_t->get_head(),cluster_tree_s->get_head());
	if (B != NULL) HMatrices[0].Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	HMatrices[0].ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;


	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xs,tabs);
	mytimes[3] = MPI_Wtime() - time;


	// Distribute necessary data
	for (int l=0;l<nb_hmatrix;l++){
		HMatrices[l].sizeWorld=sizeWorld;
		HMatrices[l].rankWorld=rankWorld;
		HMatrices[l].comm=comm;
		HMatrices[l].reqrank=this->reqrank;
		HMatrices[l].local_size=local_size;
		HMatrices[l].local_offset=local_offset;
	}


	// Infos
	for (int l=0;l<nb_hmatrix;l++){
		HMatrices[l].ComputeInfos(mytimes);
	}
}

// Full constructor
template< template<typename> class MultiLowRankMatrix, typename T >
MultiHMatrix<MultiLowRankMatrix, T >::MultiHMatrix(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), nb_hmatrix(mat.nb_matrix()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
}

// Constructor without radius, mass and tab
template< template<typename> class MultiLowRankMatrix, typename T >
MultiHMatrix<MultiLowRankMatrix, T >::MultiHMatrix(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<R3>& xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), nb_hmatrix(mat.nb_matrix()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, std::vector<double>(xt.size(),1), xs, std::vector<double>(xs.size(),0), tabs, std::vector<double>(xs.size(),1), comm0);
}


template< template<typename> class MultiLowRankMatrix, typename T >
void MultiHMatrix<MultiLowRankMatrix, T >::ComputeBlocks(MultiIMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
    #if _OPENMP
    #pragma omp parallel
    #endif
	{
        std::vector<SubMatrix<T>*>     MyNearFieldMats_local;
        std::vector<bareLowRankMatrix<T>*> MyFarFieldMats_local;
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
        for(int b=0; b<HMatrices[0].MyBlocks.size(); b++) {
            const Block& B = *(HMatrices[0].MyBlocks[b]);
        	const Cluster& t = B.tgt_();
            const Cluster& s = B.src_();
			if( B.IsAdmissible() ){
				bool test = AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,MyFarFieldMats_local,reqrank);
				
				if(test){
					AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
				}
			}
            else {
            	// MyNearFieldMats.emplace_back(mat,I,J);
            	AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            }
		}

        #if _OPENMP
        #pragma omp critical
        #endif
        {
			// std::cout << "TEST "<<nb_hmatrix<< std::endl;
			for (int l=0;l<nb_hmatrix;l++){
				int count =l;
				// std::cout << "l: "<<l<< std::endl;
				while (count<MyFarFieldMats_local.size())
				{
					HMatrices[l].MyFarFieldMats.push_back(MyFarFieldMats_local[count]);
					count+=nb_hmatrix;
				}
				count=l;
				while (count<MyNearFieldMats_local.size())
				{
					HMatrices[l].MyNearFieldMats.push_back(MyNearFieldMats_local[count*nb_hmatrix]);
					count+=nb_hmatrix;
				}
			}
        }
	}


}

// Build a dense block
template< template<typename> class MultiLowRankMatrix, typename T>
void MultiHMatrix<MultiLowRankMatrix,T >::AddNearFieldMat(MultiIMatrix<T>& mat, const Cluster& t, const Cluster& s, std::vector<SubMatrix<T>*>& MyNearFieldMats_local){

	MultiSubMatrix<T> Local_MultiSubMatrix(mat, std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset());

	for (int l=0;l<nb_hmatrix;l++){
		SubMatrix<T>* submat = new SubMatrix<T>(Local_MultiSubMatrix[l]);
		MyNearFieldMats_local.push_back(submat);
	}


}

// Build a low rank block
template< template<typename> class MultiLowRankMatrix, typename T>
bool MultiHMatrix<MultiLowRankMatrix,T >::AddFarFieldMat(MultiIMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<bareLowRankMatrix<T>*>& MyFarFieldMats_local, const int& reqrank){
	MultiLowRankMatrix<T> Local_MultiLowRankMatrix(std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),mat.nb_matrix(),t.get_offset(),s.get_offset(),reqrank);
	Local_MultiLowRankMatrix.build(mat,t,xt,tabt,s,xs,tabs);

	if (Local_MultiLowRankMatrix.rank_of()!=1){
		for (int l=0;l<nb_hmatrix;l++){
			bareLowRankMatrix<T>* lrmat = new bareLowRankMatrix<T> (Local_MultiLowRankMatrix[l]);
			MyFarFieldMats_local.push_back(lrmat);
		}
		return 0;
	}
	else{
		return 1;
	}

}


template< template<typename> class MultiLowRankMatrix, typename T >
double Frobenius_absolute_error(const MultiHMatrix<MultiLowRankMatrix,T>& B, const MultiIMatrix<T>& A, int l){
	double myerr = 0;
	for(int j=0; j<B[l].MyFarFieldMats.size(); j++){
		double test = Frobenius_absolute_error(*(B[l].MyFarFieldMats[j]), A,l);
		myerr += std::pow(test,2);

	}

	double err = 0;
	MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

	return std::sqrt(err);
}

} //namespace
#endif