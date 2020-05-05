#ifndef HTOOL_MULTIHMATRIX_HPP
#define HTOOL_MULTIHMATRIX_HPP

#include <cassert>
#include <fstream>
#include <mpi.h>
#include <map>
#include <memory>
#include "matrix.hpp"
#include "../misc/parametres.hpp"
#include "../clustering/cluster.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "../blocks/blocks.hpp"
#include "hmatrix.hpp"
#include "multimatrix.hpp"
#include "../multilrmat/multilrmat.hpp"


namespace htool {

// Friend functions --- forward declaration
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
class HMatrix;

template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
class MultiHMatrix;

template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
double Frobenius_absolute_error(const MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>& B, const MultiIMatrix<T>& A, int l);

// Class
template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
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

	std::vector<HMatrix<T,bareLowRankMatrix,ClusterImpl> >    HMatrices;
	// std::vector<Block*>		   Tasks;
	// std::vector<Block*>		   MyBlocks;

    std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_s;
	std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_t;


public:

	// Build
	void build(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor
	MultiHMatrix(MultiIMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius, tab and mass
	MultiHMatrix(MultiIMatrix<T>&, const std::vector<R3>& xt, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	
	// Internal methods
	void ComputeBlocks(MultiIMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	void AddNearFieldMat(MultiIMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, std::vector<SubMatrix<T>*>&);
	bool AddFarFieldMat(MultiIMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<bareLowRankMatrix<T,ClusterImpl>*>&, const int& reqrank=-1);

	HMatrix<T,bareLowRankMatrix,ClusterImpl>&  operator[](int j){return HMatrices[j];}; 
    const HMatrix<T,bareLowRankMatrix,ClusterImpl>&  operator[](int j) const {return HMatrices[j];}; 


	friend double Frobenius_absolute_error<T,MultiLowRankMatrix,ClusterImpl>(const MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>& B, const MultiIMatrix<T>& A, int l);

    // // Destructor
	// ~MultiHMatrix() {
	// 	for (int i=0; i<Tasks.size(); i++)
	// 		delete Tasks[i];
	// }

	// Getters
	int nb_rows() const { return nr;}
	int nb_cols() const { return nc;}
	const MPI_Comm& get_comm() const {return comm;}
	int get_nlrmat(int i) const {
		int res=HMatrices[i].MyFarFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;
	}
	int get_ndmat(int i) const {
		int res=HMatrices[i].MyNearFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;
	}
	int nb_hmats() const { return nb_hmatrix;}	

	// Mat vec prod
	void mvprod_global(int i,const T* const in, T* const out,const int& mu=1) const{
		HMatrices[i].mvprod_global(in,out,mu);
	}
};

// build
template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
void MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>::build(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	MPI_Comm_dup(comm0,&comm);
	MPI_Comm_size(comm, &sizeWorld);
	MPI_Comm_rank(comm, &rankWorld);

	std::vector<double> mytimes(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	cluster_tree_t = std::make_shared<ClusterImpl>(); // target
	cluster_tree_s = std::make_shared<ClusterImpl>(); // source
	cluster_tree_t->build(xt,rt,tabt,gt,-1,comm);
	cluster_tree_s->build(xs,rs,tabs,gs,-1,comm);

	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();

	mytimes[0] = MPI_Wtime() - time;

	// Hmatrices
	for (int l=0;l<nb_hmatrix;l++){
		HMatrices.push_back(HMatrix<T,bareLowRankMatrix,ClusterImpl> (nr,nc,cluster_tree_t,cluster_tree_s));
	}

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block<ClusterImpl>* B = HMatrices[0].BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_s->get_root());
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
template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>::MultiHMatrix(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), nb_hmatrix(mat.nb_matrix()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
}

// Constructor without radius, mass and tab
template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>::MultiHMatrix(MultiIMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<R3>& xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), nb_hmatrix(mat.nb_matrix()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, std::vector<double>(xt.size(),1), xs, std::vector<double>(xs.size(),0), tabs, std::vector<double>(xs.size(),1), comm0);
}


template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
void MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>::ComputeBlocks(MultiIMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
    #if _OPENMP
    #pragma omp parallel
    #endif
	{
        std::vector<SubMatrix<T>*>     MyNearFieldMats_local;
        std::vector<bareLowRankMatrix<T,ClusterImpl>*> MyFarFieldMats_local;
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
        for(int b=0; b<HMatrices[0].MyBlocks.size(); b++) {
            const Block<ClusterImpl>& B = *(HMatrices[0].MyBlocks[b]);
        	const Cluster<ClusterImpl>& t = B.tgt_();
            const Cluster<ClusterImpl>& s = B.src_();
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
			for (int l=0;l<nb_hmatrix;l++){
				int count =l;
				while (count<MyFarFieldMats_local.size())
				{
					HMatrices[l].MyFarFieldMats.push_back(MyFarFieldMats_local[count]);
					count+=nb_hmatrix;
				}
				count=l;
				while (count<MyNearFieldMats_local.size())
				{
					HMatrices[l].MyNearFieldMats.push_back(MyNearFieldMats_local[count]);
					count+=nb_hmatrix;
				}
			}
        }
	}


}

// Build a dense block
template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
void MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>::AddNearFieldMat(MultiIMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, std::vector<SubMatrix<T>*>& MyNearFieldMats_local){

	MultiSubMatrix<T> Local_MultiSubMatrix(mat, std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset());

	for (int l=0;l<nb_hmatrix;l++){
		SubMatrix<T>* submat = new SubMatrix<T>(Local_MultiSubMatrix[l]);
		MyNearFieldMats_local.push_back(submat);
	}


}

// Build a low rank block
template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
bool MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>::AddFarFieldMat(MultiIMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<bareLowRankMatrix<T,ClusterImpl>*>& MyFarFieldMats_local, const int& reqrank){
	MultiLowRankMatrix<T,ClusterImpl> Local_MultiLowRankMatrix(std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),mat.nb_matrix(),t.get_offset(),s.get_offset(),reqrank);
	Local_MultiLowRankMatrix.build(mat,t,xt,tabt,s,xs,tabs);

	if (Local_MultiLowRankMatrix.rank_of()!=-1){
		for (int l=0;l<nb_hmatrix;l++){
			bareLowRankMatrix<T,ClusterImpl>* lrmat = new bareLowRankMatrix<T,ClusterImpl> (Local_MultiLowRankMatrix[l]);
			MyFarFieldMats_local.push_back(lrmat);
		}
		return 0;
	}
	else{
		return 1;
	}

}


template<typename T, template<typename,typename> class MultiLowRankMatrix, class ClusterImpl>
double Frobenius_absolute_error(const MultiHMatrix<T,MultiLowRankMatrix,ClusterImpl>& B, const MultiIMatrix<T>& A, int l){
	double myerr = 0;
	std::vector<bareLowRankMatrix<T,ClusterImpl>* > MyFarFieldMats=B[l].get_MyFarFieldMats();
	for(int j=0; j<MyFarFieldMats.size(); j++){
		double test = Frobenius_absolute_error(*(MyFarFieldMats[j]), A,l);
		myerr += std::pow(test,2);

	}

	double err = 0;
	MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

	return std::sqrt(err);
}

} //namespace
#endif