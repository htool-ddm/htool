#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP


#if _OPENMP
#  include <omp.h>
#endif

#include <cassert>
#include <fstream>
#include <mpi.h>
#include <map>
#include <memory>
#include "matrix.hpp"
#include "../misc/parametres.hpp"
#include "../clustering/cluster_tree.hpp"
#include "../wrappers/wrapper_mpi.hpp"

namespace htool {


//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//
// Friend functions
template< template<typename> class LowRankMatrix, typename T >
class HMatrix;

template< template<typename> class LowRankMatrix, typename T >
double Frobenius_absolute_error(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A);

// Class
template< template<typename> class LowRankMatrix, typename T >
class HMatrix: public Parametres{

private:
	// Data members
	int nr;
	int nc;
	int reqrank;
	int local_size;
	int local_offset;

	std::vector<Block*>		   MyBlocks;

	std::vector<LowRankMatrix<T>* > MyFarFieldMats;
	std::vector<SubMatrix<T>* >     MyNearFieldMats;
	std::vector<LowRankMatrix<T>*> MyDiagFarFieldMats;
	std::vector<SubMatrix<T>*> MyDiagNearFieldMats;
    std::vector<LowRankMatrix<T>*> MyStrictlyDiagFarFieldMats;
	std::vector<SubMatrix<T>*> MyStrictlyDiagNearFieldMats;


	std::shared_ptr<Cluster_tree> cluster_tree_s;
	std::shared_ptr<Cluster_tree> cluster_tree_t;

	mutable std::map<std::string, std::string> infos;

	MPI_Comm comm;
	int rankWorld,sizeWorld;


	// Internal methods
	Block* BuildBlockTree(const Cluster&, const Cluster&);
	void ComputeBlocks(IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	bool UpdateBlocks(IMatrix<T>&mat ,const Cluster&, const Cluster&, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<SubMatrix<T>*>&, std::vector<LowRankMatrix<T>*>&);
	void AddNearFieldMat(IMatrix<T>& mat, const Cluster& t, const Cluster& s, std::vector<SubMatrix<T>*>&);
	void AddFarFieldMat(IMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<LowRankMatrix<T>*>&, const int& reqrank=-1);
	void ComputeInfos(const std::vector<double>& mytimes);


public:
	// Build
	void build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without mass
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without tab
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius, tab and mass
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Symetric build
	void build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Full symetric constructor
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Symetric constructor without radius
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without mass
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without radius, tab and mass
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Build with precomputed clusters
	void build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<int>& tabs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor with precomputed clusters
	HMatrix(IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt,  const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const std::vector<int>& tabs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without tab and with precomputed clusters
	HMatrix(IMatrix<T>&,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt,  const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Symetric build with precomputed cluster
	void build(IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Full symetric constructor with precomputed cluster
	HMatrix(IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab and with precomputed cluster
	HMatrix(IMatrix<T>&,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

  // Destructor
	~HMatrix() {
		for (int i=0; i<MyBlocks.size(); i++)
			delete MyBlocks[i];
        for (int i=0; i<MyNearFieldMats.size();i++)
            delete MyNearFieldMats[i];
        for (int i=0; i<MyFarFieldMats.size();i++)
            delete MyFarFieldMats[i];

	}


	// Getters
	int nb_rows() const { return nr;}
	int nb_cols() const { return nc;}
	const MPI_Comm& get_comm() const {return comm;}
	int get_nlrmat() const {
		int res=MyFarFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;
	}
	int get_ndmat() const {
		int res=MyNearFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;
	}

	int get_rankworld() const {return rankWorld;}
	int get_sizeworld() const {return sizeWorld;}
	int get_local_size() const {return local_size;}
	int get_local_offset() const {return local_offset;}

    const Cluster_tree& get_cluster_tree_t() const{return *(cluster_tree_t.get());}
    const Cluster_tree& get_cluster_tree_s() const{return *(cluster_tree_s.get());}
	std::vector<std::pair<int,int>> get_MasterOffset_t() const {return cluster_tree_t->get_masteroffset();}
	std::vector<std::pair<int,int>> get_MasterOffset_s() const {return cluster_tree_s->get_masteroffset();}
    std::pair<int,int> get_MasterOffset_t(int i) const {return cluster_tree_t->get_masteroffset(i);}
    std::pair<int,int> get_MasterOffset_s(int i) const {return cluster_tree_s->get_masteroffset(i);}
	const std::vector<int>& get_permt() const {return cluster_tree_t->get_perm();}
	const std::vector<int>& get_perms() const {return cluster_tree_s->get_perm();}
    int get_permt(int i) const {return cluster_tree_t->get_perm(i);}
	int get_perms(int i) const {return cluster_tree_s->get_perm(i);}
	const std::vector<SubMatrix<T>*>& get_MyNearFieldMats() const {return MyNearFieldMats;}
	const std::vector<LowRankMatrix<T>*>& get_MyFarFieldMats() const {return MyFarFieldMats;}
	const std::vector<SubMatrix<T>*>& get_MyDiagNearFieldMats() const {return MyDiagNearFieldMats;}
	const std::vector<LowRankMatrix<T>*>& get_MyDiagFarFieldMats() const {return MyDiagFarFieldMats;}

	// Infos
	const std::map<std::string, std::string>& get_infos() const {return infos;}
  std::string get_infos (const std::string& key) const { return infos[key];}
	void add_info(const std::string& keyname, const std::string& value) const {infos[keyname]=value;}
	void print_infos() const;
	void save_infos(const std::string& outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string& sep = " = ") const;
	double compression() const; // 1- !!!
	friend double Frobenius_absolute_error<LowRankMatrix,T>(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A);

	// Mat vec prod
	void mvprod_global(const T* const in, T* const out,const int& mu=1) const;
	void mvprod_local(const T* const in, T* const out, T* const work, const int& mu) const;
	void mymvprod_local(const T* const in, T* const out, const int& mu) const;
    void mvprod_subrhs(const T* const in, T* const out, const int& mu, const int& offset, const int& size, const int& local_max_size_j) const;
	std::vector<T> operator*( const std::vector<T>& x) const;

	// Permutations
	template<typename U>
	void source_to_cluster_permutation(const U* const in, U* const out) const;
	template<typename U>
	void cluster_to_target_permutation(const U* const in, U* const out) const;

	// local to global
 	void local_to_global(const T* const in, T* const out, const int& mu) const;

    // Convert
    Matrix<T> to_dense() const;
    Matrix<T> to_dense_perm() const;

    // Apply Dirichlet condition
    void apply_dirichlet(const std::vector<int>& boundary);

};

// build
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

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

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(cluster_tree_t->get_local_cluster(),cluster_tree_s->get_head());
	if (B != NULL) MyBlocks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	std::sort(MyBlocks.begin(),MyBlocks.end(),comp_block());
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xs,tabs);
	mytimes[3] = MPI_Wtime() - time;

	// Infos
	ComputeInfos(mytimes);
}

// Full constructor
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
}

// Constructor without rt and rs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {

	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, gt, xs, std::vector<double>(xs.size(),0), tabs, gs, comm0);
}

// Constructor without gt and gs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, std::vector<double>(xt.size(),1), xs, rs, tabs, std::vector<double>(xs.size(),1), comm0);
}

// Constructor without tabt and tabs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs, comm0);
}

// Constructor without radius, mass and tab
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<R3>& xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, std::vector<double>(xt.size(),1), xs, std::vector<double>(xs.size(),0), tabs, std::vector<double>(xs.size(),1), comm0);
}

// Symetric build
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, MPI_Comm comm0){
	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabt.size() );

	MPI_Comm_dup(comm0,&comm);
	MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	cluster_tree_t = std::make_shared<Cluster_tree>(xt,rt,tabt,gt,comm);
	cluster_tree_s = cluster_tree_t;
	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(cluster_tree_t->get_local_cluster(),cluster_tree_t->get_head());
	if (B != NULL) MyBlocks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	std::sort(MyBlocks.begin(),MyBlocks.end(),comp_block());
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xt,tabt);
	mytimes[3] = MPI_Wtime() - time;

	// Infos
	ComputeInfos(mytimes);

}

// Full symetric constructor
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){

		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without rt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){
		this->build(mat,xt,std::vector<double>(xt.size(),0),tabt,gt,comm0);
}


// Symetric constructor without tabt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without gt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), comm(comm0){
		this->build(mat,xt,rt,tabt,std::vector<double>(xt.size(),1),comm0);
}

// Symetric constructor without rt, tabt and gt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,std::vector<double>(xt.size(),0),tabt,std::vector<double>(xt.size(),1),comm0);
}


// build with input cluster
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<int>& tabs, MPI_Comm comm0){

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	MPI_Comm_dup(comm0,&comm);
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();

	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();


	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(cluster_tree_t->get_local_cluster(),cluster_tree_s->get_head());
	if (B != NULL) MyBlocks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	std::sort(MyBlocks.begin(),MyBlocks.end(),comp_block());
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xs,tabs);
	mytimes[3] = MPI_Wtime() - time;

	// Infos
	ComputeInfos(mytimes);
}


// Full constructor with precomputed clusters
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const std::vector<int>& tabs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0) {
	this->build(mat, xt, tabt, xs, tabs, comm0);
}

// Constructor without tabt and tabs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, tabt, xs, tabs, comm0);
}


// Full symetric constructor
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0){

		this->build(mat,xt,tabt,xt,tabt,comm0);
}

// Symetric constructor without tabt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(IMatrix<T>& mat, const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0){
	std::vector<int> tabt(xt.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,tabt,xt,tabt,comm0);
}


// Build block tree
// TODO: recursivity -> stack for buildblocktree
template< template<typename> class LowRankMatrix, typename T >
Block* HMatrix<LowRankMatrix, T >::BuildBlockTree(const Cluster& t, const Cluster& s){
	Block* B = new Block(t,s);
	int bsize = t.get_size()*s.get_size();
	B->ComputeAdmissibility();
	if( B->IsAdmissible() && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()){
		MyBlocks.push_back(B);
		return NULL;
	}
	else if( s.IsLeaf() ){
				if( t.IsLeaf() ){
					return B;
				}
				else{
					Block* r1 = BuildBlockTree(t.get_son(0),s);
					Block* r2 = BuildBlockTree(t.get_son(1),s);
					if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL) && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()) {
						delete r1;
						delete r2;
						return B;
					}
					else {
						if (r1 != NULL) MyBlocks.push_back(r1);
						if (r2 != NULL) MyBlocks.push_back(r2);
                        delete B;
						return NULL;
					}
				}
	}
	else{
		if( t.IsLeaf() ){
			Block* r3 = BuildBlockTree(t,s.get_son(0));
			Block* r4 = BuildBlockTree(t,s.get_son(1));
			if ((bsize <= maxblocksize) && (r3 != NULL) && (r4 != NULL)&& t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()) {
				delete r3;
				delete r4;
				return B;
			}
			else {
				if (r3 != NULL) MyBlocks.push_back(r3);
				if (r4 != NULL) MyBlocks.push_back(r4);
                delete B;
				return NULL;
			}
		}
		else{

				// // Other kind of partition
				// Block* r1 = BuildBlockTree(t.get_son(0),s.get_son(0));
				// Block* r2 = BuildBlockTree(t.get_son(1),s.get_son(1));
				// Block* r3 = BuildBlockTree(t.get_son(1),s.get_son(0));
				// Block* r4 = BuildBlockTree(t.get_son(0),s.get_son(1));
				// if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL) && (r3 != NULL) && (r4 != NULL) && t.get_rank()==cluster_tree_t->get_local_cluster().get_rank() && t.get_depth()>=GetMinTargetDepth()) {
				// 	delete r1;
				// 	delete r2;
				// 	delete r3;
				// 	delete r4;
				// 	return B;
				// }
				// else {
				// 	if (r1 != NULL) Tasks.push_back(r1);
				// 	if (r2 != NULL) Tasks.push_back(r2);
				// 	if (r3 != NULL) Tasks.push_back(r3);
				// 	if (r4 != NULL) Tasks.push_back(r4);
				// 	return NULL;
				// }



			if (t.get_size()>s.get_size()){
				Block* r1 = BuildBlockTree(t.get_son(0),s);
				Block* r2 = BuildBlockTree(t.get_son(1),s);
				if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL)&& t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()) {
					delete r1;
					delete r2;
					return B;
				}
				else {
					if (r1 != NULL) MyBlocks.push_back(r1);
					if (r2 != NULL) MyBlocks.push_back(r2);
                    delete B;
					return NULL;
				}
			}
			else{
				Block* r3 = BuildBlockTree(t,s.get_son(0));
				Block* r4 = BuildBlockTree(t,s.get_son(1));
				if ((bsize <= maxblocksize) && (r3 != NULL) && (r4 != NULL)&& t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()) {
					delete r3;
					delete r4;
					return B;
				}
				else {
					if (r3 != NULL) MyBlocks.push_back(r3);
					if (r4 != NULL) MyBlocks.push_back(r4);
                    delete B;
					return NULL;
				}
			}
		}
	}
}

// // Scatter tasks
// template< template<typename> class LowRankMatrix, typename T >
// void HMatrix<LowRankMatrix, T >::ScatterTasks(){

// 	// std::cout << "Tasks : "<<Tasks.size()<<std::endl;
//   for(int b=0; b<Tasks.size(); b++){
//     	//if (b%sizeWorld == rankWorld)
//     if ((*(Tasks[b])).tgt_().get_rank() == cluster_tree_t->get_local_cluster().get_rank()){
//     		MyBlocks.push_back(Tasks[b]);
// 		}
// 	}
// 	std::cout << Tasks.size()<<" "<<MyBlocks.size()<<std::endl;
//     std::sort(MyBlocks.begin(),MyBlocks.end(),comp_block());
// 	// std::cout << "rank : "<<rankWorld<<" "<<"Block : "<<MyBlocks.size() <<std::endl;
// }

// Compute blocks recursively
// TODO: recursivity -> stack for compute blocks
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix,T >::ComputeBlocks(IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
    #if _OPENMP
    #pragma omp parallel
    #endif
    {
        // IMatrix<T> mat = mat;
        std::vector<SubMatrix<T>*>     MyNearFieldMats_local;
        std::vector<LowRankMatrix<T>*> MyFarFieldMats_local;
        // int tid = omp_get_thread_num();
        // std::cout<<"Hello World from thread = "+ NbrToStr(tid)<<std::endl;
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
        for(int b=0; b<MyBlocks.size(); b++) {
            const Block& B = *(MyBlocks[b]);
        	const Cluster& t = B.tgt_();
            const Cluster& s = B.src_();
            if( B.IsAdmissible() ){
        	    AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,MyFarFieldMats_local,reqrank);
            	if(MyFarFieldMats_local.back()->rank_of()==-1){
                    delete MyFarFieldMats_local.back();
            		MyFarFieldMats_local.pop_back();


            		if( s.IsLeaf() ){
            			if( t.IsLeaf() ){
            				// MyNearFieldMats.emplace_back(mat,I,J);
            				AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            			}
            			else{
            				bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            				bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            				if ((b1 != true) && (b2 != true))
            					// MyNearFieldMats.emplace_back(mat,I,J);
            					AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            				else {
            					if (b1 != true)
            						// 	MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
            						AddNearFieldMat(mat,t.get_son(0),s,MyNearFieldMats_local);
            					if (b2 != true)
            						// 	MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
            						AddNearFieldMat(mat,t.get_son(1),s,MyNearFieldMats_local);
            				}
            			}
            		}
            		else{
            			if( t.IsLeaf() ){
            				bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            				bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            				if ((b3 != true) && (b4 != true))
            					// MyNearFieldMats.emplace_back(mat,I,J);
            					AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            				else {
            					if (b3 != true)
            						// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
            						AddNearFieldMat(mat,t,s.get_son(0),MyNearFieldMats_local);
            					if (b4 != true)
            						// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
            						AddNearFieldMat(mat,t,s.get_son(1),MyNearFieldMats_local);
            				}
            			}
            			else{
            				if (t.get_size()>s.get_size()){
            					bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            					bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            					if ((b1 != true) && (b2 != true))
            						// MyNearFieldMats.emplace_back(mat,I,J);
            						AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            					else {
            						if (b1 != true)
            								// MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
            								AddNearFieldMat(mat,t.get_son(0),s,MyNearFieldMats_local);
            						if (b2 != true)
            								// MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
            								AddNearFieldMat(mat,t.get_son(1),s,MyNearFieldMats_local);
            					}
            				}
            				else{
            					bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            					bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
            					if ((b3 != true) && (b4 != true))
            						// MyNearFieldMats.emplace_back(mat,I,J);
            						AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            					else {
            						if (b3 != true)
            								// MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
            								AddNearFieldMat(mat,t,s.get_son(0),MyNearFieldMats_local);
            						if (b4 != true)
            								// MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
            								AddNearFieldMat(mat,t,s.get_son(1),MyNearFieldMats_local);
            					}
            				}
            			}
            		}
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
            MyFarFieldMats.insert(MyFarFieldMats.end(),MyFarFieldMats_local.begin(),MyFarFieldMats_local.end());
            MyNearFieldMats.insert(MyNearFieldMats.end(),MyNearFieldMats_local.begin(),MyNearFieldMats_local.end());
        }
    }

    // Build vectors of pointers for diagonal blocks
    for (int i=0;i<MyFarFieldMats.size();i++){
        if (local_offset<=MyFarFieldMats[i]->get_offset_j() && MyFarFieldMats[i]->get_offset_j()<local_offset+local_size){
            MyDiagFarFieldMats.push_back(MyFarFieldMats[i]);
            if (MyFarFieldMats[i]->get_offset_j()==MyFarFieldMats[i]->get_offset_i())
                MyStrictlyDiagFarFieldMats.push_back(MyFarFieldMats[i]);
        }
    }
    for (int i=0;i<MyNearFieldMats.size();i++){
        if (local_offset<=MyNearFieldMats[i]->get_offset_j() && MyNearFieldMats[i]->get_offset_j()<local_offset+local_size){
            MyDiagNearFieldMats.push_back(MyNearFieldMats[i]);
            if (MyNearFieldMats[i]->get_offset_j()==MyNearFieldMats[i]->get_offset_i())
                MyStrictlyDiagNearFieldMats.push_back(MyNearFieldMats[i]);
        }
    }
}

template< template<typename> class LowRankMatrix, typename T >
bool HMatrix<LowRankMatrix,T >::UpdateBlocks(IMatrix<T>& mat,const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<SubMatrix<T>*>& MyNearFieldMats_local, std::vector<LowRankMatrix<T>*>& MyFarFieldMats_local){
	int bsize = t.get_size()*s.get_size();
	Block B(t,s);
	B.ComputeAdmissibility();
	if( B.IsAdmissible() ){

		AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,MyFarFieldMats_local,reqrank);
		if(MyFarFieldMats_local.back()->rank_of()!=-1){
			return true;
		}
		else {
            delete MyFarFieldMats_local.back();
			MyFarFieldMats_local.pop_back();
		}
	}
	if( s.IsLeaf() ){
		if( t.IsLeaf() ){
			return false;
		}
		else{
			bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
				return false;
			else {
				if (b1 != true)
					// MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
					AddNearFieldMat(mat,t.get_son(0),s,MyNearFieldMats_local);
				if (b2 != true)
					// MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
					AddNearFieldMat(mat,t.get_son(1),s,MyNearFieldMats_local);
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
				return false;
			else{
				if (b3 != true)
					// MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
					AddNearFieldMat(mat,t,s.get_son(0),MyNearFieldMats_local);
				if (b4 != true)
					// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
					AddNearFieldMat(mat,t,s.get_son(1),MyNearFieldMats_local);
				return true;
			}
		}
		else{

			if (t.get_size()>s.get_size()){
				bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
					return false;
				else {
					if (b1 != true)
						// 	MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
						AddNearFieldMat(mat,t.get_son(0),s,MyNearFieldMats_local);
					if (b2 != true)
						// 	MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
						AddNearFieldMat(mat,t.get_son(1),s,MyNearFieldMats_local);
					return true;
				}
			}
			else{
				bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
					return false;
				else{
					if (b3 != true)
						// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
						AddNearFieldMat(mat,t,s.get_son(0),MyNearFieldMats_local);
					if (b4 != true)
						// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
						AddNearFieldMat(mat,t,s.get_son(1),MyNearFieldMats_local);
					return true;
				}
			}
		}
	}
}

// Build a dense block
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::AddNearFieldMat(IMatrix<T>& mat, const Cluster& t, const Cluster& s, std::vector<SubMatrix<T>*>& MyNearFieldMats_local){
    SubMatrix<T>* submat = new SubMatrix<T>(mat, std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset());

	MyNearFieldMats_local.push_back(submat);

}

// Build a low rank block
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::AddFarFieldMat(IMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<LowRankMatrix<T>*>& MyFarFieldMats_local, const int& reqrank){
    LowRankMatrix<T>* lrmat = new LowRankMatrix<T> (std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset(),reqrank);
    MyFarFieldMats_local.push_back(lrmat);
	MyFarFieldMats_local.back()->build(mat,t,xt,tabt,s,xs,tabs);

}

// Compute infos
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::ComputeInfos(const std::vector<double>& mytime){
	// 0 : cluster tree ; 1 : block tree ; 2 : scatter tree ; 3 : compute blocks ;
	std::vector<double> maxtime(4), meantime(4);
	// 0 : dense mat ; 1 : lr mat ; 2 : rank ; 3 : local_size
	std::vector<int> maxinfos(4,0),mininfos(4,std::max(nc,nr));
	std::vector<double> meaninfos(4,0);
	// Infos
	for (int i=0;i<MyNearFieldMats.size();i++){
		int size = MyNearFieldMats[i]->nb_rows()*MyNearFieldMats[i]->nb_cols();
		maxinfos[0] = std::max(maxinfos[0],size);
		mininfos[0] = std::min(mininfos[0],size);
		meaninfos[0] += size;
	}
	for (int i=0;i<MyFarFieldMats.size();i++){
		int size = MyFarFieldMats[i]->nb_rows()*MyFarFieldMats[i]->nb_cols();
		int rank = MyFarFieldMats[i]->rank_of();
		maxinfos[1] = std::max(maxinfos[1],size);
		mininfos[1] = std::min(mininfos[1],size);
		meaninfos[1] += size;
		maxinfos[2] = std::max(maxinfos[2],rank);
		mininfos[2] = std::min(mininfos[2],rank);
		meaninfos[2] += rank;
	}
    maxinfos[3]=local_size;
    mininfos[3]=local_size;
    meaninfos[3]=local_size;

	if (rankWorld==0){
		MPI_Reduce(MPI_IN_PLACE, &(maxinfos[0]), 4, MPI_INT, MPI_MAX, 0,comm);
		MPI_Reduce(MPI_IN_PLACE, &(mininfos[0]), 4, MPI_INT, MPI_MIN, 0,comm);
		MPI_Reduce(MPI_IN_PLACE, &(meaninfos[0]),4, MPI_DOUBLE, MPI_SUM, 0,comm);
	}
	else{
		MPI_Reduce(&(maxinfos[0]), &(maxinfos[0]), 4, MPI_INT, MPI_MAX, 0,comm);
		MPI_Reduce(&(mininfos[0]), &(mininfos[0]), 4, MPI_INT, MPI_MIN, 0,comm);
		MPI_Reduce(&(meaninfos[0]), &(meaninfos[0]),4, MPI_DOUBLE, MPI_SUM, 0,comm);
	}

	int nlrmat = this->get_nlrmat();
	int ndmat = this->get_ndmat();
	meaninfos[0] = (ndmat  == 0 ? 0 : meaninfos[0]/ndmat);
	meaninfos[1] = (nlrmat == 0 ? 0 : meaninfos[1]/nlrmat);
	meaninfos[2] = (nlrmat == 0 ? 0 : meaninfos[2]/nlrmat);
    meaninfos[3] = meaninfos[3]/sizeWorld;
	mininfos[0] = (ndmat  == 0 ? 0 : mininfos[0]);
	mininfos[1] = (nlrmat  == 0 ? 0 : mininfos[1]);
	mininfos[2] = (nlrmat  == 0 ? 0 : mininfos[2]);

	// timing
	MPI_Reduce(&(mytime[0]), &(maxtime[0]), 4, MPI_DOUBLE, MPI_MAX, 0,comm);
	MPI_Reduce(&(mytime[0]), &(meantime[0]), 4, MPI_DOUBLE, MPI_SUM, 0,comm);

	meantime /= sizeWorld;

	// Times
	infos["Cluster_tree_mean"]=NbrToStr(meantime[0]);
	infos["Cluster_tree_max"]=NbrToStr(maxtime[0]);
	infos["Block_tree_mean"]=NbrToStr(meantime[1]);
	infos["Block_tree_max"]=NbrToStr(maxtime[1]);
	infos["Scatter_tree_mean"]=NbrToStr(meantime[2]);
	infos["Scatter_tree_max"]=NbrToStr(maxtime[2]);
	infos["Blocks_mean"]=NbrToStr(meantime[3]);
	infos["Blocks_max"]=NbrToStr(maxtime[3]);

	// Size
	infos["Source_size"] = NbrToStr(this->nc);
	infos["Target_size"] = NbrToStr(this->nr);
	infos["Dense_block_size_max"]  = NbrToStr(maxinfos[0]);
	infos["Dense_block_size_mean"] = NbrToStr(meaninfos[0]);
	infos["Dense_block_size_min"]  = NbrToStr(mininfos[0]);
	infos["Low_rank_block_size_max"]  = NbrToStr(maxinfos[1]);
	infos["Low_rank_block_size_mean"] = NbrToStr(meaninfos[1]);
	infos["Low_rank_block_size_min"]  = NbrToStr(mininfos[1]);

	infos["Rank_max"]  = NbrToStr(maxinfos[2]);
	infos["Rank_mean"] = NbrToStr(meaninfos[2]);
	infos["Rank_min"]  = NbrToStr(mininfos[2]);
	infos["Number_of_lrmat"] = NbrToStr(nlrmat);
	infos["Number_of_dmat"]  = NbrToStr(ndmat);
	infos["Compression"] = NbrToStr(this->compression());
    infos["Local_size_max"]  = NbrToStr(maxinfos[3]);
    infos["Local_size_mean"] = NbrToStr(meaninfos[3]);
    infos["Local_size_min"]  = NbrToStr(mininfos[3]);


	infos["Number_of_MPI_tasks"] = NbrToStr(sizeWorld);
    #if _OPENMP
    infos["Number_of_threads_per_tasks"] = NbrToStr(omp_get_max_threads());
    infos["Number_of_procs"] = NbrToStr(sizeWorld*omp_get_max_threads());
    #else
    infos["Number_of_procs"] = NbrToStr(sizeWorld);
    #endif


	infos["Eta"] = NbrToStr(GetEta());
	infos["Eps"] = NbrToStr(GetEpsilon());
	infos["MinTargetDepth"] = NbrToStr(GetMinTargetDepth());
	infos["MinSourceDepth"] = NbrToStr(GetMinSourceDepth());


}



template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mymvprod_local(const T* const in, T* const out, const int& mu) const{

	std::fill(out,out+local_size*mu,0);

	// Contribution champ lointain
    #if _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<T> temp(local_size*mu,0);
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
    	for(int b=0; b<MyFarFieldMats.size(); b++){
    		const LowRankMatrix<T>&  M  = *(MyFarFieldMats[b]);
    		int offset_i     = M.get_offset_i();
    		int offset_j     = M.get_offset_j();

    		M.add_mvprod_row_major(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);

    	}
    	// Contribution champ proche
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
    	for(int b=0; b<MyNearFieldMats.size(); b++){
    		const SubMatrix<T>&  M  = *(MyNearFieldMats[b]);
    		int offset_i     = M.get_offset_i();
    		int offset_j     = M.get_offset_j();

    		M.add_mvprod_row_major(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);
    	}
        #if _OPENMP
        #pragma omp critical
        #endif
        std::transform (temp.begin(), temp.end(), out, out, std::plus<T>());

    }

}


// template< template<typename> class LowRankMatrix, typename T>
// void HMatrix<LowRankMatrix,T >::local_to_global(const T* const in, T* const out, const int& mu) const{
// 	// Allgather
// 	std::vector<int> recvcounts(sizeWorld);
// 	std::vector<int>  displs(sizeWorld);
//
// 	displs[0] = 0;
//
// 	for (int i=0; i<sizeWorld; i++) {
// 		recvcounts[i] = (cluster_tree_t->get_masteroffset(i).second)*mu;
// 		if (i > 0)
// 			displs[i] = displs[i-1] + recvcounts[i-1];
// 	}
//
// 	MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out + (mu==1 ? 0 : mu*nc), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
//
//     //
//     if (mu!=1){
//         for (int i=0 ;i<mu;i++){
//             for (int j=0; j<sizeWorld;j++){
//                 std::copy_n(out+mu*nc+displs[j]+i*recvcounts[j]/mu,recvcounts[j]/mu,out+i*nc+displs[j]/mu);
//             }
//         }
//     }
// }

template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::local_to_global(const T* const in, T* const out, const int& mu) const{
  // Allgather
  std::vector<int> recvcounts(sizeWorld);
  std::vector<int>  displs(sizeWorld);


  displs[0] = 0;

  for (int i=0; i<sizeWorld; i++) {
      recvcounts[i] = (cluster_tree_t->get_masteroffset(i).second)*mu;
      if (i > 0)
          displs[i] = displs[i-1] + recvcounts[i-1];
  }



  MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);


}




template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_local(const T* const in, T* const out, T* const work, const int& mu) const{
	double time = MPI_Wtime();

    this->local_to_global(in, work,mu);
    this->mymvprod_local(work,out,mu);

	infos["nb_mat_vec_prod"] = NbrToStr(1+StrToNbr<int>(infos["nb_mat_vec_prod"]));
	infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime()-time+StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}


template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_global(const T* const in, T* const out, const int& mu) const{
    double time = MPI_Wtime();

    if (mu==1){
    	std::vector<T> out_perm(local_size);
        std::vector<T> buffer(std::max(nc,nr));

        // Permutation
        cluster_tree_s->global_to_cluster(in,buffer.data());

        //
    	mymvprod_local(buffer.data(),out_perm.data(),1);

        // Allgather
    	std::vector<int> recvcounts(sizeWorld);
    	std::vector<int>  displs(sizeWorld);

    	displs[0] = 0;

    	for (int i=0; i<sizeWorld; i++) {
    		recvcounts[i] = cluster_tree_t->get_masteroffset(i).second*mu;
    		if (i > 0)
    			displs[i] = displs[i-1] + recvcounts[i-1];
    	}

    	MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), buffer.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);

        // Permutation
        cluster_tree_t->cluster_to_global(buffer.data(),out);

    }
    else{


    	std::vector<T> in_perm(std::max(nr,nc)*mu*2);
    	std::vector<T> out_perm(local_size*mu);
        std::vector<T> buffer(nc);

        for (int i=0;i<mu;i++){
    	    // Permutation
    	    cluster_tree_s->global_to_cluster(in+i*nc,buffer.data());


            // Transpose
            for (int j=0;j<nc;j++){
                in_perm[i+j*mu]=buffer[j];
            }
        }

    	mymvprod_local(in_perm.data(),in_perm.data()+nc*mu,mu);

        // Tranpose
        for (int i=0;i<mu;i++){
            for (int j=0;j<local_size;j++){
                out_perm[i*local_size+j]=in_perm[i+j*mu+nc*mu];
            }
        }



    	// Allgather
    	std::vector<int> recvcounts(sizeWorld);
    	std::vector<int>  displs(sizeWorld);

    	displs[0] = 0;

    	for (int i=0; i<sizeWorld; i++) {
    		recvcounts[i] = cluster_tree_t->get_masteroffset(i).second*mu;
    		if (i > 0)
    			displs[i] = displs[i-1] + recvcounts[i-1];
    	}

    	MPI_Allgatherv(out_perm.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), in_perm.data() + mu*nr, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);

        for (int i=0 ;i<mu;i++){
            for (int j=0; j<sizeWorld;j++){
                std::copy_n(in_perm.data()+mu*nr+displs[j]+i*recvcounts[j]/mu,recvcounts[j]/mu,in_perm.data()+i*nr+displs[j]/mu);
            }

            // Permutation
            cluster_tree_t->cluster_to_global(in_perm.data()+i*nr,out+i*nr);
        }
    }
	// Timing
	infos["nb_mat_vec_prod"] = NbrToStr(1+StrToNbr<int>(infos["nb_mat_vec_prod"]));
	infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime()-time+StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}

template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_subrhs(const T* const in, T* const out, const int& mu, const int& offset, const int& size, const int& local_max_size_j) const{
    std::fill(out,out+local_size*mu,0);

	// Contribution champ lointain
    #if _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<T> temp(local_size*mu,0);
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
    	for(int b=0; b<MyFarFieldMats.size(); b++){
            const LowRankMatrix<T>&  M  = *(MyFarFieldMats[b]);
            int offset_i     = M.get_offset_i();
            int offset_j     = M.get_offset_j();
            int size_j       = M.nb_cols();

            if (offset_j <= offset+size && offset<= offset_j+size_j){
        		M.add_mvprod_row_major(in+(offset_j-offset+local_max_size_j)*mu,temp.data()+(offset_i-local_offset)*mu,mu);
            }
    	}
    	// Contribution champ proche
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
    	for(int b=0; b<MyNearFieldMats.size(); b++){
            const SubMatrix<T>&  M  = *(MyNearFieldMats[b]);
            int offset_i     = M.get_offset_i();
            int offset_j     = M.get_offset_j();
            int size_j       = M.nb_cols();

            if (offset_j <= offset+size && offset<= offset_j+size_j){
    		    M.add_mvprod_row_major(in+(offset_j-offset+local_max_size_j)*mu,temp.data()+(offset_i-local_offset)*mu,mu);
    		}
    	}
        #if _OPENMP
        #pragma omp critical
        #endif
        std::transform (temp.begin(), temp.end(), out, out, std::plus<T>());

    }

}

template< template<typename> class LowRankMatrix, typename T>
template<typename U>
void HMatrix<LowRankMatrix,T >::source_to_cluster_permutation(const U* const in, U* const out) const {
	cluster_tree_s->global_to_cluster(in,out);
}

template< template<typename> class LowRankMatrix, typename T>
template<typename U>
void HMatrix<LowRankMatrix,T >::cluster_to_target_permutation(const U* const in, U* const out) const{
	cluster_tree_t->cluster_to_global(in,out);
}




template< template<typename> class LowRankMatrix, typename T>
std::vector<T> HMatrix<LowRankMatrix,T >::operator*(const std::vector<T>& x) const{
	assert(x.size()==nc);
	std::vector<T> result(nr,0);
	mvprod_global(x.data(),result.data(),1);
	return result;
}


template< template<typename> class LowRankMatrix, typename T >
double HMatrix<LowRankMatrix,T >::compression() const{

	double mycomp = 0.;
	double size = ((long int)this->nr)*this->nc;
	double nr_b ,nc_b,rank;

	for(int j=0; j<MyFarFieldMats.size(); j++){
		nr_b  = MyFarFieldMats[j]->nb_rows();
		nc_b   = MyFarFieldMats[j]->nb_cols();
		rank = MyFarFieldMats[j]->rank_of();
		mycomp += rank*(nr_b + nc_b)/size;
	}

	for(int j=0; j<MyNearFieldMats.size(); j++){
		nr_b   = MyNearFieldMats[j]->nb_rows();
		nc_b   = MyNearFieldMats[j]->nb_cols();
		mycomp += nr_b*nc_b/size;
	}

	double comp = 0;
	MPI_Allreduce(&mycomp, &comp, 1, MPI_DOUBLE, MPI_SUM, comm);

	return 1-comp;
}

template<template<typename> class LowRankMatrix,typename T >
void HMatrix<LowRankMatrix,T >::print_infos() const{
	int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

	if (rankWorld==0){
		for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
			std::cout<<it->first<<"\t"<<it->second<<std::endl;
		}
		std::cout << std::endl;
	}
}

template<template<typename> class LowRankMatrix,typename T >
void HMatrix<LowRankMatrix,T >::save_infos(const std::string& outputname,std::ios_base::openmode mode, const std::string& sep) const{
	int rankWorld;
  MPI_Comm_rank(comm, &rankWorld);

	if (rankWorld==0){
		std::ofstream outputfile(outputname,mode);
		if (outputfile){
			for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
				outputfile<<it->first<<sep<<it->second<<std::endl;
			}
			outputfile.close();
		}
		else{
			std::cout << "Unable to create "<<outputname<<std::endl;
		}
	}
}

template< template<typename> class LowRankMatrix, typename T >
double Frobenius_absolute_error(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A){
	double myerr = 0;
	for(int j=0; j<B.MyFarFieldMats.size(); j++){
		double test = Frobenius_absolute_error(*(B.MyFarFieldMats[j]), A);
		myerr += std::pow(test,2);

	}

	double err = 0;
	MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

	return std::sqrt(err);
}
template< template<typename> class LowRankMatrix, typename T >
Matrix<T> HMatrix<LowRankMatrix,T >::to_dense() const{
    Matrix<T> Dense(nr,nc);
    // Internal dense blocks
    for (int l=0;l<MyNearFieldMats.size();l++){
      const SubMatrix<T>& submat = *(MyNearFieldMats[l]);
      int local_nr = submat.nb_rows();
      int local_nc = submat.nb_cols();
      int offset_i = submat.get_offset_i();
      int offset_j = submat.get_offset_j();
      for (int k=0;k<local_nc;k++){
        std::copy_n(&(submat(0,k)),local_nr,Dense.data()+offset_i+(offset_j+k)*local_size);
      }
    }

    // Internal compressed block
    Matrix<T> FarFielBlock(local_size,local_size);
    for (int l=0;l<MyFarFieldMats.size();l++){
      const LowRankMatrix<T>& lmat = *(MyFarFieldMats[l]);
      int local_nr = lmat.nb_rows();
      int local_nc = lmat.nb_cols();
      int offset_i = lmat.get_offset_i();
      int offset_j = lmat.get_offset_j();
      FarFielBlock.resize(local_nr,local_nc);
      lmat.get_whole_matrix(&(FarFielBlock(0,0)));
      for (int k=0;k<local_nc;k++){
        std::copy_n(&(FarFielBlock(0,k)),local_nr,Dense.data()+offset_i+(offset_j+k)*local_size);
      }
    }
    return Dense;
}

template< template<typename> class LowRankMatrix, typename T >
Matrix<T> HMatrix<LowRankMatrix,T >::to_dense_perm() const{
	Matrix<T> Dense(nr,nc);
	// Internal dense blocks
	for (int l=0;l<MyNearFieldMats.size();l++){
		const SubMatrix<T>& submat = *(MyNearFieldMats[l]);
		int local_nr = submat.nb_rows();
		int local_nc = submat.nb_cols();
		int offset_i = submat.get_offset_i();
		int offset_j = submat.get_offset_j();
		for (int k=0;k<local_nc;k++)
			for (int j=0;j<local_nr;j++)
				Dense(get_permt(j+offset_i),get_perms(k+offset_j))=submat(j,k);
	}

	// Internal compressed block
	Matrix<T> FarFielBlock(local_size,local_size);
	for (int l=0;l<MyFarFieldMats.size();l++){
		const LowRankMatrix<T>& lmat = *(MyFarFieldMats[l]);
		int local_nr = lmat.nb_rows();
		int local_nc = lmat.nb_cols();
		int offset_i = lmat.get_offset_i();
		int offset_j = lmat.get_offset_j();
		FarFielBlock.resize(local_nr,local_nc);
		lmat.get_whole_matrix(&(FarFielBlock(0,0)));
		for (int k=0;k<local_nc;k++)
			for (int j=0;j<local_nr;j++)
				Dense(get_permt(j+offset_i),get_perms(k+offset_j))=FarFielBlock(j,k);
	}
	return Dense;
}

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix,T >::apply_dirichlet(const std::vector<int>& boundary){
    // Renum
    std::vector<int> boundary_renum(boundary.size());
    cluster_tree_t->global_to_cluster(boundary.data(),boundary_renum.data());

    //
    for (int j=0;j<MyStrictlyDiagNearFieldMats.size();j++){
        SubMatrix<T>& submat = *(MyStrictlyDiagNearFieldMats[j]);
        int local_nr = submat.nb_rows();
        int local_nc = submat.nb_cols();
        int offset_i = submat.get_offset_i();
        int offset_j = submat.get_offset_j();
        for (int i=offset_i;i<offset_i+std::min(local_nr,local_nc);i++){
            if (boundary_renum[i])
                submat(i-offset_i,i-offset_i)=1e30;
        }
    }
}

} //namespace
#endif
