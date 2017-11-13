#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#include <cassert>
#include <fstream>
#include <mpi.h>
#include <map>
#include <memory>
#include "matrix.hpp"
#include "parametres.hpp"
#include "cluster_tree.hpp"
#include "wrapper_mpi.hpp"
#include "infos.hpp"

namespace htool {


//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//
// TODO visualisation for hmat
// Friend functions
template< template<typename> class LowRankMatrix, typename T >
class HMatrix;

template< template<typename> class LowRankMatrix, typename T >
double Frobenius_absolute_error(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A);

// Class
template< template<typename> class LowRankMatrix, typename T >
class HMatrix: public Parametres, private Counter<HMatrix<LowRankMatrix,T>>{

private:
	// Data members
	int nr;
	int nc;
	int reqrank;
	int local_size;
	int local_offset;

	std::vector<Block*>		   Tasks;
	std::vector<Block*>		   MyBlocks;

	std::vector<LowRankMatrix<T> > MyFarFieldMats;
	std::vector<SubMatrix<T> >     MyNearFieldMats;
	std::vector<int> MyDiagFarFieldMats;
	std::vector<int> MyDiagNearFieldMats;


	std::shared_ptr<Cluster_tree> cluster_tree_s;
	std::shared_ptr<Cluster_tree> cluster_tree_t;

	std::map<std::string, double> stats;

	MPI_Comm comm;
	int rankWorld,sizeWorld;

	std::string name;

	// Internal methods
	void ScatterTasks();
	Block* BuildBlockTree(const Cluster&, const Cluster&);
	void ComputeBlocks(const IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	bool UpdateBlocks(const IMatrix<T>&mat ,const Cluster&, const Cluster&, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	void AddNearFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s);
	void AddFarFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, const int& reqrank=-1);
	void ComputeStats(const std::vector<double>& mytimes);


public:
	// make howMany public
	using Counter<HMatrix>::howMany;

	// Build
	void build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without tab
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius, tab and mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<R3>&xs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Symetric build
	void build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Full symetric constructor
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Symetric constructor without radius
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without radius, tab and mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Build with precomputed clusters
	void build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<int>& tabs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor with precomputed clusters
	HMatrix(const IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt,  const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const std::vector<int>& tabs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without tab and with precomputed clusters
	HMatrix(const IMatrix<T>&,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt,  const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Symetric build with precomputed cluster
	void build(const IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Full symetric constructor with precomputed cluster
	HMatrix(const IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab and with precomputed cluster
	HMatrix(const IMatrix<T>&,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const int& reqrank=-1, const std::string& name = "", MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

  // Destructor
	~HMatrix() {
		for (int i=0; i<Tasks.size(); i++)
			delete Tasks[i];
	}


	// Getters
	int nb_rows() const { return nr;}
	int nb_cols() const { return nc;}
	MPI_Comm get_comm() const {return comm;}
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

	std::vector<std::pair<int,int>> get_MasterOffset_t() const {return cluster_tree_t->get_masteroffset();}
	std::vector<std::pair<int,int>> get_MasterOffset_s() const {return cluster_tree_s->get_masteroffset();}
	std::vector<int> get_permt() const {return cluster_tree_t->get_perm();}
	std::vector<int> get_perms() const {return cluster_tree_s->get_perm();}
	const std::vector<SubMatrix<T>>& get_MyNearFieldMats() const {return MyNearFieldMats;}
	const std::vector<LowRankMatrix<T>>& get_MyFarFieldMats() const {return MyFarFieldMats;}
	const std::vector<int>& get_MyDiagNearFieldMats() const {return MyDiagNearFieldMats;}
	const std::vector<int>& get_MyDiagFarFieldMats() const {return MyDiagFarFieldMats;}
	std::string get_name() const {return name;}

	// Infos
	const std::map<std::string,double>& get_stats() const { return Get_infos(name);}
	void add_stats(const std::string& keyname, const double& value){Add_info(name,keyname,value);}
	void print_stats();
	double compression() const; // 1- !!!
	friend double Frobenius_absolute_error<LowRankMatrix,T>(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A);

	// Mat vec prod
	void mvprod_global(const T* const in, T* const out) const;
	void mvprod_local(const T* const in, T* const out, T* const work) const;
	std::vector<T> operator*( const std::vector<T>& x) const;
	void mymvprod_local(const T* const in, T* const out) const;
	// Permutations
	void source_to_cluster_permutation(const T* const in, T* const out) const;
	void cluster_to_target_permutation(const T* const in, T* const out) const;

	// local to global
 	void local_to_global(const T* const in, T* const out) const;


};

// build
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	MPI_Comm_dup(comm0,&comm);
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4);
	if (name=="")
		name = "hmat "+NbrToStr(this->howMany());

	// Construction arbre des paquets
	double time = MPI_Wtime();
	cluster_tree_t = std::make_shared<Cluster_tree>(xt,rt,tabt,gt); // target
	cluster_tree_s = std::make_shared<Cluster_tree>(xs,rs,tabs,gs); // source

	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_s->get_root());
	if (B != NULL) Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xs,tabs);
	mytimes[3] = MPI_Wtime() - time;

	// Stats
	ComputeStats(mytimes);
}

// Full constructor
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), name(name0) {
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
}

// Constructor without rt and rs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), name(name0) {

	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, gt, xs, std::vector<double>(xs.size(),0), tabs, gs, comm0);
}

// Constructor without gt and gs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), name(name0) {
	this->build(mat, xt, rt, tabt, std::vector<double>(xt.size(),1), xs, rs, tabs, std::vector<double>(xs.size(),1), comm0);
}

// Constructor without tabt and tabs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), name(name0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs, comm0);
}

// Constructor without radius, mass and tab
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<R3>& xs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0), name(name0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, std::vector<double>(xt.size(),1), xs, std::vector<double>(xs.size(),0), tabs, std::vector<double>(xs.size(),1), comm0);
}

// Symetric build
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, MPI_Comm comm0){
	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabt.size() );

	MPI_Comm_dup(comm0,&comm);
	MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4), maxtime(4), meantime(4);
	if (name=="")
		name = "hmat "+NbrToStr(this->howMany());

	// Construction arbre des paquets
	double time = MPI_Wtime();
	cluster_tree_t = std::make_shared<Cluster_tree>(xt,rt,tabt,gt);
	cluster_tree_s = cluster_tree_t;
	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_t->get_root());
	if (B != NULL) Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xt,tabt);
	mytimes[3] = MPI_Wtime() - time;

	// Stats info
	ComputeStats(mytimes);

}

// Full symetric constructor
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), name(name0){

		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without rt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), name(name0){
		this->build(mat,xt,std::vector<double>(xt.size(),0),tabt,gt,comm0);
}


// Symetric constructor without tabt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), name(name0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without gt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), name(name0), comm(comm0){
		this->build(mat,xt,rt,tabt,std::vector<double>(xt.size(),1),comm0);
}

// Symetric constructor without rt, tabt and gt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), name(name0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,std::vector<double>(xt.size(),0),tabt,std::vector<double>(xt.size(),1),comm0);
}


// build with input cluster
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<int>& tabs, MPI_Comm comm0){

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	MPI_Comm_dup(comm0,&comm);
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4);
	if (name=="")
		name = "hmat "+NbrToStr(this->howMany());

	// Construction arbre des paquets
	double time = MPI_Wtime();

	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();


	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_s->get_root());
	if (B != NULL) Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xs,tabs);
	mytimes[3] = MPI_Wtime() - time;

	// Stats
	ComputeStats(mytimes);
}


// Full constructor with precomputed clusters
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,  const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const std::vector<int>& tabs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0), name(name0) {
	this->build(mat, xt, tabt, xs, tabs, comm0);
}

// Constructor without tabt and tabs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::shared_ptr<Cluster_tree>& s, const std::vector<R3>&xs, const int& reqrank0, const std::string& name0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0), name(name0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, tabt, xs, tabs, comm0);
}


// Full symetric constructor
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0), name(name0){

		this->build(mat,xt,tabt,xt,tabt,comm0);
}

// Symetric constructor without tabt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::shared_ptr<Cluster_tree>& t, const std::vector<R3>& xt, const int& reqrank0, const std::string& name0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0), name(name0){
	std::vector<int> tabt(xt.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,tabt,xt,tabt,comm0);
}


// Build block tree
// TODO recursivity -> stack for buildblocktree
template< template<typename> class LowRankMatrix, typename T >
Block* HMatrix<LowRankMatrix, T >::BuildBlockTree(const Cluster& t, const Cluster& s){
	Block* B = new Block(t,s);
	int bsize = t.get_size()*s.get_size();
	B->ComputeAdmissibility();
	if( B->IsAdmissible() && t.get_rank()>=0){
		Tasks.push_back(B);
		return NULL;
	}
	else if( s.IsLeaf() ){
				if( t.IsLeaf() ){
					return B;
				}
				else{
					Block* r1 = BuildBlockTree(t.get_son(0),s);
					Block* r2 = BuildBlockTree(t.get_son(1),s);
					if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL) && t.get_rank()>=0) {
						delete r1;
						delete r2;
						return B;
					}
					else {
						if (r1 != NULL) Tasks.push_back(r1);
						if (r2 != NULL) Tasks.push_back(r2);
						return NULL;
					}
				}
	}
	else{
		if( t.IsLeaf() ){
			Block* r3 = BuildBlockTree(t,s.get_son(0));
			Block* r4 = BuildBlockTree(t,s.get_son(1));
			if ((bsize <= maxblocksize) && (r3 != NULL) && (r4 != NULL)&& t.get_rank()>=0) {
				delete r3;
				delete r4;
				return B;
			}
			else {
				if (r3 != NULL) Tasks.push_back(r3);
				if (r4 != NULL) Tasks.push_back(r4);
				return NULL;
			}
		}
		else{

				// // Other kind of partition
				// Block* r1 = BuildBlockTree(t.get_son(0),s.get_son(0));
				// Block* r2 = BuildBlockTree(t.get_son(1),s.get_son(1));
				// Block* r3 = BuildBlockTree(t.get_son(1),s.get_son(0));
				// Block* r4 = BuildBlockTree(t.get_son(0),s.get_son(1));
				// if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL) && (r3 != NULL) && (r4 != NULL) && t.get_rank()>=0) {
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
				if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL)&& t.get_rank()>=0) {
					delete r1;
					delete r2;
					return B;
				}
				else {
					if (r1 != NULL) Tasks.push_back(r1);
					if (r2 != NULL) Tasks.push_back(r2);
					return NULL;
				}
			}
			else{
				Block* r3 = BuildBlockTree(t,s.get_son(0));
				Block* r4 = BuildBlockTree(t,s.get_son(1));
				if ((bsize <= maxblocksize) && (r3 != NULL) && (r4 != NULL)&& t.get_rank()>=0) {
					delete r3;
					delete r4;
					return B;
				}
				else {
					if (r3 != NULL) Tasks.push_back(r3);
					if (r4 != NULL) Tasks.push_back(r4);
					return NULL;
				}
			}
		}
	}
}

// Scatter tasks
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::ScatterTasks(){

	// std::cout << "Tasks : "<<Tasks.size()<<std::endl;
  for(int b=0; b<Tasks.size(); b++){
    	//if (b%sizeWorld == rankWorld)
    if ((*(Tasks[b])).tgt_().get_rank() == rankWorld){
    		MyBlocks.push_back(Tasks[b]);
		}
	}
	// std::cout << "rank : "<<rankWorld<<" "<<"Block : "<<MyBlocks.size() <<std::endl;
}

// Compute blocks recursively
// TODO recursivity -> stack for compute blocks
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix,T >::ComputeBlocks(const IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
  for(int b=0; b<MyBlocks.size(); b++) {
  	const Block& B = *(MyBlocks[b]);
 		const Cluster& t = B.tgt_();
		const Cluster& s = B.src_();
		if( B.IsAdmissible() ){
			// MyFarFieldMats.emplace_back(I,J,reqrank);
			// MyFarFieldMats.back().build(mat,t,s);
			AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,reqrank);
			if(MyFarFieldMats.back().rank_of()==-1){
				MyFarFieldMats.pop_back();
				if (s.get_rank()==rankWorld)
					MyDiagFarFieldMats.pop_back();
				if( s.IsLeaf() ){
					if( t.IsLeaf() ){
						// MyNearFieldMats.emplace_back(mat,I,J);
						AddNearFieldMat(mat,t,s);
					}
					else{
						bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs);
						bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs);
						if ((b1 != true) && (b2 != true))
							// MyNearFieldMats.emplace_back(mat,I,J);
							AddNearFieldMat(mat,t,s);
						else {
							if (b1 != true)
								// 	MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
								AddNearFieldMat(mat,t.get_son(0),s);
							if (b2 != true)
								// 	MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
								AddNearFieldMat(mat,t.get_son(1),s);
						}
					}
				}
				else{
					if( t.IsLeaf() ){
						bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs);
						bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs);
						if ((b3 != true) && (b4 != true))
							// MyNearFieldMats.emplace_back(mat,I,J);
							AddNearFieldMat(mat,t,s);
						else {
							if (b3 != true)
								// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
								AddNearFieldMat(mat,t,s.get_son(0));
							if (b4 != true)
								// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
								AddNearFieldMat(mat,t,s.get_son(1));
						}
					}
					else{
						if (t.get_size()>s.get_size()){
							bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs);
							bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs);
							if ((b1 != true) && (b2 != true))
								// MyNearFieldMats.emplace_back(mat,I,J);
								AddNearFieldMat(mat,t,s);
							else {
								if (b1 != true)
										// MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
										AddNearFieldMat(mat,t.get_son(0),s);
								if (b2 != true)
										// MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
										AddNearFieldMat(mat,t.get_son(1),s);
							}
						}
						else{
							bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs);
							bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs);
							if ((b3 != true) && (b4 != true))
								// MyNearFieldMats.emplace_back(mat,I,J);
								AddNearFieldMat(mat,t,s);
							else {
								if (b3 != true)
										// MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
										AddNearFieldMat(mat,t,s.get_son(0));
								if (b4 != true)
										// MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
										AddNearFieldMat(mat,t,s.get_son(1));
							}
						}
					}
				}
			}
		}
		else {
			// MyNearFieldMats.emplace_back(mat,I,J);
			AddNearFieldMat(mat,t,s);
		}
	}
}

template< template<typename> class LowRankMatrix, typename T >
bool HMatrix<LowRankMatrix,T >::UpdateBlocks(const IMatrix<T>& mat,const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
	int bsize = t.get_size()*s.get_size();
	Block B(t,s);
	B.ComputeAdmissibility();
	if( B.IsAdmissible() ){
		// MyFarFieldMats.emplace_back(I,J,reqrank);
		// MyFarFieldMats.back().build(mat,t,s);
		AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,reqrank);
		if(MyFarFieldMats.back().rank_of()!=-1){
			return true;
		}
		else {
			MyFarFieldMats.pop_back();
			if (s.get_rank()==rankWorld)
				MyDiagFarFieldMats.pop_back();
		}
	}
	if( s.IsLeaf() ){
		if( t.IsLeaf() ){
			return false;
		}
		else{
			bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs);
			bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs);
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
				return false;
			else {
				if (b1 != true)
					// MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
					AddNearFieldMat(mat,t.get_son(0),s);
				if (b2 != true)
					// MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
					AddNearFieldMat(mat,t.get_son(1),s);
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs);
			bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs);
			if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
				return false;
			else{
				if (b3 != true)
					// MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
					AddNearFieldMat(mat,t,s.get_son(0));
				if (b4 != true)
					// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
					AddNearFieldMat(mat,t,s.get_son(1));
				return true;
			}
		}
		else{

			if (t.get_size()>s.get_size()){
				bool b1 = UpdateBlocks(mat,t.get_son(0),s,xt,tabt,xs,tabs);
				bool b2 = UpdateBlocks(mat,t.get_son(1),s,xt,tabt,xs,tabs);
				if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
					return false;
				else {
					if (b1 != true)
						// 	MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
						AddNearFieldMat(mat,t.get_son(0),s);
					if (b2 != true)
						// 	MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
						AddNearFieldMat(mat,t.get_son(1),s);
					return true;
				}
			}
			else{
				bool b3 = UpdateBlocks(mat,t,s.get_son(0),xt,tabt,xs,tabs);
				bool b4 = UpdateBlocks(mat,t,s.get_son(1),xt,tabt,xs,tabs);
				if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
					return false;
				else{
					if (b3 != true)
						// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
						AddNearFieldMat(mat,t,s.get_son(0));
					if (b4 != true)
						// 	MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
						AddNearFieldMat(mat,t,s.get_son(1));
					return true;
				}
			}
		}
	}
}

// Build a dense block
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::AddNearFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s){
	MyNearFieldMats.emplace_back(mat, std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset());
	if (s.get_rank()==rankWorld){
        MyDiagNearFieldMats.push_back(MyNearFieldMats.size()-1);
				// std::cout <<"pouet"<<(*MyDiagNearFieldMats.back()).get_ir() << std::endl;
	}
}

// Build a low rank block
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::AddFarFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, const int& reqrank){
	MyFarFieldMats.emplace_back(std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset(),reqrank);
	MyFarFieldMats.back().build(mat,t,xt,tabt,s,xs,tabs);
	if (s.get_rank()==rankWorld){
        MyDiagFarFieldMats.push_back(MyFarFieldMats.size()-1);
				// std::cout <<(*MyDiagFarFieldMats.back()).rank_of() << std::endl;
	}
}

// Compute stats
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::ComputeStats(const std::vector<double>& mytime){
	// 0 : cluster tree ; 1 : block tree ; 2 : scatter tree ; 3 : compute blocks
	std::vector<double> maxtime(4), meantime(4);
	// 0 : dense mat ; 1 : lr mat ; 2 : rank
	std::vector<int> maxstats(3,0),minstats(3,std::max(nc,nr));
	std::vector<double> meanstats(3,0);
	// stats
	for (int i=0;i<MyNearFieldMats.size();i++){
		int size = MyNearFieldMats[i].nb_rows()*MyNearFieldMats[i].nb_cols();
		maxstats[0] = std::max(maxstats[0],size);
		minstats[0] = std::min(minstats[0],size);
		meanstats[0] += size;
	}
	for (int i=0;i<MyFarFieldMats.size();i++){
		int size = MyFarFieldMats[i].nb_rows()*MyFarFieldMats[i].nb_cols();
		int rank = MyFarFieldMats[i].rank_of();
		maxstats[1] = std::max(maxstats[1],size);
		minstats[1] = std::min(minstats[1],size);
		meanstats[1] += size;
		maxstats[2] = std::max(maxstats[2],rank);
		minstats[2] = std::min(minstats[2],rank);
		meanstats[2] += rank;
	}

	if (rankWorld==0){
		MPI_Reduce(MPI_IN_PLACE, &(maxstats[0]), 3, MPI_INT, MPI_MAX, 0,comm);
		MPI_Reduce(MPI_IN_PLACE, &(minstats[0]), 3, MPI_INT, MPI_MIN, 0,comm);
		MPI_Reduce(MPI_IN_PLACE, &(meanstats[0]),3, MPI_DOUBLE, MPI_SUM, 0,comm);
	}
	else{
		MPI_Reduce(&(maxstats[0]), &(maxstats[0]), 3, MPI_INT, MPI_MAX, 0,comm);
		MPI_Reduce(&(minstats[0]), &(minstats[0]), 3, MPI_INT, MPI_MIN, 0,comm);
		MPI_Reduce(&(meanstats[0]), &(meanstats[0]),3, MPI_DOUBLE, MPI_SUM, 0,comm);
	}

	int nlrmat = this->get_nlrmat();
	int ndmat = this->get_ndmat();
	meanstats[0] = (ndmat  == 0 ? 0 : meanstats[0]/ndmat);
	meanstats[1] = (nlrmat == 0 ? 0 : meanstats[1]/nlrmat);
	meanstats[2] = (nlrmat == 0 ? 0 : meanstats[2]/nlrmat);
	minstats[0] = (ndmat  == 0 ? 0 : minstats[0]);
	minstats[1] = (nlrmat  == 0 ? 0 : minstats[1]);
	minstats[2] = (nlrmat  == 0 ? 0 : minstats[2]);

	// timing
	MPI_Reduce(&(mytime[0]), &(maxtime[0]), 4, MPI_DOUBLE, MPI_MAX, 0,comm);
	MPI_Reduce(&(mytime[0]), &(meantime[0]), 4, MPI_DOUBLE, MPI_SUM, 0,comm);

	meantime /= sizeWorld;

	// Times
	Add_info(name,"Cluster tree (mean)",meantime[0]);
	Add_info(name,"Cluster tree  (max)",maxtime[0]);
	Add_info(name,"Block tree   (mean)",meantime[1]);
	Add_info(name,"Block tree    (max)",maxtime[1]);
	Add_info(name,"Scatter tree (mean)",meantime[2]);
	Add_info(name,"Scatter tree  (max)",maxtime[2]);
	Add_info(name,"Blocks       (mean)",meantime[3]);
	Add_info(name,"Blocks        (max)",maxtime[3]);

	// Size
	Add_info(name,"Source size",this->nc);
	Add_info(name,"Target size", this->nr);
	Add_info(name,"Dense block size (max)", maxstats[0]);
	Add_info(name,"Dense block size (mean)", meanstats[0]);
	Add_info(name,"Dense block size (min)", minstats[0]);
	Add_info(name,"Low rank block size (max)", maxstats[1]);
	Add_info(name,"Low rank  block size (mean)", meanstats[1]);
	Add_info(name,"Low rank  block size (min)", minstats[1]);

	Add_info(name,"Rank (max)", maxstats[2]);
	Add_info(name,"Rank (mean)", meanstats[2]);
	Add_info(name,"Rank (min)", minstats[2]);
	Add_info(name,"Number of lrmat", nlrmat);
	Add_info(name,"Number of dmat", ndmat);
	Add_info(name,"Compression", this->compression());

	Add_info(name,"nbr mat vec prod", 0);
	Add_info(name,"total time mat vec prod", 0);

}



template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mymvprod_local(const T* const in, T* const out) const{

	std::fill(out,out+local_size,0);

	// Contribution champ lointain
	for(int b=0; b<MyFarFieldMats.size(); b++){
		const LowRankMatrix<T>&  M  = MyFarFieldMats[b];
		int offset_i     = M.get_offset_i();
		int offset_j     = M.get_offset_j();

		M.add_mvprod(in+offset_j,out+offset_i-local_offset);

	}
	// Contribution champ proche
	for(int b=0; b<MyNearFieldMats.size(); b++){
		const SubMatrix<T>&  M  = MyNearFieldMats[b];
		int offset_i     = M.get_offset_i();
		int offset_j     = M.get_offset_j();

		M.add_mvprod(in+offset_j,out+offset_i-local_offset);
	}

}


template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::local_to_global(const T* const in, T* const out) const{
	// Allgather
	std::vector<int> recvcounts(sizeWorld);
	std::vector<int>  displs(sizeWorld);

	displs[0] = 0;

	for (int i=0; i<sizeWorld; i++) {
		recvcounts[i] = cluster_tree_t->get_masteroffset(i).second;
		if (i > 0)
			displs[i] = displs[i-1] + recvcounts[i-1];
	}

	MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);


}



template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_local(const T* const in, T* const out, T* const work) const{
	double time = MPI_Wtime();
	this->local_to_global(in, work);
	this->mymvprod_local(work,out);
	Add_value(name, "nbr mat vec prod",1 );
	Add_value(name,"total time mat vec prod", MPI_Wtime()-time);
}


template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_global(const T* const in, T* const out) const{

	double time = MPI_Wtime();
	std::vector<T> in_perm(nc);
	std::vector<T> out_not_perm(nr);

	// Permutation
	cluster_tree_s->global_to_cluster(in,in_perm.data());

	// mvprod local
	mymvprod_local(in_perm.data(),out_not_perm.data()+local_offset);


	// Allgather
	std::vector<T> snd(local_size);
	std::vector<int> recvcounts(sizeWorld);
	std::vector<int>  displs(sizeWorld);

	displs[0] = 0;

	for (int i=0; i<sizeWorld; i++) {
		recvcounts[i] = cluster_tree_t->get_masteroffset(i).second;
		if (i > 0)
			displs[i] = displs[i-1] + recvcounts[i-1];
	}

	// std::copy_n(snd.data(),size,out+offset);
	MPI_Allgatherv(MPI_IN_PLACE, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out_not_perm.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<T>::mpi_type(), comm);
	// MPI_Allgatherv(snd.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts.front()), &(displs.front()), wrapper_mpi<T>::mpi_type(), comm);

	// Permutation
	cluster_tree_t->cluster_to_global(out_not_perm.data(),out);

	Add_value(name, "nbr mat vec prod",1 );
	Add_value(name,"total time mat vec prod", MPI_Wtime()-time);
}

template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::source_to_cluster_permutation(const T* const in, T* const out) const {
	cluster_tree_s->global_to_cluster(in,out);
}

template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::cluster_to_target_permutation(const T* const in, T* const out) const{
	cluster_tree_t->cluster_to_global(in,out);
}




template< template<typename> class LowRankMatrix, typename T>
std::vector<T> HMatrix<LowRankMatrix,T >::operator*(const std::vector<T>& x) const{
	assert(x.size()==nc);
	std::vector<T> result(nr,0);
	mvprod_global(x.data(),result.data());
	return result;
}


template< template<typename> class LowRankMatrix, typename T >
double HMatrix<LowRankMatrix,T >::compression() const{

	double mycomp = 0.;
	double size = ((long int)this->nr)*this->nc;
	double nr_b ,nc_b,rank;

	for(int j=0; j<MyFarFieldMats.size(); j++){
		nr_b  = MyFarFieldMats[j].nb_rows();
		nc_b   = MyFarFieldMats[j].nb_cols();
		rank = MyFarFieldMats[j].rank_of();
		mycomp += rank*(nr_b + nc_b)/size;
	}

	for(int j=0; j<MyNearFieldMats.size(); j++){
		nr_b   = MyNearFieldMats[j].nb_rows();
		nc_b   = MyNearFieldMats[j].nb_cols();
		mycomp += nr_b*nc_b/size;
	}

	double comp = 0;
	MPI_Allreduce(&mycomp, &comp, 1, MPI_DOUBLE, MPI_SUM, comm);

	return 1-comp;
}

template<template<typename> class LowRankMatrix,typename T >
void HMatrix<LowRankMatrix,T >::print_stats(){
	if (rankWorld==0){
		Print_infos(name);
	}
}

template< template<typename> class LowRankMatrix, typename T >
double Frobenius_absolute_error(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A){
	double myerr = 0;
	for(int j=0; j<B.MyFarFieldMats.size(); j++){
		double test = Frobenius_absolute_error(B.MyFarFieldMats[j], A);
		myerr += std::pow(test,2);

	}

	double err = 0;
	MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

	return std::sqrt(err);
}

} //namespace
#endif
