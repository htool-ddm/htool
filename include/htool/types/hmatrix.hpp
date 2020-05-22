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
#include "multihmatrix.hpp"
#include "../misc/parametres.hpp"
#include "../clustering/cluster.hpp"
#include "../blocks/blocks.hpp"
#include "../wrappers/wrapper_mpi.hpp"


namespace htool {


//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//
// Friend functions --- forward declaration
template<typename T, template<typename,typename> class MultiLowRankMatrix, typename ClusterImpl >
class MultiHMatrix;
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
class HMatrix;

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
double Frobenius_absolute_error(const HMatrix<T, LowRankMatrix, ClusterImpl>& B, const IMatrix<T>& A);

// Class
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
class HMatrix: public Parametres{

private:
	// Data members
	int nr;
	int nc;
	int reqrank;
	int local_size;
	int local_offset;

	bool symmetric;

	std::vector<Block<ClusterImpl>*>		   Tasks;
	std::vector<Block<ClusterImpl>*>		   MyBlocks;

	std::vector<LowRankMatrix<T,ClusterImpl>* > MyFarFieldMats;
	std::vector<SubMatrix<T>* >     MyNearFieldMats;
	std::vector<LowRankMatrix<T,ClusterImpl>*> MyDiagFarFieldMats;
	std::vector<SubMatrix<T>*> MyDiagNearFieldMats;
    std::vector<LowRankMatrix<T,ClusterImpl>*> MyStrictlyDiagFarFieldMats;
	std::vector<SubMatrix<T>*> MyStrictlyDiagNearFieldMats;


	std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_s;
	std::shared_ptr<Cluster<ClusterImpl>> cluster_tree_t;

	mutable std::map<std::string, std::string> infos;

	MPI_Comm comm;
	int rankWorld,sizeWorld;


	// Internal methods
	void ScatterTasks();
	Block<ClusterImpl>* BuildBlockTree(const Cluster<ClusterImpl>&, const Cluster<ClusterImpl>&);
	Block<ClusterImpl>* BuildSymBlockTree(const Cluster<ClusterImpl>&, const Cluster<ClusterImpl>&);
	void ComputeBlocks(IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	void ComputeSymBlocks(IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	bool UpdateBlocks(IMatrix<T>&mat ,const Cluster<ClusterImpl>&, const Cluster<ClusterImpl>&, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<SubMatrix<T>*>&, std::vector<LowRankMatrix<T,ClusterImpl>*>&);
	bool UpdateSymBlocks(IMatrix<T>&mat ,const Cluster<ClusterImpl>&, const Cluster<ClusterImpl>&, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<SubMatrix<T>*>&, std::vector<LowRankMatrix<T,ClusterImpl>*>&);
	void AddNearFieldMat(IMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, std::vector<SubMatrix<T>*>&);
	void AddFarFieldMat(IMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<LowRankMatrix<T,ClusterImpl>*>&, const int& reqrank=-1);
	void ComputeInfos(const std::vector<double>& mytimes);

	// Friends
	template<typename U,template<typename,typename> class MultiLowRankMatrix, typename ClusterImplU > friend class MultiHMatrix; 


	// Special constructor for hand-made build (for MultiHMatrix for example)
	HMatrix(int nr0, int nc0,const std::shared_ptr<Cluster<ClusterImpl>>& cluster_tree_t0, const std::shared_ptr<Cluster<ClusterImpl>>& cluster_tree_s0,bool symmetry0=false): nr(nr0), nc(nc0), cluster_tree_t(cluster_tree_t0), cluster_tree_s(cluster_tree_s0), symmetric(symmetry0){};


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
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Symetric constructor without radius
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without mass
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without radius, tab and mass
	HMatrix(IMatrix<T>&, const std::vector<R3>& xt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Build with precomputed clusters
	void build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<int>& tabs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor with precomputed clusters
	HMatrix(IMatrix<T>& mat,  const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, const std::vector<int>& tabt,  const std::shared_ptr<Cluster<ClusterImpl>>& s, const std::vector<R3>&xs, const std::vector<int>& tabs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without tab and with precomputed clusters
	HMatrix(IMatrix<T>&,  const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt,  const std::shared_ptr<Cluster<ClusterImpl>>& s, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Symetric build with precomputed cluster
	void build(IMatrix<T>& mat,  const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Full symetric constructor with precomputed cluster
	HMatrix(IMatrix<T>& mat,  const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab and with precomputed cluster
	HMatrix(IMatrix<T>&,  const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, bool Symmetry=false, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

  // Destructor
	~HMatrix() {
		for (int i=0; i<Tasks.size(); i++)
			delete Tasks[i];
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

    const Cluster<ClusterImpl>& get_cluster_tree_t() const{return *(cluster_tree_t.get());}
    const Cluster<ClusterImpl>& get_cluster_tree_s() const{return *(cluster_tree_s.get());}
	std::vector<std::pair<int,int>> get_MasterOffset_t() const {return cluster_tree_t->get_masteroffset();}
	std::vector<std::pair<int,int>> get_MasterOffset_s() const {return cluster_tree_s->get_masteroffset();}
    std::pair<int,int> get_MasterOffset_t(int i) const {return cluster_tree_t->get_masteroffset(i);}
    std::pair<int,int> get_MasterOffset_s(int i) const {return cluster_tree_s->get_masteroffset(i);}
	const std::vector<int>& get_permt() const {return cluster_tree_t->get_perm();}
	const std::vector<int>& get_perms() const {return cluster_tree_s->get_perm();}
    int get_permt(int i) const {return cluster_tree_t->get_perm(i);}
	int get_perms(int i) const {return cluster_tree_s->get_perm(i);}
	const std::vector<SubMatrix<T>*>& get_MyNearFieldMats() const {return MyNearFieldMats;}
	const std::vector<LowRankMatrix<T,ClusterImpl>*>& get_MyFarFieldMats() const {return MyFarFieldMats;}
	const std::vector<SubMatrix<T>*>& get_MyDiagNearFieldMats() const {return MyDiagNearFieldMats;}
	const std::vector<LowRankMatrix<T,ClusterImpl>*>& get_MyDiagFarFieldMats() const {return MyDiagFarFieldMats;}
        const std::vector<SubMatrix<T>*>& get_MyStrictlyDiagNearFieldMats() const {return MyStrictlyDiagNearFieldMats;}
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& get_MyStrictlyDiagFarFieldMats() const {return MyStrictlyDiagFarFieldMats;}

	// Infos
	const std::map<std::string, std::string>& get_infos() const {return infos;}
  std::string get_infos (const std::string& key) const { return infos[key];}
	void add_info(const std::string& keyname, const std::string& value) const {infos[keyname]=value;}
	void print_infos() const;
	void save_infos(const std::string& outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string& sep = " = ") const;
	void save_plot(const std::string& outputname) const;
	double compression() const; // 1- !!!
	friend double Frobenius_absolute_error<T,LowRankMatrix,ClusterImpl>(const HMatrix<T, LowRankMatrix, ClusterImpl>& B, const IMatrix<T>& A);

	// Mat vec prod
	void mvprod_global(const T* const in, T* const out,const int& mu=1) const;
	void mvprod_local(const T* const in, T* const out, T* const work, const int& mu) const;
	void mymvprod_local(const T* const in, T* const out, const int& mu) const;
    void mvprod_subrhs(const T* const in, T* const out, const int& mu, const int& offset, const int& size, const int& local_max_size_j) const;
	std::vector<T> operator*( const std::vector<T>& x) const;
	Matrix<T> operator*( const Matrix<T>& x) const;

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
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

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

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block<ClusterImpl>* B=nullptr;
	B = BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_s->get_root());
	if (B !=nullptr) Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat,xt,tabt,xs,tabs);
	mytimes[3] = MPI_Wtime() - time;

	// Infos
	ComputeInfos(mytimes);
}

// Full constructor
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
}

// Constructor without rt and rs
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {

	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, gt, xs, std::vector<double>(xs.size(),0), tabs, gs, comm0);
}

// Constructor without gt and gs
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, std::vector<double>(xt.size(),1), xs, rs, tabs, std::vector<double>(xs.size(),1), comm0);
}

// Constructor without tabt and tabs
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs, comm0);
}

// Constructor without radius, mass and tab
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<R3>& xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_s(nullptr), cluster_tree_t(nullptr), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, std::vector<double>(xt.size(),1), xs, std::vector<double>(xs.size(),0), tabs, std::vector<double>(xs.size(),1), comm0);
}

// Symetric build
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::build(IMatrix<T>& mat,const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, MPI_Comm comm0){
	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabt.size() );

	MPI_Comm_dup(comm0,&comm);
	MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	cluster_tree_t = std::make_shared<ClusterImpl>();
	cluster_tree_s = cluster_tree_t;
	cluster_tree_t->build(xt,rt,tabt,gt,-1,comm);
	local_size   = cluster_tree_t->get_local_size();
	local_offset = cluster_tree_t->get_local_offset();

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block<ClusterImpl>* B=nullptr;
	if (!symmetric){
		B = BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_t->get_root());
	}
	else{
		B = BuildSymBlockTree(cluster_tree_t->get_root(),cluster_tree_t->get_root());
	}
	
	if (B !=nullptr) Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	if (!symmetric){
		ComputeBlocks(mat,xt,tabt,xt,tabt);
	}
	else{
		ComputeSymBlocks(mat,xt,tabt,xt,tabt);
	}
	mytimes[3] = MPI_Wtime() - time;

	// Infos
	ComputeInfos(mytimes);

}

// Full symetric constructor
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0), cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){

		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without rt
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0), cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){
		this->build(mat,xt,std::vector<double>(xt.size(),0),tabt,gt,comm0);
}


// Symetric constructor without tabt
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without gt
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0), comm(comm0){
		this->build(mat,xt,rt,tabt,std::vector<double>(xt.size(),1),comm0);
}

// Symetric constructor without rt, tabt and gt
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat,
		 const std::vector<R3>& xt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0),cluster_tree_s(nullptr),cluster_tree_t(nullptr),reqrank(reqrank0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,std::vector<double>(xt.size(),0),tabt,std::vector<double>(xt.size(),1),comm0);
}


// build with input cluster
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::build(IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<int>& tabs, MPI_Comm comm0){

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
	Block<ClusterImpl>* B=nullptr;
	if (!symmetric){
		B = BuildBlockTree(cluster_tree_t->get_root(),cluster_tree_s->get_root());
	}
	else{
		B = BuildSymBlockTree(cluster_tree_t->get_root(),cluster_tree_s->get_root());
	}
	
	if (B !=nullptr) Tasks.push_back(B);
	mytimes[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	mytimes[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	if (!symmetric){
		ComputeBlocks(mat,xt,tabt,xs,tabs);
	}
	else{
		ComputeSymBlocks(mat,xt,tabt,xs,tabs);
	}
	mytimes[3] = MPI_Wtime() - time;

	// Infos
	ComputeInfos(mytimes);
}


// Full constructor with precomputed clusters
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat,  const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::shared_ptr<Cluster<ClusterImpl>>& s, const std::vector<R3>&xs, const std::vector<int>& tabs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0) {
	this->build(mat, xt, tabt, xs, tabs, comm0);
}

// Constructor without tabt and tabs
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, const std::shared_ptr<Cluster<ClusterImpl>>& s, const std::vector<R3>&xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(false), cluster_tree_t(t), cluster_tree_s(s), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, tabt, xs, tabs, comm0);
}


// Full symetric constructor
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, const std::vector<int>& tabt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0){

		this->build(mat,xt,tabt,xt,tabt,comm0);
}

// Symetric constructor without tabt
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
HMatrix<T, LowRankMatrix, ClusterImpl>::HMatrix(IMatrix<T>& mat, const std::shared_ptr<Cluster<ClusterImpl>>& t, const std::vector<R3>& xt, bool symmetric0, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()), symmetric(symmetric0), cluster_tree_t(t), cluster_tree_s(t), reqrank(reqrank0){
	std::vector<int> tabt(xt.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,tabt,xt,tabt,comm0);
}



// Build block tree
// TODO: recursivity -> stack for buildblocktree
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
Block<ClusterImpl>* HMatrix<T, LowRankMatrix, ClusterImpl>::BuildBlockTree(const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s){
	
	Block<ClusterImpl>* B = new Block<ClusterImpl>(t,s);
	int bsize = t.get_size()*s.get_size();
	B->ComputeAdmissibility();
	if( B->IsAdmissible() && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth() && (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )){
		Tasks.push_back(B);
		return nullptr;
	}
	else if( s.IsLeaf() ){
		if( t.IsLeaf() ){
			return B;
		}
		else{
			std::vector<Block<ClusterImpl>*> Blocks(t.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				Blocks[p] = BuildBlockTree(t.get_son(p),s);
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth() && (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
				for (auto block : Blocks){
					delete block;
				} 
				return B;
			}
			else{
				for (auto block : Blocks){
					if (block !=nullptr) Tasks.push_back(block);
				}
				delete B; 
				return nullptr;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			std::vector<Block<ClusterImpl>*> Blocks(s.get_nb_sons());
			for (int p=0; p <s.get_nb_sons();p++){
				Blocks[p] = BuildBlockTree(t,s.get_son(p));
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&& (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
				for (auto block : Blocks){
					delete block;
				} 
				return B;
			}
			else{
				for (auto block : Blocks){
					if (block !=nullptr) Tasks.push_back(block);
				} 
				delete B;
				return nullptr;
			}
		}
		else{
			if (t.get_size()>s.get_size()){
				std::vector<Block<ClusterImpl>*> Blocks(t.get_nb_sons());
				for (int p=0; p <t.get_nb_sons();p++){
					Blocks[p] = BuildBlockTree(t.get_son(p),s);
				}
				if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&& (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
					for (auto block : Blocks){
						delete block;
					} 
					return B;
				}
				else{
					for (auto block : Blocks){
						if (block !=nullptr) Tasks.push_back(block);
					} 
					delete B;
					return nullptr;
				}
			}
			else{
				std::vector<Block<ClusterImpl>*> Blocks(s.get_nb_sons());
				for (int p=0; p <s.get_nb_sons();p++){
					Blocks[p] = BuildBlockTree(t,s.get_son(p));
				}
				if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&& (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
					for (auto block : Blocks){
						delete block;
					} 
					return B;
				}
				else{
					for (auto block : Blocks){
						if (block !=nullptr) Tasks.push_back(block);
					} 
					delete B;
					return nullptr;
				}
			}
		}
	}
}

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
Block<ClusterImpl>* HMatrix<T, LowRankMatrix, ClusterImpl>::BuildSymBlockTree(const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s){
	
	Block<ClusterImpl>* B = new Block<ClusterImpl>(t,s);
	int bsize = t.get_size()*s.get_size();
	B->ComputeAdmissibility();
	if( B->IsAdmissible() && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth() && (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )){
		Tasks.push_back(B);
		return nullptr;
	}
	else if( s.IsLeaf() ){
		if( t.IsLeaf() ){
			return B;
		}
		else{
			std::vector<Block<ClusterImpl>*> Blocks(t.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				Blocks[p] = BuildSymBlockTree(t.get_son(p),s);
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth() && (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
				for (auto block : Blocks){
					delete block;
				} 
				return B;
			}
			else{
				for (auto block : Blocks){
					if (block !=nullptr) Tasks.push_back(block);
				}
				delete B; 
				return nullptr;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			std::vector<Block<ClusterImpl>*> Blocks(s.get_nb_sons());
			for (int p=0; p <s.get_nb_sons();p++){
				Blocks[p] = BuildSymBlockTree(t,s.get_son(p));
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&& (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
				for (auto block : Blocks){
					delete block;
				} 
				return B;
			}
			else{
				for (auto block : Blocks){
					if (block !=nullptr) Tasks.push_back(block);
				} 
				delete B;
				return nullptr;
			}
		}
		else{
			std::vector<Block<ClusterImpl>*> Blocks(t.get_nb_sons()*s.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				for (int l=0; l <s.get_nb_sons();l++){
					Blocks[p+l*t.get_nb_sons()] = BuildSymBlockTree(t.get_son(p),s.get_son(l));
				}
			}
			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](Block<ClusterImpl>* block){return block!=nullptr;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&& (!symmetric || (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
				for (auto block : Blocks){
					delete block;
				} 
				return B;
			}
			else{
				for (auto block : Blocks){
					if (block !=nullptr) Tasks.push_back(block);
				} 
				delete B;
				return nullptr;
			}
		}
	}
}

// Scatter tasks
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::ScatterTasks(){


  	for(int b=0; b<Tasks.size(); b++){
    	if ((*(Tasks[b])).tgt_().get_rank() == cluster_tree_t->get_local_cluster().get_rank()){
			if (symmetric)
			{
				if (((*(Tasks[b])).src_().get_offset()<=(*(Tasks[b])).tgt_().get_offset() || (*(Tasks[b])).src_().get_offset()>=local_offset+local_size)){
    				MyBlocks.push_back(Tasks[b]);
				}
			}
			else{
				MyBlocks.push_back(Tasks[b]);
			}
		}
	}
    std::sort(MyBlocks.begin(),MyBlocks.end(),comp_block());

}

// Compute blocks recursively
// TODO: recursivity -> stack for compute blocks
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::ComputeBlocks(IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
    #if _OPENMP
    #pragma omp parallel
    #endif
    {
        // IMatrix<T> mat = mat;
        std::vector<SubMatrix<T>*>     MyNearFieldMats_local;
        std::vector<LowRankMatrix<T,ClusterImpl>*> MyFarFieldMats_local;
        // int tid = omp_get_thread_num();
        // std::cout<<"Hello World from thread = "+ NbrToStr(tid)<<std::endl;
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
        for(int b=0; b<MyBlocks.size(); b++) {
            const Block<ClusterImpl>& B = *(MyBlocks[b]);
        	const Cluster<ClusterImpl>& t = B.tgt_();
            const Cluster<ClusterImpl>& s = B.src_();
			int bsize = t.get_size()*s.get_size();
            if( B.IsAdmissible() ){
        	    AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,MyFarFieldMats_local,reqrank);
            	if(MyFarFieldMats_local.back()->rank_of()==-1){
                    delete MyFarFieldMats_local.back();
            		MyFarFieldMats_local.pop_back();

					// AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            		if( s.IsLeaf() ){
            			if( t.IsLeaf() ){
            				AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            			}
            			else{
							std::vector<bool> Blocks(t.get_nb_sons());
							for (int p=0; p <t.get_nb_sons();p++){
								Blocks[p] = UpdateBlocks(mat,t.get_son(p),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
							}

							if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
								AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
							}
							else{
								for (int p=0;p<Blocks.size();p++){
									if (Blocks[p] !=true) AddNearFieldMat(mat,t.get_son(p),s,MyNearFieldMats_local);
								} 
							}
            			}
            		}
            		else{
            			if( t.IsLeaf() ){
							std::vector<bool> Blocks(s.get_nb_sons());
							for (int p=0; p <s.get_nb_sons();p++){
								Blocks[p] = UpdateBlocks(mat,t,s.get_son(p),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
							}

							if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
								AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
							}
							else{
								for (int p=0;p<Blocks.size();p++){
									if (Blocks[p] !=true) AddNearFieldMat(mat,t,s.get_son(p),MyNearFieldMats_local);
								} 
							}
            			}
            			else{
            				if (t.get_size()>s.get_size()){
            					std::vector<bool> Blocks(t.get_nb_sons());
								for (int p=0; p <t.get_nb_sons();p++){
									Blocks[p] = UpdateBlocks(mat,t.get_son(p),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
								}

								if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
									AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
								}
								else{
									for (int p=0;p<Blocks.size();p++){
										if (Blocks[p] !=true) AddNearFieldMat(mat,t.get_son(p),s,MyNearFieldMats_local);
									} 
								}
            				}
            				else{
            					std::vector<bool> Blocks(s.get_nb_sons());
								for (int p=0; p <s.get_nb_sons();p++){
									Blocks[p] = UpdateBlocks(mat,t,s.get_son(p),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
								}

								if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
									AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
								}
								else{
									for (int p=0;p<Blocks.size();p++){
										if (Blocks[p] !=true) AddNearFieldMat(mat,t,s.get_son(p),MyNearFieldMats_local);
									} 
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::ComputeSymBlocks(IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs){
    #if _OPENMP
    #pragma omp parallel
    #endif
    {
        // IMatrix<T> mat = mat;
        std::vector<SubMatrix<T>*>     MyNearFieldMats_local;
        std::vector<LowRankMatrix<T,ClusterImpl>*> MyFarFieldMats_local;
        // int tid = omp_get_thread_num();
        // std::cout<<"Hello World from thread = "+ NbrToStr(tid)<<std::endl;
        #if _OPENMP
        #pragma omp for schedule(guided)
        #endif
        for(int b=0; b<MyBlocks.size(); b++) {
            const Block<ClusterImpl>& B = *(MyBlocks[b]);
        	const Cluster<ClusterImpl>& t = B.tgt_();
            const Cluster<ClusterImpl>& s = B.src_();
			int bsize = t.get_size()*s.get_size();
            if( B.IsAdmissible() ){
        	    AddFarFieldMat(mat,t,s,xt,tabt,xs,tabs,MyFarFieldMats_local,reqrank);
            	if(MyFarFieldMats_local.back()->rank_of()==-1){
                    delete MyFarFieldMats_local.back();
            		MyFarFieldMats_local.pop_back();

					// AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            		if( s.IsLeaf() ){
            			if( t.IsLeaf() ){
            				AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
            			}
            			else{
							std::vector<bool> Blocks(t.get_nb_sons());
							for (int p=0; p <t.get_nb_sons();p++){
								Blocks[p] = UpdateBlocks(mat,t.get_son(p),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
							}

							if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
								AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
							}
							else{
								for (int p=0;p<Blocks.size();p++){
									if (Blocks[p] !=true) AddNearFieldMat(mat,t.get_son(p),s,MyNearFieldMats_local);
								} 
							}
            			}
            		}
            		else{
            			if( t.IsLeaf() ){
							std::vector<bool> Blocks(s.get_nb_sons());
							for (int p=0; p <s.get_nb_sons();p++){
								Blocks[p] = UpdateBlocks(mat,t,s.get_son(p),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
							}

							if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
								AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
							}
							else{
								for (int p=0;p<Blocks.size();p++){
									if (Blocks[p] !=true) AddNearFieldMat(mat,t,s.get_son(p),MyNearFieldMats_local);
								} 
							}
            			}
            			else{
							std::vector<bool> Blocks(t.get_nb_sons()*s.get_nb_sons());
							for (int p=0; p <t.get_nb_sons();p++){
								for (int l=0; l <s.get_nb_sons();l++){
									Blocks[p+l*t.get_nb_sons()] =UpdateBlocks(mat,t.get_son(p),s.get_son(l),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
								}
							}
							if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
									AddNearFieldMat(mat,t,s,MyNearFieldMats_local);
							}
							else{
								for (int p=0; p <t.get_nb_sons();p++){
									for (int l=0; l <s.get_nb_sons();l++){
										if (Blocks[p+l*t.get_nb_sons()] !=true) AddNearFieldMat(mat,t.get_son(p),s.get_son(l),MyNearFieldMats_local);
									}
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
bool HMatrix<T, LowRankMatrix, ClusterImpl>::UpdateBlocks(IMatrix<T>& mat,const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<SubMatrix<T>*>& MyNearFieldMats_local, std::vector<LowRankMatrix<T,ClusterImpl>*>& MyFarFieldMats_local){
	int bsize = t.get_size()*s.get_size();
	Block<ClusterImpl> B(t,s);
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
			std::vector<bool> Blocks(t.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				Blocks[p] = UpdateBlocks(mat,t.get_son(p),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
				return false;
			}
			else{
				for (int p=0;p<Blocks.size();p++){
					if (Blocks[p] !=true) AddNearFieldMat(mat,t.get_son(p),s,MyNearFieldMats_local);
				} 
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			std::vector<bool> Blocks(t.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				Blocks[p] = UpdateBlocks(mat,t,s.get_son(p),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
				return false;
			}
			else{
				for (int p=0;p<Blocks.size();p++){
					if (Blocks[p] !=true) AddNearFieldMat(mat,t,s.get_son(p),MyNearFieldMats_local);
				} 
				return true;
			}
		}
		else{

			if (t.get_size()>s.get_size()){
				std::vector<bool> Blocks(t.get_nb_sons());
				for (int p=0; p <t.get_nb_sons();p++){
					Blocks[p] = UpdateBlocks(mat,t.get_son(p),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				}

				if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
					return false;
				}
				else{
					for (int p=0;p<Blocks.size();p++){
						if (Blocks[p] !=true) AddNearFieldMat(mat,t.get_son(p),s,MyNearFieldMats_local);
					} 
					return true;
				}
			}
			else{
				std::vector<bool> Blocks(t.get_nb_sons());
				for (int p=0; p <t.get_nb_sons();p++){
					Blocks[p] = UpdateBlocks(mat,t,s.get_son(p),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				}

				if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
					return false;
				}
				else{
					for (int p=0;p<Blocks.size();p++){
						if (Blocks[p] !=true) AddNearFieldMat(mat,t,s.get_son(p),MyNearFieldMats_local);
					} 
					return true;
				}
			}
		}
	}
}

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
bool HMatrix<T, LowRankMatrix, ClusterImpl>::UpdateSymBlocks(IMatrix<T>& mat,const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<SubMatrix<T>*>& MyNearFieldMats_local, std::vector<LowRankMatrix<T,ClusterImpl>*>& MyFarFieldMats_local){
	int bsize = t.get_size()*s.get_size();
	Block<ClusterImpl> B(t,s);
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
			std::vector<bool> Blocks(t.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				Blocks[p] = UpdateBlocks(mat,t.get_son(p),s,xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
				return false;
			}
			else{
				for (int p=0;p<Blocks.size();p++){
					if (Blocks[p] !=true) AddNearFieldMat(mat,t.get_son(p),s,MyNearFieldMats_local);
				} 
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			std::vector<bool> Blocks(t.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				Blocks[p] = UpdateBlocks(mat,t,s.get_son(p),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
			}

			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
				return false;
			}
			else{
				for (int p=0;p<Blocks.size();p++){
					if (Blocks[p] !=true) AddNearFieldMat(mat,t,s.get_son(p),MyNearFieldMats_local);
				} 
				return true;
			}
		}
		else{
			std::vector<bool> Blocks(t.get_nb_sons()*s.get_nb_sons());
			for (int p=0; p <t.get_nb_sons();p++){
				for (int l=0; l <s.get_nb_sons();l++){
					Blocks[p+l*t.get_nb_sons()] =UpdateBlocks(mat,t.get_son(p),s.get_son(l),xt,tabt,xs,tabs,MyNearFieldMats_local,MyFarFieldMats_local);
				}
			}
			if ((bsize <= maxblocksize) && std::all_of(Blocks.begin(), Blocks.end(),[](bool block){return block!=true;} ) ) {
					return false;
			}
			else{
				for (int p=0; p <t.get_nb_sons();p++){
					for (int l=0; l <s.get_nb_sons();l++){
						if (Blocks[p+l*t.get_nb_sons()] !=true) AddNearFieldMat(mat,t.get_son(p),s.get_son(l),MyNearFieldMats_local);
					}
				}
				return true;
			}
		}
	}
}

// Build a dense block
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::AddNearFieldMat(IMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, std::vector<SubMatrix<T>*>& MyNearFieldMats_local){
    SubMatrix<T>* submat = new SubMatrix<T>(mat, std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset());

	MyNearFieldMats_local.push_back(submat);

}

// Build a low rank block
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::AddFarFieldMat(IMatrix<T>& mat, const Cluster<ClusterImpl>& t, const Cluster<ClusterImpl>& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, std::vector<LowRankMatrix<T,ClusterImpl>*>& MyFarFieldMats_local, const int& reqrank){
    LowRankMatrix<T,ClusterImpl>* lrmat = new LowRankMatrix<T,ClusterImpl> (std::vector<int>(cluster_tree_t->get_perm_start()+t.get_offset(),cluster_tree_t->get_perm_start()+t.get_offset()+t.get_size()), std::vector<int>(cluster_tree_s->get_perm_start()+s.get_offset(),cluster_tree_s->get_perm_start()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset(),reqrank);
    MyFarFieldMats_local.push_back(lrmat);
	MyFarFieldMats_local.back()->build(mat,t,xt,tabt,s,xs,tabs);

}

// Compute infos
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::ComputeInfos(const std::vector<double>& mytime){
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


	infos["Cluster_mean"]=NbrToStr(meantime[0]);
	infos["Cluster_max"]=NbrToStr(maxtime[0]);
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



template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::mymvprod_local(const T* const in, T* const out, const int& mu) const{

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
    		const LowRankMatrix<T,ClusterImpl>&  M  = *(MyFarFieldMats[b]);
    		int offset_i     = M.get_offset_i();
    		int offset_j     = M.get_offset_j();

			if (!symmetric || offset_i!=offset_j){// remove strictly diagonal blocks
    			M.add_mvprod_row_major(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);
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

			if (!symmetric || offset_i!=offset_j){// remove strictly diagonal blocks
    			M.add_mvprod_row_major(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);
			}
    	}

		// Symmetric part of the diagonal part
		if (symmetric){
			#if _OPENMP
			#pragma omp for schedule(guided)
			#endif
			for(int b=0; b<MyDiagFarFieldMats.size(); b++){
				const LowRankMatrix<T,ClusterImpl>&  M  = *(MyDiagFarFieldMats[b]);
				int offset_i     = M.get_offset_j();
				int offset_j     = M.get_offset_i();

				if (offset_i!=offset_j){// remove strictly diagonal blocks
					M.add_mvprod_row_major(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu,'C');
				}

			}
			// #if _OPENMP
			// #pragma omp for schedule(guided)
			// #endif
			// for(int b=0; b<MyStrictlyDiagFarFieldMats.size(); b++){
			// 	const LowRankMatrix<T,ClusterImpl>&  M  = *(MyStrictlyDiagFarFieldMats[b]);
			// 	int offset_i     = M.get_offset_j();
			// 	int offset_j     = M.get_offset_i();

			// 	M.add_mvprod_row_major_sym(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);

			// }

			// Contribution champ proche
			#if _OPENMP
			#pragma omp for schedule(guided)
			#endif
			for(int b=0; b<MyDiagNearFieldMats.size(); b++){
				const SubMatrix<T>&  M  = *(MyDiagNearFieldMats[b]);
				int offset_i     = M.get_offset_j();
				int offset_j     = M.get_offset_i();
				
				if (offset_i!=offset_j){// remove strictly diagonal blocks
					M.add_mvprod_row_major(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu,'C');
				}
			}

			#if _OPENMP
			#pragma omp for schedule(guided)
			#endif
			for(int b=0; b<MyStrictlyDiagNearFieldMats.size(); b++){
				const SubMatrix<T>&  M  = *(MyStrictlyDiagNearFieldMats[b]);
				int offset_i     = M.get_offset_j();
				int offset_j     = M.get_offset_i();
				M.add_mvprod_row_major_sym(in+offset_j*mu,temp.data()+(offset_i-local_offset)*mu,mu);
			}
		}
    	
        #if _OPENMP
        #pragma omp critical
        #endif
        std::transform (temp.begin(), temp.end(), out, out, std::plus<T>());

    }

}


// template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
// void HMatrix<T, LowRankMatrix, ClusterImpl>::local_to_global(const T* const in, T* const out, const int& mu) const{
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::local_to_global(const T* const in, T* const out, const int& mu) const{
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




template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::mvprod_local(const T* const in, T* const out, T* const work, const int& mu) const{
	double time = MPI_Wtime();

    this->local_to_global(in, work,mu);
    this->mymvprod_local(work,out,mu);

	infos["nb_mat_vec_prod"] = NbrToStr(1+StrToNbr<int>(infos["nb_mat_vec_prod"]));
	infos["total_time_mat_vec_prod"] = NbrToStr(MPI_Wtime()-time+StrToNbr<double>(infos["total_time_mat_vec_prod"]));
}


template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::mvprod_global(const T* const in, T* const out, const int& mu) const{
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::mvprod_subrhs(const T* const in, T* const out, const int& mu, const int& offset, const int& size, const int& local_max_size_j) const{
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
            const LowRankMatrix<T,ClusterImpl>&  M  = *(MyFarFieldMats[b]);
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
template<typename U>
void HMatrix<T, LowRankMatrix, ClusterImpl>::source_to_cluster_permutation(const U* const in, U* const out) const {
	cluster_tree_s->global_to_cluster(in,out);
}

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
template<typename U>
void HMatrix<T, LowRankMatrix, ClusterImpl>::cluster_to_target_permutation(const U* const in, U* const out) const{
	cluster_tree_t->cluster_to_global(in,out);
}




template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
std::vector<T> HMatrix<T, LowRankMatrix, ClusterImpl>::operator*(const std::vector<T>& x) const{
	assert(x.size()==nc);
	std::vector<T> result(nr,0);
	mvprod_global(x.data(),result.data(),1);
	return result;
}


template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
double HMatrix<T, LowRankMatrix, ClusterImpl>::compression() const{

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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::print_infos() const{
	int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

	if (rankWorld==0){
		for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
			std::cout<<it->first<<"\t"<<it->second<<std::endl;
		}
		std::cout << std::endl;
	}
}

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::save_infos(const std::string& outputname,std::ios_base::openmode mode, const std::string& sep) const{
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::save_plot(const std::string& outputname) const{



	std::ofstream outputfile((outputname+"_"+NbrToStr(rankWorld)+".csv").c_str());

	if (outputfile){
		outputfile<<nr<<","<<nc<<std::endl;
		for (typename std::vector<SubMatrix<T>*>::const_iterator it = MyNearFieldMats.begin() ; it != MyNearFieldMats.end() ; ++it){
			outputfile<<(*it)->get_offset_i()<<","<<(*it)->get_ir().size()<<","<<(*it)->get_offset_j()<<","<<(*it)->get_ic().size()<<","<<-1<<std::endl;
		}
		for (typename std::vector<LowRankMatrix<T,ClusterImpl>*>::const_iterator it = MyFarFieldMats.begin() ; it != MyFarFieldMats.end() ; ++it){
			outputfile<<(*it)->get_offset_i()<<","<<(*it)->get_ir().size()<<","<<(*it)->get_offset_j()<<","<<(*it)->get_ic().size()<<","<<(*it)->rank_of()<<std::endl;
		}
		outputfile.close();
	}
	else{
		std::cout << "Unable to create "<<outputname<<std::endl;
	}
}

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
double Frobenius_absolute_error(const HMatrix<T, LowRankMatrix, ClusterImpl>& B, const IMatrix<T>& A){
	double myerr = 0;
	for(int j=0; j<B.MyFarFieldMats.size(); j++){
		double test = Frobenius_absolute_error(*(B.MyFarFieldMats[j]), A);
		myerr += std::pow(test,2);

	}

	double err = 0;
	MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, B.comm);

	return std::sqrt(err);
}
template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
Matrix<T> HMatrix<T, LowRankMatrix, ClusterImpl>::to_dense() const{
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
      const LowRankMatrix<T,ClusterImpl>& lmat = *(MyFarFieldMats[l]);
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
Matrix<T> HMatrix<T, LowRankMatrix, ClusterImpl>::to_dense_perm() const{
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
		const LowRankMatrix<T,ClusterImpl>& lmat = *(MyFarFieldMats[l]);
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

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void HMatrix<T, LowRankMatrix, ClusterImpl>::apply_dirichlet(const std::vector<int>& boundary){
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
