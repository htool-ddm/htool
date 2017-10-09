#ifndef HMATRIX_HPP
#define HMATRIX_HPP

#include <cassert>
#include <fstream>
#include <mpi.h>
#include <map>
#include "matrix.hpp"
#include "parametres.hpp"
#include "cluster.hpp"
#include "wrapper_mpi.hpp"

namespace htool {


//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//
// TODO visualisation for hmat
// TODO take into account symetric structur (same cluster target and source)
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

	std::vector<Block*>		   Tasks;
	std::vector<Block*>		   MyBlocks;

	std::vector<LowRankMatrix<T> > MyFarFieldMats;
	std::vector<SubMatrix<T> >     MyNearFieldMats;
	std::vector<LowRankMatrix<T>*> MyDiagFarFieldMats;
	std::vector<SubMatrix<T>*>		 MyDiagNearFieldMats;

	std::vector<int>               perms;
	std::vector<int>               permt;

	std::vector<std::pair<int,int>> MasterOffsetss;
	std::vector<std::pair<int,int>> MasterOffsett;



	std::map<std::string, double> stats;

	MPI_Comm comm;
	int rankWorld,sizeWorld;


	// Internal methods
	void SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt);
	void SetRanks(Cluster& t);
	void ScatterTasks();
	Block* BuildBlockTree(const Cluster&, const Cluster&);
	void ComputeBlocks(const IMatrix<T>& mat, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	bool UpdateBlocks(const IMatrix<T>&mat ,const Cluster&, const Cluster&, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs);
	void AddNearFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s);
	void AddFarFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, const int& reqrank=-1);
	void ComputeStats(const std::vector<double>& mytimes);


public:
	// TODO constructors that take clusters as arguments

	// Build
	void build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Full constructor
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without tab
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Constructor without radius, tab and mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<R3>&xs, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters

	// Symetric build
	void build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Full symetric constructor
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Symetric constructor without radius
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without tab
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters

	// Constructor without radius, tab and mass
	HMatrix(const IMatrix<T>&, const std::vector<R3>& xt, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one different clusters



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
	LowRankMatrix<T> get_lrmat(int i) const{return MyFarFieldMats[i];}

	void compute_diag_block(T* diag_block);

	const std::map<std::string,double>& get_stats () const { return stats;}
	void add_stats(const std::string& keyname, const double& value){stats[keyname]=value;}
	void print_stats();

	// void local_to_global(const T* const in, T* const out) const;
	void mvprod_global(const T* const in, T* const out) const;
	void mvprod_local(const T* const in, T* const out) const;
	std::vector<T> operator*( const std::vector<T>& x) const;

	// 1- !!!
	double compression() const;

	friend double Frobenius_absolute_error<LowRankMatrix,T>(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A);

};

// build
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::build(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, MPI_Comm comm0){

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	MPI_Comm_dup(comm0,&comm);
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> mytimes(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	Cluster t(xt,rt,tabt,gt,permt); Cluster s(xs,rs,tabs,gs,perms);
	assert(std::pow(2,t.get_min_depth())>sizeWorld);
	SetRanks(t);

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(t,s);
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
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs,comm0);
}

// Constructor without rt and rs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<int>& tabs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), reqrank(reqrank0) {

	this->build(mat, xt, std::vector<double>(xt.size(),0), tabt, gt, xs, std::vector<double>(xs.size(),0), tabs, gs, comm0);
}

// Constructor without gt and gs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<int>& tabs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), reqrank(reqrank0) {
	this->build(mat, xt, rt, tabt, std::vector<double>(xt.size(),1), xs, rs, tabs, std::vector<double>(xs.size(),1), comm0);
}

// Constructor without tabt and tabs
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const std::vector<R3>&xs, const std::vector<double>& rs, const std::vector<double>& gs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), reqrank(reqrank0) {
	std::vector<int> tabt(xt.size()), tabs(xs.size());
	std::iota(tabt.begin(),tabt.end(),int(0));
	std::iota(tabs.begin(),tabs.end(),int(0));
	this->build(mat, xt, rt, tabt, gt, xs, rs, tabs, gs, comm0);
}

// Constructor without radius, mass and tab
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat, const std::vector<R3>& xt, const std::vector<R3>& xs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), reqrank(reqrank0) {
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

	// Construction arbre des paquets
	double time = MPI_Wtime();
	Cluster t(xt,rt,tabt,gt,permt);
  perms=permt; // bof
	assert(std::pow(2,t.get_min_depth())>sizeWorld);
	SetRanks(t);

	mytimes[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(t,t);
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
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),reqrank(reqrank0){

		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without rt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<double>& gt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),reqrank(reqrank0){
		this->build(mat,xt,std::vector<int>(xt.size(),0),tabt,gt,comm0);
}


// Symetric constructor without tabt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<double>& gt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),reqrank(reqrank0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,rt,tabt,gt,comm0);
}

// Symetric constructor without gt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<double>& rt, const std::vector<int>& tabt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),reqrank(reqrank0), comm(comm0){
		this->build(mat,xt,rt,tabt,std::vector<int>(xt.size(),1),comm0);
}

// Symetric constructor without rt, tabt and gt
template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),reqrank(reqrank0){
		std::vector<int> tabt(xt.size());
 		std::iota(tabt.begin(),tabt.end(),int(0));
		this->build(mat,xt,std::vector<int>(xt.size(),0),tabt,std::vector<int>(xt.size(),1),comm0);
}



// Rank tags
template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt){
	if(t.get_depth()<depth){
		t.set_rank(-1);
		SetRanksRec(t.get_son(0), depth, 2*cnt);
		SetRanksRec(t.get_son(1), depth, 2*cnt+1);
	}
	else{
		t.set_rank(cnt-pow(2,depth));
		if (t.get_depth() == depth){
			MasterOffsett[cnt-pow(2,depth)] = std::pair<int,int>(t.get_offset(),t.get_size());
		}
		if (!t.IsLeaf()){
			SetRanksRec(t.get_son(0), depth, cnt);
			SetRanksRec(t.get_son(1), depth, cnt);
		}
	}
}

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::SetRanks(Cluster& t){
	int rankWorld, sizeWorld;
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  MasterOffsett.resize(sizeWorld);

	SetRanksRec(t, log2(sizeWorld), 1);
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

	std::cout << "Tasks : "<<Tasks.size()<<std::endl;
  for(int b=0; b<Tasks.size(); b++){
    	//if (b%sizeWorld == rankWorld)
    if ((*(Tasks[b])).tgt_().get_rank() == rankWorld){
    		MyBlocks.push_back(Tasks[b]);
		}
	}
	std::cout << "rank : "<<rankWorld<<" "<<"Block : "<<MyBlocks.size() <<std::endl;
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
	MyNearFieldMats.emplace_back(mat, std::vector<int>(permt.begin()+t.get_offset(),permt.begin()+t.get_offset()+t.get_size()), std::vector<int>(perms.begin()+s.get_offset(),perms.begin()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset());
	if (s.get_rank()==rankWorld){
		MyDiagNearFieldMats.push_back(&(MyNearFieldMats.back()));
	}
}

// Build a low rank block
template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::AddFarFieldMat(const IMatrix<T>& mat, const Cluster& t, const Cluster& s, const std::vector<R3> xt,const std::vector<int> tabt, const std::vector<R3> xs, const std::vector<int>tabs, const int& reqrank){
	MyFarFieldMats.emplace_back(std::vector<int>(permt.begin()+t.get_offset(),permt.begin()+t.get_offset()+t.get_size()), std::vector<int>(perms.begin()+s.get_offset(),perms.begin()+s.get_offset()+s.get_size()),t.get_offset(),s.get_offset(),reqrank);
	MyFarFieldMats.back().build(mat,t,xt,tabt,s,xs,tabs);
	if (s.get_rank()==rankWorld){
		MyDiagFarFieldMats.push_back(&(MyFarFieldMats.back()));
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
		MPI_Reduce(MPI_IN_PLACE, &maxstats.front(), 3, MPI_INT, MPI_MAX, 0,comm);
		MPI_Reduce(MPI_IN_PLACE, &minstats.front(), 3, MPI_INT, MPI_MIN, 0,comm);
		MPI_Reduce(MPI_IN_PLACE, &meanstats.front(),3, MPI_DOUBLE, MPI_SUM, 0,comm);
	}
	else{
		MPI_Reduce(&maxstats.front(), &maxstats.front(), 3, MPI_INT, MPI_MAX, 0,comm);
		MPI_Reduce(&minstats.front(), &minstats.front(), 3, MPI_INT, MPI_MIN, 0,comm);
		MPI_Reduce(&meanstats.front(), &meanstats.front(),3, MPI_DOUBLE, MPI_SUM, 0,comm);
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
	MPI_Reduce(&mytime.front(), &maxtime.front(), 4, MPI_DOUBLE, MPI_MAX, 0,comm);
	MPI_Reduce(&mytime.front(), &meantime.front(), 4, MPI_DOUBLE, MPI_SUM, 0,comm);

	meantime /= sizeWorld;

	// save
	stats["Cluster tree (mean)"]=meantime[0];
	stats["Cluster tree  (max)"]=maxtime[0];
	stats["Block tree   (mean)"]=meantime[1];
	stats["Block tree    (max)"]=maxtime[1];
	stats["Scatter tree (mean)"]=meantime[2];
	stats["Scatter tree  (max)"]=maxtime[2];
	stats["Blocks       (mean)"]=meantime[3];
	stats["Blocks           (max)"]  =maxtime[3];
	stats["Dense block size (max)"]  =maxstats[0];
	stats["Dense block size (mean)"] =meanstats[0];
	stats["Dense block size (min)"]  =minstats[0];
	stats["Low rank block size (max)"]   =maxstats[1];
	stats["Low rank  block size (mean)"] =meanstats[1];
	stats["Low rank  block size (min)"]  =minstats[1];
	stats["Rank (max)"]  =maxstats[2];
	stats["Rank (mean)"] =meanstats[2];
	stats["Rank (min)"]  =minstats[2];
	stats["Number of lrmat"] = nlrmat;
	stats["Number of dmat"]  = ndmat;
}



template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_local(const T* const in, T* const out) const{

	double time = MPI_Wtime();
	std::fill(out,out+MasterOffsett[rankWorld].second,0);

	// Contribution champ lointain
	for(int b=0; b<MyFarFieldMats.size(); b++){
		const LowRankMatrix<T>&  M  = MyFarFieldMats[b];
		int offset_i     = M.get_offset_i();
		int offset_j     = M.get_offset_j();

		M.add_mvprod(in+offset_j,out+offset_i-MasterOffsett[rankWorld].first);

	}
	// Contribution champ proche
	for(int b=0; b<MyNearFieldMats.size(); b++){
		const SubMatrix<T>&  M  = MyNearFieldMats[b];
		int offset_i     = M.get_offset_i();
		int offset_j     = M.get_offset_j();

		M.add_mvprod(in+offset_j,out+offset_i-MasterOffsett[rankWorld].first);
	}

}


// template< template<typename> class LowRankMatrix, typename T>
// void HMatrix<LowRankMatrix,T >::local_to_global(const T* const in, T* const out) const{
// 	//
// 	int offset = MasterOffsett[rankWorld].first;
// 	int size   = MasterOffsett[rankWorld].second;
//
// 	// Allgather
// 	std::vector<T> rcv;
// 	rcv.resize(nr);
//
// 	std::vector<int> recvcounts(sizeWorld);
// 	std::vector<int>  displs(sizeWorld);
//
// 	displs[0] = 0;
//
// 	for (int i=0; i<sizeWorld; i++) {
// 		recvcounts[i] = MasterOffsett[i].second;
// 		if (i > 0)
// 			displs[i] = displs[i-1] + recvcounts[i-1];
// 	}
//
// 	MPI_Allgatherv(in, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), &(rcv.front()), &(recvcounts.front()), &(displs.front()), wrapper_mpi<T>::mpi_type(), comm);
//
// 	// for (int i=0; i<sizeWorld; i++)
// 	// for (int j=0; j< MasterClusters[i].size(); j++)
// 	// 	out[MasterClusters[i][j]] = rcv[displs[i]+j];
//
//
// 	// Permutation
// 	for (int i = 0; i<permt.size();i++){
// 		out[permt[i]]=rcv[i];
// 	}
// }












template< template<typename> class LowRankMatrix, typename T>
void HMatrix<LowRankMatrix,T >::mvprod_global(const T* const in, T* const out) const{
	//
	int offset = MasterOffsett[rankWorld].first;
	int size   = MasterOffsett[rankWorld].second;

	double time = MPI_Wtime();
	std::vector<T> in_perm(nc);
	std::vector<T> out_not_perm(nr);

	// Permutation
	for (int i = 0; i<perms.size();i++){
		in_perm[i]=in[perms[i]];
	}
	// mvprod local
	mvprod_local(in_perm.data(),out_not_perm.data()+offset);


	// Allgather
	std::vector<T> snd(size);
	std::vector<int> recvcounts(sizeWorld);
	std::vector<int>  displs(sizeWorld);

	displs[0] = 0;

	for (int i=0; i<sizeWorld; i++) {
		recvcounts[i] = MasterOffsett[i].second;
		if (i > 0)
			displs[i] = displs[i-1] + recvcounts[i-1];
	}

	// std::copy_n(snd.data(),size,out+offset);
	MPI_Allgatherv(MPI_IN_PLACE, recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out_not_perm.data(), &(recvcounts.front()), &(displs.front()), wrapper_mpi<T>::mpi_type(), comm);
	// MPI_Allgatherv(snd.data(), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), out, &(recvcounts.front()), &(displs.front()), wrapper_mpi<T>::mpi_type(), comm);

	// Permutation
	for (int i = 0; i<permt.size();i++){
		out[permt[i]]=out_not_perm[i];
	}
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
	int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

	if (rankWorld==0){
		for (std::map<std::string,double>::const_iterator it = stats.begin() ; it != stats.end() ; ++it){
			std::cout<<it->first<<"\t"<<it->second<<std::endl;
		}
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
