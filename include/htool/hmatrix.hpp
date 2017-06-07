#ifndef HMATRIX_HPP
#define HMATRIX_HPP

#include <cassert>
#include <fstream>
#include <mpi.h>
#include <map>
#include "matrix.hpp"
#include "parametres.hpp"
#include "cluster.hpp"

// #include "loading.hpp"
// #include "lrmat.hpp"
namespace htool {


//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//
template< template<typename> class LowRankMatrix, typename T >
class HMatrix: public Parametres{

private:

	const std::vector<R3>& xt;
	const std::vector<R3>& xs;
	const std::vector<int>& tabt;
	const std::vector<int>& tabs;


	std::vector<LowRankMatrix<T> > FarFieldMat;
	std::vector<SubMatrix<T> >     NearFieldMat;

	std::vector<Block*>		  Tasks;
	std::vector<Block*>		  MyBlocks;

	Block* BuildBlockTree(const Cluster&, const Cluster&);

	bool UpdateBlocks(const Cluster&, const Cluster&);

	void ScatterTasks();
	void ComputeBlocks();

	std::vector<LowRankMatrix<T> > MyFarFieldMats;
	std::vector<SubMatrix<T> >     MyNearFieldMats;

	const int& reqrank;

	std::map<std::string, double> stats;

public:

	HMatrix(const VirtualMatrix<T>&, const std::vector<R3>&, const std::vector<double>&, const std::vector<int>&, const std::vector<R3>&, const std::vector<double>&, const std::vector<int>&, const int& reqrank=-1, const MPI_Comm& comm=MPI_COMM_WORLD); // To be used with two different clusters
	HMatrix(const VirtualMatrix<T>&, const std::vector<R3>&, const std::vector<double>&, const std::vector<int>&, const int& reqrank=-1, const MPI_Comm& comm=MPI_COMM_WORLD); // To be used with one cluster

	~HMatrix() {
		for (int i=0; i<Tasks.size(); i++)
			delete Tasks[i];
	}

	friend const int& nb_rows(const HMatrix& A){ return nb_rows(A.mat);}
	friend const int& nb_cols(const HMatrix& A){ return nb_cols(A.mat);}

	friend const std::map<std::string,double>& get_stats (const HMatrix& A) { return A.stats;}
	friend void add_stats(HMatrix& A, const std::string& keyname, const double& value){A.stats[keyname]=value;}
	friend void print_stats(const HMatrix& A, const MPI_Comm& comm/*=MPI_COMM_WORLD*/);

	friend const int nb_lrmats(const HMatrix& A, const MPI_Comm& comm=MPI_COMM_WORLD){ int res=A.MyFarFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;}
	friend const int nb_densemats(const HMatrix& A, const MPI_Comm& comm=MPI_COMM_WORLD){ int res=A.MyNearFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;}

	friend void MvProd(std::vector<Cplx>&, const HMatrix&, const std::vector<Cplx>&);
	friend std::pair<double,double> MvProdMPI(std::vector<Cplx>& f, const HMatrix& A, const std::vector<Cplx>& x, const MPI_Comm& comm/*=MPI_COMM_WORLD*/);

	// 1- !!!
	friend double CompressionRate(const HMatrix& hmat, const MPI_Comm& comm/*=MPI_COMM_WORLD*/);

	friend void Output(const HMatrix&, std::string filename);
	friend const LowRankMatrix<T>& GetLowRankMatrix(HMatrix m, int i){
		assert(i<m.FarFieldMats.size());
		return m.FarFieldMats[i];}

	friend double squared_absolute_error(const HMatrix& B, const VirtualMatrix<T>& A, const MPI_Comm& comm=MPI_COMM_WORLD){
		double myerr = 0;
		for(int j=0; j<B.MyFarFieldMats.size(); j++){
			SubMatrix<T> subm(A,ir_(B.MyFarFieldMats[j]),ic_(B.MyFarFieldMats[j]));
			myerr += squared_absolute_error(B.MyFarFieldMats[j], subm);
		}
		for(int j=0; j<B.MyNearFieldMats.size(); j++){
			SubMatrix<T> subm(A,ir_(B.MyNearFieldMats[j]),ic_(B.MyNearFieldMats[j]));
			myerr += squared_absolute_error(B.MyNearFieldMats[j], subm);
		}


		double err = 0;
		MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, comm);

		return err;
	}

};

template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const VirtualMatrix<T>& mat0,
		 const std::vector<R3>& xt0, const std::vector<double>& rt, const std::vector<int>& tabt0,
		 const std::vector<R3>& xs0, const std::vector<double>& rs, const std::vector<int>& tabs0, const int& reqrank0, const MPI_Comm& comm):

mat(mat0), xt(xt0), xs(xs0), tabt(tabt0), tabs(tabs0),reqrank(reqrank0) {
	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	int rankWorld, sizeWorld;
    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> myttime(4), maxtime(4), meantime(4);


	// Construction arbre des paquets
	double time = MPI_Wtime();
	Cluster t(xt,rt,tabt); Cluster s(xs,rs,tabs);
	myttime[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(t,s);
	if (B != NULL) Tasks.push_back(B);
	myttime[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	myttime[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks();
	myttime[3] = MPI_Wtime() - time;

	MPI_Reduce(&myttime.front(), &maxtime.front(), 4, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&myttime.front(), &meantime.front(), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
	meantime /= sizeWorld;
	stats["Cluster tree (mean)"]=meantime[0];
	stats["Cluster tree  (max)"]=maxtime[0];
	stats["Block tree   (mean)"]=meantime[1];
	stats["Block tree    (max)"]=maxtime[1];
	stats["Scatter tree (mean)"]=meantime[2];
	stats["Scatter tree  (max)"]=maxtime[2];
	stats["Blocks       (mean)"]=meantime[3];
	stats["Blocks        (max)"]=maxtime[3];
}

template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const VirtualMatrix<T>& mat0,
		 const vectR3& xt0, const vectReal& rt, const vectInt& tabt0, const int& reqrank0, const MPI_Comm& comm):

mat(mat0), xt(xt0), xs(xt0), tabt(tabt0), tabs(tabt0),reqrank(reqrank0) {
	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );

	int rankWorld, sizeWorld;
    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    std::vector<double> myttime(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	Cluster t(xt,rt,tabt);
	myttime[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	Block* B = BuildBlockTree(t,t);
	if (B != NULL) Tasks.push_back(B);
	myttime[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	myttime[2] = MPI_Wtime() - time;

	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks();
	myttime[3] = MPI_Wtime() - time;

	MPI_Reduce(&myttime.front(), &maxtime.front(), 4, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&myttime.front(), &meantime.front(), 4, MPI_DOUBLE, MPI_SUM, 0, comm);
	meantime /= sizeWorld;
	stats["Cluster tree (mean)"]=meantime[0];
	stats["Cluster tree  (max)"]=maxtime[0];
	stats["Block tree   (mean)"]=meantime[1];
	stats["Block tree    (max)"]=maxtime[1];
	stats["Scatter tree (mean)"]=meantime[2];
	stats["Scatter tree  (max)"]=maxtime[2];
	stats["Blocks       (mean)"]=meantime[3];
	stats["Blocks        (max)"]=maxtime[3];
}

template< template<typename> class LowRankMatrix, typename T >
Block* HMatrix<LowRankMatrix, T >::BuildBlockTree(const Cluster& t, const Cluster& s){
	Block* B = new Block(t,s);
	int bsize = t.num_().size()*s.num_().size();
	B->ComputeAdmissibility();
	if( B->IsAdmissible() ){
		Tasks.push_back(B);
		return NULL;
	}
	else if( s.IsLeaf() ){
				if( t.IsLeaf() ){

					return B;
				}
				else{
					Block* r1 = BuildBlockTree(t.son_(0),s);
					Block* r2 = BuildBlockTree(t.son_(1),s);
					if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL)) {
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
			Block* r3 = BuildBlockTree(t,s.son_(0));
			Block* r4 = BuildBlockTree(t,s.son_(1));
			if ((bsize <= maxblocksize) && (r3 != NULL) && (r4 != NULL)) {
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
			if (t.num_().size()>s.num_().size()){
				Block* r1 = BuildBlockTree(t.son_(0),s);
				Block* r2 = BuildBlockTree(t.son_(1),s);
				if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL)) {
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
				Block* r3 = BuildBlockTree(t,s.son_(0));
				Block* r4 = BuildBlockTree(t,s.son_(1));
				if ((bsize <= maxblocksize) && (r3 != NULL) && (r4 != NULL)) {
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

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::ScatterTasks(){
	int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    for(int b=0; b<Tasks.size(); b++)
    	if (b%sizeWorld == rankWorld)
    		MyBlocks.push_back(Tasks[b]);
}

template< template<typename> class LowRankMatrix, typename T >
bool HMatrix<LowRankMatrix,T >::UpdateBlocks(const Cluster& t, const Cluster& s){
	const std::vector<int>& I = t.num_();
	const std::vector<int>& J = s.num_();
	int bsize = t.num_().size()*s.num_().size();
	Block B(t,s);
	B.ComputeAdmissibility();
	if( B.IsAdmissible() ){
		LowRankMatrix<T> lrm(mat,I,J,t,s);
		if(rank_of(lrm)!=-5){
			MyFarFieldMats.push_back(lrm);
			return true;
		}
	}
	if( s.IsLeaf() ){
		if( t.IsLeaf() ){
			return false;
		}
		else{
			bool b1 = UpdateBlocks(t.son_(0),s);
			bool b2 = UpdateBlocks(t.son_(1),s);
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
				return false;
			else {
				if (b1 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(0).num_(),J));
				if (b2 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(1).num_(),J));
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			bool b3 = UpdateBlocks(t,s.son_(0));
			bool b4 = UpdateBlocks(t,s.son_(1));
			if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
				return false;
			else{
				if (b3 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(0).num_()));
				if (b4 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(1).num_()));
				return true;
			}
		}
		else{

			if (t.num_().size()>s.num_().size()){
				bool b1 = UpdateBlocks(t.son_(0),s);
				bool b2 = UpdateBlocks(t.son_(1),s);
				if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
					return false;
				else {
					if (b1 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(0).num_(),J));
					if (b2 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(1).num_(),J));
					return true;
				}
			}
			else{
				bool b3 = UpdateBlocks(t,s.son_(0));
				bool b4 = UpdateBlocks(t,s.son_(1));
				if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
					return false;
				else{
					if (b3 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(0).num_()));
					if (b4 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(1).num_()));
					return true;
				}
			}
		}
	}
}

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix,T >::ComputeBlocks(){
    for(int b=0; b<MyBlocks.size(); b++) {
    	const Block& B = *(MyBlocks[b]);
   		const Cluster& t = B.tgt_();
			const Cluster& s = B.src_();
			const std::vector<int>& I = t.num_();
			const std::vector<int>& J = s.num_();
			if( B.IsAdmissible() ){
				LowRankMatrix<T> lrm(mat,I,J,t,s);
				if(rank_of(lrm)!=-5){
					MyFarFieldMats.push_back(lrm);
				}
			else {
				if( s.IsLeaf() ){
					if( t.IsLeaf() ){
						MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
					}
					else{
						bool b1 = UpdateBlocks(t.son_(0),s);
						bool b2 = UpdateBlocks(t.son_(1),s);
						if ((b1 != true) && (b2 != true))
							MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
						else {
							if (b1 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(0).num_(),J));
							if (b2 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(1).num_(),J));
						}
					}
				}
				else{
					if( t.IsLeaf() ){
						bool b3 = UpdateBlocks(t,s.son_(0));
						bool b4 = UpdateBlocks(t,s.son_(1));
						if ((b3 != true) && (b4 != true))
							MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
						else {
							if (b3 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(0).num_()));
							if (b4 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(1).num_()));
						}
					}
					else{
						if (t.num_().size()>s.num_().size()){
							bool b1 = UpdateBlocks(t.son_(0),s);
							bool b2 = UpdateBlocks(t.son_(1),s);
							if ((b1 != true) && (b2 != true))
								MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
							else {
								if (b1 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(0).num_(),J));
								if (b2 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,t.son_(1).num_(),J));
							}
						}
						else{
							bool b3 = UpdateBlocks(t,s.son_(0));
							bool b4 = UpdateBlocks(t,s.son_(1));
							if ((b3 != true) && (b4 != true))
								MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
							else {
								if (b3 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(0).num_()));
								if (b4 != true) MyNearFieldMats.push_back(SubMatrix<T>(mat,I,s.son_(1).num_()));
							}
						}
					}
				}
			}
		}
		else {
			MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
		}
    }
}
//
// template<typename T, template<typename> class LowRankMatrix >
// void MvProd(std::vector<T>& f, const HMatrix<LowRankMatrix,T>& A, const std::vector<T>& x){
// 	assert(f.size()==x.size()); std::fill(f.begin(), f.end(), 0);
//
// 	const std::vector<LowRankMatrix>&    FarFieldMat  = A.FarFieldMats;
// 	const std::vector<SubMatrix<T> >&    NearFieldMat = A.NearFieldMats;
//
// 	// Contribution champ lointain
// 	for(int b=0; b<FarFieldMat.size(); b++){
// 		const LowRankMatrix&  M  = FarFieldMat[b];
// 		const std::vector<int>&        It = M.ir_();
// 		const std::vector<int>&        Is = M.ic_();
//
// 		ConstSubVectCplx xx(x,Is);
// 		SubVectCplx ff(f,It);
// 		MvProd(ff,M,xx);
// 	}
//
// 	// Contribution champ proche
// 	for(int b=0; b<NearFieldMat.size(); b++){
// 		const SubMatrix<T>&  M  = NearFieldMat[b];
//
// 		const std::vector<int>& It = M.ir_();
// 		const std::vector<int>& Is = M.ic_();
//
// 		ConstSubVectCplx xx(x,Is);
// 		SubVectCplx ff(f,It);
// 		MvProd(ff,M,xx);
// 	}
//
// }

template< template<typename> class LowRankMatrix, typename T >
std::pair<double,double> MvProdMPI(std::vector<T>& f, const HMatrix<LowRankMatrix,T >& A, const std::vector<T>& x, const MPI_Comm& comm=MPI_COMM_WORLD){
	int rankWorld, sizeWorld;
   	MPI_Comm_size(comm, &sizeWorld);
   	MPI_Comm_rank(comm, &rankWorld);

	double time = MPI_Wtime();

	assert(nb_rows(A.mat)==size(f) && nb_cols(A.mat)==size(x)); std::fill(f.begin(), f.end(), 0);
	const std::vector<LowRankMatrix<T> >&    MyFarFieldMats  = A.MyFarFieldMats;
	const std::vector<SubMatrix<T> >&    MyNearFieldMats = A.MyNearFieldMats;

	// Contribution champ lointain
	for(int b=0; b<MyFarFieldMats.size(); b++){
		const LowRankMatrix<T>&  M  = MyFarFieldMats[b];
		const std::vector<int>&        It = ir_(M);
		const std::vector<int>&        Is = ic_(M);

		/*
		ConstSubVectCplx xx(x,Is);
		SubVectCplx ff(f,It);
		MvProd(ff,M,xx);
		*/
		std::vector<T> lhs(It.size());
		std::vector<T> rhs(Is.size());

		for (int i=0; i<Is.size(); i++)
			rhs[i] = x[Is[i]];

		MvProd(lhs,M,rhs);

		for (int i=0; i<It.size(); i++)
			f[It[i]] += lhs[i];
	}

	// Contribution champ proche
	for(int b=0; b<MyNearFieldMats.size(); b++){
		const SubMatrix<T>&  M  = MyNearFieldMats[b];

		const std::vector<int>& It = ir_ (M);
		const std::vector<int>& Is = ic_ (M);

		std::vector<T> lhs(It.size());
		std::vector<T> rhs(Is.size());

		for (int i=0; i<Is.size(); i++)
			rhs[i] = x[Is[i]];

		MvProd(lhs,M,rhs);

		for (int i=0; i<It.size(); i++)
			f[It[i]] += lhs[i];
	}
	time = MPI_Wtime() - time;

	double meantime;
	double maxtime;
	MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&time, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
// 	if (rankWorld == 0) {
		meantime /= sizeWorld;
// 		std::cout << "MvProd: \t mean = " << meantime << ", \t max = " << maxtime << endl;
// 	}

	MPI_Allreduce(MPI_IN_PLACE, &f.front(), f.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, comm);

	return std::pair<double,double>(meantime,maxtime);
}

template< template<typename> class LowRankMatrix, typename T >
double CompressionRate(const HMatrix<LowRankMatrix,T >& hmat, const MPI_Comm& comm=MPI_COMM_WORLD){

	double mycomp = 0.;
	double size = ( (hmat.tabt).size() )*( (hmat.tabs).size() );
	const std::vector<LowRankMatrix<T> >& MyFarFieldMats  = hmat.MyFarFieldMats;
	const std::vector<SubMatrix<T> >& MyNearFieldMats = hmat.MyNearFieldMats;

	for(int j=0; j<MyFarFieldMats.size(); j++){
		double nr   = nb_rows(MyFarFieldMats[j]);
		double nc   = nb_cols(MyFarFieldMats[j]);
		double rank = rank_of(MyFarFieldMats[j]);
		mycomp += rank*(nr + nc)/size;
	}

	for(int j=0; j<MyNearFieldMats.size(); j++){
		double nr   = nb_rows(MyNearFieldMats[j]);
		double nc   = nb_cols(MyNearFieldMats[j]);
		mycomp += nr*nc/size;
	}

	double comp = 0;
	MPI_Allreduce(&mycomp, &comp, 1, MPI_DOUBLE, MPI_SUM, comm);

	return 1-comp;
}

template<typename T, template<typename> class LowRankMatrix >
void print_stats(const HMatrix<LowRankMatrix,T>& A, const MPI_Comm& comm=MPI_COMM_WORLD){
	int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);

	if (rankWorld==0){
		for (std::map<std::string,double>::const_iterator it = A.stats.begin() ; it != A.stats.end() ; ++it){
			std::cout<<it->first<<"\t"<<it->second<<std::endl;
		}
	}
}


} //namespace
#endif
