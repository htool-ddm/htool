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

// #include "loading.hpp"
// #include "lrmat.hpp"
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

	std::vector<Block*>		   Tasks;
	std::vector<Block*>		   MyBlocks;

	std::vector<std::vector<int>>	   MasterClusters;

	std::vector<LowRankMatrix<T> > MyFarFieldMats;
	std::vector<SubMatrix<T> >     MyNearFieldMats;

	std::map<std::string, double> stats;

	MPI_Comm comm;

	// Internal methods
	Block* BuildBlockTree(const Cluster&, const Cluster&);
	bool UpdateBlocks(const IMatrix<T>&mat ,const Cluster&, const Cluster&);
	void ScatterTasks();
	void ComputeBlocks(const IMatrix<T>& mat);
	void SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt);
	void SetRanks(Cluster& t);

public:
	HMatrix(const IMatrix<T>&, const std::vector<R3>&, const std::vector<int>&, const std::vector<R3>&, const std::vector<int>&, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with two different clusters
	HMatrix(const IMatrix<T>&, const std::vector<R3>&, const std::vector<int>&, const int& reqrank=-1, MPI_Comm comm=MPI_COMM_WORLD); // To be used with one cluster

	~HMatrix() {
		for (int i=0; i<Tasks.size(); i++)
			delete Tasks[i];
	}


	// Getters
	int nb_rows() const { return nr;}
	int nb_cols() const { return nc;}
	MPI_Comm get_comm() const {return comm;}
	int get_nlrmat() const {return MyFarFieldMats.size();}
	int get_ndmat() const {return MyNearFieldMats.size();}
	LowRankMatrix<T> get_lrmat(int i) const{return MyFarFieldMats[i];}

	std::map<std::string,double>& get_stats () const { return stats;}
	void add_stats(const std::string& keyname, const double& value){stats[keyname]=value;}
	void print_stats();

	int nb_lrmats(){
		int res=MyFarFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;
	}
	int nb_densemats(){
		int res=MyNearFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;}

	std::vector<T> operator*( const std::vector<T>& x) const;

	// 1- !!!
	double compression() const;

	friend double Frobenius_absolute_error<LowRankMatrix,T>(const HMatrix<LowRankMatrix,T>& B, const IMatrix<T>& A);

};

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt){
	if(t.get_depth()<depth){
		t.set_rank(-1);
		SetRanksRec(t.get_son(0), depth, 2*cnt);
		SetRanksRec(t.get_son(1), depth, 2*cnt+1);
	}
	else{
		t.set_rank(cnt-pow(2,depth));
		if (t.get_depth() == depth)
			MasterClusters[cnt-pow(2,depth)] = t.get_num();
		if (!t.IsLeaf()){
			SetRanksRec(t.get_son(0), depth, cnt);
			SetRanksRec(t.get_son(1), depth, cnt);
		}
	}
}

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix, T >::SetRanks(Cluster& t){
	int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    MasterClusters.resize(sizeWorld);

	SetRanksRec(t, log2(sizeWorld), 1);
}

template< template<typename> class LowRankMatrix, typename T >
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
	const std::vector<R3>& xt, const std::vector<int>& tabt, const std::vector<R3>& xs, const std::vector<int>& tabs, const int& reqrank0, MPI_Comm comm0): nr(mat.nb_rows()),nc(mat.nb_cols()), reqrank(reqrank0), comm(comm0) {

	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabs.size() );
	int rankWorld, sizeWorld;
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> myttime(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	Cluster t(xt,tabt); Cluster s(xs,tabs);
  t.build();s.build();
	SetRanks(t);
	myttime[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	if (rankWorld==0){
	Block* B = BuildBlockTree(t,s);
	if (B != NULL) Tasks.push_back(B);
	myttime[1] = MPI_Wtime() - time;

	// Repartition des blocs sur les processeurs
	time = MPI_Wtime();
	ScatterTasks();
	myttime[2] = MPI_Wtime() - time;
	}
	// Assemblage des sous-matrices
	time = MPI_Wtime();
	ComputeBlocks(mat);
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
HMatrix<LowRankMatrix, T >::HMatrix(const IMatrix<T>& mat,
		 const std::vector<R3>& xt, const std::vector<int>& tabt, const int& reqrank0,  MPI_Comm comm0):nr(mat.nb_rows()),nc(mat.nb_cols()),reqrank(reqrank0), comm(comm0){
	assert( mat.nb_rows()==tabt.size() && mat.nb_cols()==tabt.size() );

	int rankWorld, sizeWorld;
	MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  std::vector<double> myttime(4), maxtime(4), meantime(4);

	// Construction arbre des paquets
	double time = MPI_Wtime();
	Cluster t(xt,tabt);t.build();
	SetRanks(t);
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
	ComputeBlocks(mat);
	myttime[3] = MPI_Wtime() - time;

	MPI_Reduce(&myttime.front(), &maxtime.front(), 4, MPI_DOUBLE, MPI_MAX, 0,comm);
	MPI_Reduce(&myttime.front(), &meantime.front(), 4, MPI_DOUBLE, MPI_SUM, 0,comm);
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
	int bsize = t.get_num().size()*s.get_num().size();
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
					Block* r1 = BuildBlockTree(t.get_son(0),s);
					Block* r2 = BuildBlockTree(t.get_son(1),s);
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
			Block* r3 = BuildBlockTree(t,s.get_son(0));
			Block* r4 = BuildBlockTree(t,s.get_son(1));
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
			if (t.get_num().size()>s.get_num().size()){
				Block* r1 = BuildBlockTree(t.get_son(0),s);
				Block* r2 = BuildBlockTree(t.get_son(1),s);
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
				Block* r3 = BuildBlockTree(t,s.get_son(0));
				Block* r4 = BuildBlockTree(t,s.get_son(1));
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
    	//if (b%sizeWorld == rankWorld)
    	if ((*(Tasks[b])).tgt_().get_rank() == rankWorld)
    		MyBlocks.push_back(Tasks[b]);
}

template< template<typename> class LowRankMatrix, typename T >
bool HMatrix<LowRankMatrix,T >::UpdateBlocks(const IMatrix<T>& mat,const Cluster& t, const Cluster& s){
	const std::vector<int>& I = t.get_num();
	const std::vector<int>& J = s.get_num();
	int bsize = t.get_num().size()*s.get_num().size();
	Block B(t,s);
	B.ComputeAdmissibility();
	if( B.IsAdmissible() ){
		MyFarFieldMats.emplace_back(I,J,reqrank);
		MyFarFieldMats.back().build(mat,t,s);
		if(MyFarFieldMats.back().rank_of()!=-1){
			return true;
		}
	}
	if( s.IsLeaf() ){
		if( t.IsLeaf() ){
			return false;
		}
		else{
			bool b1 = UpdateBlocks(mat,t.get_son(0),s);
			bool b2 = UpdateBlocks(mat,t.get_son(1),s);
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
				return false;
			else {
				if (b1 != true) MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
				if (b2 != true) MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			bool b3 = UpdateBlocks(mat,t,s.get_son(0));
			bool b4 = UpdateBlocks(mat,t,s.get_son(1));
			if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
				return false;
			else{
				if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
				if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
				return true;
			}
		}
		else{

			if (t.get_num().size()>s.get_num().size()){
				bool b1 = UpdateBlocks(mat,t.get_son(0),s);
				bool b2 = UpdateBlocks(mat,t.get_son(1),s);
				if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
					return false;
				else {
					if (b1 != true) MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
					if (b2 != true) MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
					return true;
				}
			}
			else{
				bool b3 = UpdateBlocks(mat,t,s.get_son(0));
				bool b4 = UpdateBlocks(mat,t,s.get_son(1));
				if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
					return false;
				else{
					if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
					if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
					return true;
				}
			}
		}
	}
}

template< template<typename> class LowRankMatrix, typename T >
void HMatrix<LowRankMatrix,T >::ComputeBlocks(const IMatrix<T>& mat){
    for(int b=0; b<MyBlocks.size(); b++) {
    	const Block& B = *(MyBlocks[b]);
   		const Cluster& t = B.tgt_();
			const Cluster& s = B.src_();
			const std::vector<int>& I = t.get_num();
			const std::vector<int>& J = s.get_num();
			if( B.IsAdmissible() ){
				MyFarFieldMats.emplace_back(I,J,reqrank);
				MyFarFieldMats.back().build(mat,t,s);
				if(MyFarFieldMats.back().rank_of()==-1){
					MyFarFieldMats.pop_back();
					if( s.IsLeaf() ){
						if( t.IsLeaf() ){
							MyNearFieldMats.emplace_back(mat,I,J);
						}
						else{
							bool b1 = UpdateBlocks(mat,t.get_son(0),s);
							bool b2 = UpdateBlocks(mat,t.get_son(1),s);
							if ((b1 != true) && (b2 != true))
								MyNearFieldMats.emplace_back(mat,I,J);
							else {
								if (b1 != true) MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
								if (b2 != true) MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
							}
						}
					}
					else{
						if( t.IsLeaf() ){
							bool b3 = UpdateBlocks(mat,t,s.get_son(0));
							bool b4 = UpdateBlocks(mat,t,s.get_son(1));
							if ((b3 != true) && (b4 != true))
								MyNearFieldMats.emplace_back(mat,I,J);
							else {
								if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
								if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
							}
						}
						else{
							if (t.get_num().size()>s.get_num().size()){
								bool b1 = UpdateBlocks(mat,t.get_son(0),s);
								bool b2 = UpdateBlocks(mat,t.get_son(1),s);
								if ((b1 != true) && (b2 != true))
									MyNearFieldMats.emplace_back(mat,I,J);
								else {
									if (b1 != true) MyNearFieldMats.emplace_back(mat,t.get_son(0).get_num(),J);
									if (b2 != true) MyNearFieldMats.emplace_back(mat,t.get_son(1).get_num(),J);
								}
							}
							else{
								bool b3 = UpdateBlocks(mat,t,s.get_son(0));
								bool b4 = UpdateBlocks(mat,t,s.get_son(1));
								if ((b3 != true) && (b4 != true))
									MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
								else {
									if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(0).get_num());
									if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.get_son(1).get_num());
								}
							}
						}
					}
				}
			}
			else {
				MyNearFieldMats.emplace_back(mat,I,J);
			}
		}
	}

template< template<typename> class LowRankMatrix, typename T >
std::vector<T> HMatrix<LowRankMatrix,T >::operator*(const std::vector<T>& x) const{
	int rankWorld, sizeWorld;
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);

	double time = MPI_Wtime();
	assert(nc==x.size());

	std::vector<T> result(nr,0);

	// Contribution champ lointain
	for(int b=0; b<MyFarFieldMats.size(); b++){
		const LowRankMatrix<T>&  M  = MyFarFieldMats[b];
		const std::vector<int>&        It = M.get_ir();
		const std::vector<int>&        Is = M.get_ic();


		std::vector<T> lhs(It.size());
		std::vector<T> rhs(Is.size());

		for (int i=0; i<Is.size(); i++)
			rhs[i] = x[Is[i]];

		lhs=M*rhs;

		for (int i=0; i<It.size(); i++)
			result[It[i]] += lhs[i];
	}

	// Contribution champ proche
	for(int b=0; b<MyNearFieldMats.size(); b++){
		const SubMatrix<T>&  M  = MyNearFieldMats[b];

		const std::vector<int>& It = M.get_ir();
		const std::vector<int>& Is = M.get_ic();

		std::vector<T> lhs(It.size());
		std::vector<T> rhs(Is.size());

		for (int i=0; i<Is.size(); i++)
			rhs[i] = x[Is[i]];

		lhs=M*rhs;

		for (int i=0; i<It.size(); i++)
			result[It[i]] += lhs[i];
	}

	/*
	vectCplx res;
	res.resize(f.size());
	int offset = 0;
	*/

	std::vector<T> snd;
	std::cout << rankWorld << std::endl;
	snd.resize(MasterClusters[rankWorld].size());

	std::vector<T> rcv;
	rcv.resize(result.size());

	std::vector<int> recvcounts(sizeWorld);
	std::vector<int>  displs(sizeWorld);

	displs[0] = 0;

	for (int i=0; i<sizeWorld; i++) {
		recvcounts[i] = MasterClusters[i].size();
		if (i > 0)
			displs[i] = displs[i-1] + recvcounts[i-1];
	}

	for (int i=0; i< MasterClusters[rankWorld].size(); i++) {
		snd[i] = result[MasterClusters[rankWorld][i]];
	}

	MPI_Allgatherv(&(snd.front()), recvcounts[rankWorld], wrapper_mpi<T>::mpi_type(), &(rcv.front()), &(recvcounts.front()), &(displs.front()), wrapper_mpi<T>::mpi_type(), comm);

	for (int i=0; i<sizeWorld; i++)
	for (int j=0; j< MasterClusters[i].size(); j++)
		result[MasterClusters[i][j]] = rcv[displs[i]+j];

	//MPI_Allreduce(MPI_IN_PLACE, &f.front(), f.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, comm);

// 	time = MPI_Wtime() - time;
//
// 	double meantime;
// 	double maxtime;
// 	MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
// 	MPI_Reduce(&time, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
// // 	if (rankWorld == 0) {
// 		meantime /= sizeWorld;
// 		std::cout << "MvProd: \t mean = " << meantime << ", \t max = " << maxtime << endl;
// 	}



// MPI_Allreduce(MPI_IN_PLACE, &result.front(), result.size(), wrapper_mpi<T>::mpi_type(), MPI_SUM, comm);
//
// return result;
	return result;

}

template< template<typename> class LowRankMatrix, typename T >
double HMatrix<LowRankMatrix,T >::compression() const{

	double mycomp = 0.;
	double size = nr*nc;


	for(int j=0; j<MyFarFieldMats.size(); j++){
		double nr   = MyFarFieldMats[j].nb_rows();
		double nc   = MyFarFieldMats[j].nb_cols();
		double rank = MyFarFieldMats[j].rank_of();
		mycomp += rank*(nr + nc)/size;
	}

	for(int j=0; j<MyNearFieldMats.size(); j++){
		double nr   = MyNearFieldMats[j].nb_rows();
		double nc   = MyNearFieldMats[j].nb_cols();
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
