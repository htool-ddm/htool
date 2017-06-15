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
template< template<typename> class LowRankMatrix, typename T >
class HMatrix: public Parametres{

private:
	// Data members
	int nr;
	int nc;
	int reqrank;

	std::vector<Block*>		  Tasks;
	std::vector<Block*>		  MyBlocks;

	std::vector<LowRankMatrix<T> > MyFarFieldMats;
	std::vector<SubMatrix<T> >     MyNearFieldMats;

	std::map<std::string, double> stats;

	MPI_Comm comm;

	// Internal methods
	Block* BuildBlockTree(const Cluster&, const Cluster&);
	bool UpdateBlocks(const IMatrix<T>&mat ,const Cluster&, const Cluster&);
	void ScatterTasks();
	void ComputeBlocks(const IMatrix<T>& mat);

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
	std::vector<LowRankMatrix<T> > get_MyFarFieldMats() const {return MyFarFieldMats;}
	std::vector<SubMatrix<T> > get_MyNearFieldMats() const {return MyNearFieldMats;}

	std::map<std::string,double>& get_stats () const { return stats;}
	void add_stats(const std::string& keyname, const double& value){stats[keyname]=value;}
	void print_stats();

	int nb_lrmats(){
		int res=MyFarFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;
	}
	int nb_densemats(){
		int res=MyNearFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;}

	std::vector<T> operator*( const std::vector<T>& x) const;

	// // 1- !!!
	// double CompressionRate(const MPI_Comm& comm=MPI_COMM_WORLD) const;
	//
	// friend void Output(const HMatrix&, std::string filename);
	// friend const LowRankMatrix<T>& GetLowRankMatrix(HMatrix m, int i){
	// 	assert(i<m.FarFieldMats.size());
	// 	return m.FarFieldMats[i];}
	//
	// friend double squared_absolute_error(const HMatrix& B, const VirtualMatrix<T>& A, const MPI_Comm& comm=MPI_COMM_WORLD){
	// 	double myerr = 0;
	// 	for(int j=0; j<B.MyFarFieldMats.size(); j++){
	// 		SubMatrix<T> subm(A,ir_(B.MyFarFieldMats[j]),ic_(B.MyFarFieldMats[j]));
	// 		myerr += squared_absolute_error(B.MyFarFieldMats[j], subm);
	// 	}
	// 	for(int j=0; j<B.MyNearFieldMats.size(); j++){
	// 		SubMatrix<T> subm(A,ir_(B.MyNearFieldMats[j]),ic_(B.MyNearFieldMats[j]));
	// 		myerr += squared_absolute_error(B.MyNearFieldMats[j], subm);
	// 	}
	//
	//
	// 	double err = 0;
	// 	MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, comm);
	//
	// 	return err;
	// }

};

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
	myttime[0] = MPI_Wtime() - time;

	// Construction arbre des blocs
	time = MPI_Wtime();
	if (rankWorld==0){
	std::cout << "start" <<std::endl;
	}
	Block* B = BuildBlockTree(t,s);
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
bool HMatrix<LowRankMatrix,T >::UpdateBlocks(const IMatrix<T>& mat,const Cluster& t, const Cluster& s){
	const std::vector<int>& I = t.num_();
	const std::vector<int>& J = s.num_();
	int bsize = t.num_().size()*s.num_().size();
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
			bool b1 = UpdateBlocks(mat,t.son_(0),s);
			bool b2 = UpdateBlocks(mat,t.son_(1),s);
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
				return false;
			else {
				if (b1 != true) MyNearFieldMats.emplace_back(mat,t.son_(0).num_(),J);
				if (b2 != true) MyNearFieldMats.emplace_back(mat,t.son_(1).num_(),J);
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			bool b3 = UpdateBlocks(mat,t,s.son_(0));
			bool b4 = UpdateBlocks(mat,t,s.son_(1));
			if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
				return false;
			else{
				if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(0).num_());
				if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(1).num_());
				return true;
			}
		}
		else{

			if (t.num_().size()>s.num_().size()){
				bool b1 = UpdateBlocks(mat,t.son_(0),s);
				bool b2 = UpdateBlocks(mat,t.son_(1),s);
				if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
					return false;
				else {
					if (b1 != true) MyNearFieldMats.emplace_back(mat,t.son_(0).num_(),J);
					if (b2 != true) MyNearFieldMats.emplace_back(mat,t.son_(1).num_(),J);
					return true;
				}
			}
			else{
				bool b3 = UpdateBlocks(mat,t,s.son_(0));
				bool b4 = UpdateBlocks(mat,t,s.son_(1));
				if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
					return false;
				else{
					if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(0).num_());
					if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(1).num_());
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
			const std::vector<int>& I = t.num_();
			const std::vector<int>& J = s.num_();
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
							bool b1 = UpdateBlocks(mat,t.son_(0),s);
							bool b2 = UpdateBlocks(mat,t.son_(1),s);
							if ((b1 != true) && (b2 != true))
								MyNearFieldMats.emplace_back(mat,I,J);
							else {
								if (b1 != true) MyNearFieldMats.emplace_back(mat,t.son_(0).num_(),J);
								if (b2 != true) MyNearFieldMats.emplace_back(mat,t.son_(1).num_(),J);
							}
						}
					}
					else{
						if( t.IsLeaf() ){
							bool b3 = UpdateBlocks(mat,t,s.son_(0));
							bool b4 = UpdateBlocks(mat,t,s.son_(1));
							if ((b3 != true) && (b4 != true))
								MyNearFieldMats.emplace_back(mat,I,J);
							else {
								if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(0).num_());
								if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(1).num_());
							}
						}
						else{
							if (t.num_().size()>s.num_().size()){
								bool b1 = UpdateBlocks(mat,t.son_(0),s);
								bool b2 = UpdateBlocks(mat,t.son_(1),s);
								if ((b1 != true) && (b2 != true))
									MyNearFieldMats.emplace_back(mat,I,J);
								else {
									if (b1 != true) MyNearFieldMats.emplace_back(mat,t.son_(0).num_(),J);
									if (b2 != true) MyNearFieldMats.emplace_back(mat,t.son_(1).num_(),J);
								}
							}
							else{
								bool b3 = UpdateBlocks(mat,t,s.son_(0));
								bool b4 = UpdateBlocks(mat,t,s.son_(1));
								if ((b3 != true) && (b4 != true))
									MyNearFieldMats.push_back(SubMatrix<T>(mat,I,J));
								else {
									if (b3 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(0).num_());
									if (b4 != true) MyNearFieldMats.emplace_back(mat,I,s.son_(1).num_());
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
std::cout << MyFarFieldMats.size() <<" "<<MyNearFieldMats.size()<<std::endl;
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
	time = MPI_Wtime() - time;

	double meantime;
	double maxtime;
	MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	MPI_Reduce(&time, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
// 	if (rankWorld == 0) {
		meantime /= sizeWorld;
// 		std::cout << "MvProd: \t mean = " << meantime << ", \t max = " << maxtime << endl;
// 	}

	MPI_Allreduce(MPI_IN_PLACE, &result.front(), result.size(), wrapper_mpi<T>::mpi_type(), MPI_SUM, comm);

	return result;
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
