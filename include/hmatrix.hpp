#ifndef HMATRIX_HPP
#define HMATRIX_HPP

#include <cassert>
#include <fstream>
#include "matrix.hpp"
#include "loading.hpp"
#include "lrmat.hpp"
namespace htool {


//===============================//
//     MATRICE HIERARCHIQUE      //
//===============================//

class HMatrix: public Parametres{

private:

	const VirtualMatrix& mat;
	const vectR3& xt;
	const vectR3& xs;
	const vectInt& tabt;
	const vectInt& tabs;


	std::vector<LowRankMatrix> FarFieldMat;
	std::vector<SubMatrix>     NearFieldMat;

	std::vector<Block*>		  Tasks;
	std::vector<Block*>		  MyBlocks;

	Block* BuildBlockTree(const Cluster&, const Cluster&);

	bool UpdateBlocks(const Cluster&, const Cluster&);

	void ScatterTasks();
	void ComputeBlocks();

	std::vector<LowRankMatrix> MyFarFieldMats;
	std::vector<SubMatrix>     MyNearFieldMats;

	const int& reqrank;

	std::map<std::string, double> stats;

public:

	HMatrix(const VirtualMatrix&, const vectR3&, const vectReal&, const vectInt&, const vectR3&, const vectReal&, const vectInt&, const int& reqrank=-1, const MPI_Comm& comm=MPI_COMM_WORLD); // To be used with two different clusters
	HMatrix(const VirtualMatrix&, const vectR3&, const vectReal&, const vectInt&, const int& reqrank=-1, const MPI_Comm& comm=MPI_COMM_WORLD); // To be used with one cluster

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

	friend void MvProd(vectCplx&, const HMatrix&, const vectCplx&);
	friend std::pair<double,double> MvProdMPI(vectCplx& f, const HMatrix& A, const vectCplx& x, const MPI_Comm& comm/*=MPI_COMM_WORLD*/);

	// 1- !!!
	friend Real CompressionRate(const HMatrix& hmat, const MPI_Comm& comm/*=MPI_COMM_WORLD*/);

	friend void Output(const HMatrix&, std::string filename);
	friend const LowRankMatrix& GetLowRankMatrix(HMatrix m, int i){
		assert(i<m.FarFieldMat.size());
		return m.FarFieldMat[i];}

	friend Real squared_absolute_error(const HMatrix& B, const VirtualMatrix& A, const MPI_Comm& comm=MPI_COMM_WORLD){
		Real myerr = 0;
		for(int j=0; j<B.MyFarFieldMats.size(); j++){
			SubMatrix subm(A,ir_(B.MyFarFieldMats[j]),ic_(B.MyFarFieldMats[j]));
			myerr += squared_absolute_error(B.MyFarFieldMats[j], subm);
		}
		for(int j=0; j<B.MyNearFieldMats.size(); j++){
			SubMatrix subm(A,ir_(B.MyNearFieldMats[j]),ic_(B.MyNearFieldMats[j]));
			myerr += squared_absolute_error(B.MyNearFieldMats[j], subm);
		}


		Real err = 0;
		MPI_Allreduce(&myerr, &err, 1, MPI_DOUBLE, MPI_SUM, comm);

		return err;
	}

};

HMatrix::HMatrix(const VirtualMatrix& mat0,
		 const vectR3& xt0, const vectReal& rt, const vectInt& tabt0,
		 const vectR3& xs0, const vectReal& rs, const vectInt& tabs0, const int& reqrank0, const MPI_Comm& comm):

mat(mat0), xt(xt0), xs(xs0), tabt(tabt0), tabs(tabs0),reqrank(reqrank0) {
	assert( nb_rows(mat)==tabt.size() && nb_cols(mat)==tabs.size() );

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

HMatrix::HMatrix(const VirtualMatrix& mat0,
		 const vectR3& xt0, const vectReal& rt, const vectInt& tabt0, const int& reqrank0, const MPI_Comm& comm):

mat(mat0), xt(xt0), xs(xt0), tabt(tabt0), tabs(tabt0),reqrank(reqrank0) {
	assert( nb_rows(mat)==tabt.size() && nb_cols(mat)==tabs.size() );

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

Block* HMatrix::BuildBlockTree(const Cluster& t, const Cluster& s){
	Block* B = new Block(t,s);
	int bsize = size(num_(t))*size(num_(s));
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
			Block* r1 = BuildBlockTree(son_(t,0),s);
			Block* r2 = BuildBlockTree(son_(t,1),s);
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
			Block* r3 = BuildBlockTree(t,son_(s,0));
			Block* r4 = BuildBlockTree(t,son_(s,1));
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
			if (size(num_(t))>size(num_(s))){
				Block* r1 = BuildBlockTree(son_(t,0),s);
				Block* r2 = BuildBlockTree(son_(t,1),s);
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
				Block* r3 = BuildBlockTree(t,son_(s,0));
				Block* r4 = BuildBlockTree(t,son_(s,1));
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
// 			Block* r1 = BuildBlockTree(son_(t,0),son_(s,0));
// 			Block* r2 = BuildBlockTree(son_(t,0),son_(s,1));
// 			Block* r3 = BuildBlockTree(son_(t,1),son_(s,0));
// 			Block* r4 = BuildBlockTree(son_(t,1),son_(s,1));
// 			if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL) && (r3 != NULL) && (r4 != NULL)) {
// 				delete r1;
// 				delete r2;
// 				delete r3;
// 				delete r4;
// 				return B;
// 			}
// 			else {
// 				if (r1 != NULL) Tasks.push_back(r1);
// 				if (r2 != NULL) Tasks.push_back(r2);
// 				if (r3 != NULL) Tasks.push_back(r3);
// 				if (r4 != NULL) Tasks.push_back(r4);
// 				return NULL;
// 			}
		}
	}
}

void HMatrix::ScatterTasks(){
	int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    for(int b=0; b<Tasks.size(); b++)
    	if (b%sizeWorld == rankWorld)
    		MyBlocks.push_back(Tasks[b]);
}

bool HMatrix::UpdateBlocks(const Cluster& t, const Cluster& s){
	const vectInt& I = num_(t);
	const vectInt& J = num_(s);
	int bsize = size(num_(t))*size(num_(s));
	Block B(t,s);
	B.ComputeAdmissibility();
	if( B.IsAdmissible() ){
		LowRankMatrix lrm(mat,I,J,t,s);
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
			bool b1 = UpdateBlocks(son_(t,0),s);
			bool b2 = UpdateBlocks(son_(t,1),s);
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
				return false;
			else {
				if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),J));
				if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),J));
				return true;
			}
		}
	}
	else{
		if( t.IsLeaf() ){
			bool b3 = UpdateBlocks(t,son_(s,0));
			bool b4 = UpdateBlocks(t,son_(s,1));
			if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
				return false;
			else{
				if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,0))));
				if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,1))));
				return true;
			}
		}
		else{
			
			if (size(num_(t))>size(num_(s))){
				bool b1 = UpdateBlocks(son_(t,0),s);
				bool b2 = UpdateBlocks(son_(t,1),s);
				if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true))
					return false;
				else {
					if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),J));
					if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),J));
					return true;
				}
			}
			else{
				bool b3 = UpdateBlocks(t,son_(s,0));
				bool b4 = UpdateBlocks(t,son_(s,1));
				if ((bsize <= maxblocksize) && (b3 != true) && (b4 != true))
					return false;
				else{
					if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,0))));
					if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,1))));
					return true;
				}
			}
		
// 			bool b1 = UpdateBlocks(son_(t,0),son_(s,0));
// 			bool b2 = UpdateBlocks(son_(t,0),son_(s,1));
// 			bool b3 = UpdateBlocks(son_(t,1),son_(s,0));
// 			bool b4 = UpdateBlocks(son_(t,1),son_(s,1));
// 			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true) && (b3 != true) && (b4 != true))
// 				return false;
// 			else {
// 				if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,0))));
// 				if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,1))));
// 				if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,0))));
// 				if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,1))));
// 				return true;
// 			}


		}
	}
}

void HMatrix::ComputeBlocks(){
    for(int b=0; b<MyBlocks.size(); b++) {
    	const Block& B = *(MyBlocks[b]);
   		const Cluster& t = tgt_(B);
		const Cluster& s = src_(B);
		const vectInt& I = num_(t);
		const vectInt& J = num_(s);
		if( B.IsAdmissible() ){
			LowRankMatrix lrm(mat,I,J,t,s);
			if(rank_of(lrm)!=-5){
				MyFarFieldMats.push_back(lrm);
			}
			else {
				if( s.IsLeaf() ){
					if( t.IsLeaf() ){
						MyNearFieldMats.push_back(SubMatrix(mat,I,J));
					}
					else{
						bool b1 = UpdateBlocks(son_(t,0),s);
						bool b2 = UpdateBlocks(son_(t,1),s);
						if ((b1 != true) && (b2 != true))
							MyNearFieldMats.push_back(SubMatrix(mat,I,J));
						else {
							if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),J));
							if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),J));
						}
					}
				}
				else{
					if( t.IsLeaf() ){
						bool b3 = UpdateBlocks(t,son_(s,0));
						bool b4 = UpdateBlocks(t,son_(s,1));
						if ((b3 != true) && (b4 != true))
							MyNearFieldMats.push_back(SubMatrix(mat,I,J));
						else {
							if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,0))));
							if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,1))));
						}
					}
					else{
						if (size(num_(t))>size(num_(s))){
							bool b1 = UpdateBlocks(son_(t,0),s);
							bool b2 = UpdateBlocks(son_(t,1),s);
							if ((b1 != true) && (b2 != true))
								MyNearFieldMats.push_back(SubMatrix(mat,I,J));
							else {
								if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),J));
								if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),J));
							}
						}
						else{
							bool b3 = UpdateBlocks(t,son_(s,0));
							bool b4 = UpdateBlocks(t,son_(s,1));
							if ((b3 != true) && (b4 != true))
								MyNearFieldMats.push_back(SubMatrix(mat,I,J));
							else {
								if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,0))));
								if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,I,num_(son_(s,1))));
							}
						}
							
							
							
						
						
// 						bool b1 = UpdateBlocks(son_(t,0),son_(s,0));
// 						bool b2 = UpdateBlocks(son_(t,0),son_(s,1));
// 						bool b3 = UpdateBlocks(son_(t,1),son_(s,0));
// 						bool b4 = UpdateBlocks(son_(t,1),son_(s,1));
// 						if ((b1 != true) && (b2 != true) && (b3 != true) && (b4 != true))
// 							MyNearFieldMats.push_back(SubMatrix(mat,I,J));
// 						else {
// 							if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,0))));
// 							if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,1))));
// 							if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,0))));
// 							if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,1))));
// 						}

					}
				}
			}
		}
		else {
			MyNearFieldMats.push_back(SubMatrix(mat,I,J));
		}
    }
}



// void Output(const HMatrix& hmat, string filename){
// 	//string path=GetOutputPath()+"/"+filename;
//     string path=filename;
//
// 	ofstream outputfile(path.c_str());
//
// 	if (!outputfile){
// 		cerr << "Output file cannot be created in "+path << endl;
// 		exit(1);
// 	}
// 	else{
//
// 		const std::vector<LowRankMatrix>& FarFieldMat  = hmat.FarFieldMat;
// 		//		const std::vector<SubMatrix>&     NearFieldMat = hmat.NearFieldMat;
//
// 		for(int i=0; i<FarFieldMat.size(); i++){
//
// 			vectInt ir = ir_(FarFieldMat[i]);
// 			vectInt ic = ic_(FarFieldMat[i]);
// 			Real local_compression = CompressionRate(FarFieldMat[i]);
//
// 			for (int j=0;j<ir.size();j++){
// 				for (int k=0;k<ic.size();k++){
// 					outputfile<<ir[j]<<" "<<ic[k]<<" "<<local_compression<<endl;
// 				}
// 			}
// 		}
//
// 		//		for(int i=0; i<NearFieldMat.size(); i++){
// 		//			vectInt ir = ir_(NearFieldMat[i]);
// 		//			vectInt ic = ic_(NearFieldMat[i]);
// 		//			for (int j=0;j<ir.size();j++){
// 		//				for (int k=0;k<ic.size();k++){
// 		//					outputfile<<ir[j]<<" "<<ic[k]<<" "<<1<<endl;
// 		//				}
// 		//			}
// 		//		}
// 	}
//
// }




void MvProd(vectCplx& f, const HMatrix& A, const vectCplx& x){
	assert(size(f)==size(x)); fill(f,0.);

	const std::vector<LowRankMatrix>&    FarFieldMat  = A.FarFieldMat;
	const std::vector<SubMatrix>&        NearFieldMat = A.NearFieldMat;

	// Contribution champ lointain
	for(int b=0; b<FarFieldMat.size(); b++){
		const LowRankMatrix&  M  = FarFieldMat[b];
		const vectInt&        It = ir_(M);
		const vectInt&        Is = ic_(M);

		ConstSubVectCplx xx(x,Is);
		SubVectCplx ff(f,It);
		MvProd(ff,M,xx);
	}

	// Contribution champ proche
	for(int b=0; b<NearFieldMat.size(); b++){
		const SubMatrix&  M  = NearFieldMat[b];

		const vectInt& It = ir_ (M);
		const vectInt& Is = ic_ (M);

		ConstSubVectCplx xx(x,Is);
		SubVectCplx ff(f,It);
		MvProd(ff,M,xx);
	}

}
std::pair<double,double> MvProdMPI(vectCplx& f, const HMatrix& A, const vectCplx& x, const MPI_Comm& comm=MPI_COMM_WORLD){
	int rankWorld, sizeWorld;
   	MPI_Comm_size(comm, &sizeWorld);
   	MPI_Comm_rank(comm, &rankWorld);

	double time = MPI_Wtime();

	assert(nb_rows(A.mat)==size(f) && nb_cols(A.mat)==size(x)); fill(f,0.);
	const std::vector<LowRankMatrix>&    MyFarFieldMats  = A.MyFarFieldMats;
	const std::vector<SubMatrix>&        MyNearFieldMats = A.MyNearFieldMats;

	// Contribution champ lointain
	for(int b=0; b<MyFarFieldMats.size(); b++){
		const LowRankMatrix&  M  = MyFarFieldMats[b];
		const vectInt&        It = ir_(M);
		const vectInt&        Is = ic_(M);

		ConstSubVectCplx xx(x,Is);
		SubVectCplx ff(f,It);
		MvProd(ff,M,xx);
	}

	// Contribution champ proche
	for(int b=0; b<MyNearFieldMats.size(); b++){
		const SubMatrix&  M  = MyNearFieldMats[b];

		const vectInt& It = ir_ (M);
		const vectInt& Is = ic_ (M);

		std::vector<Cplx> lhs(size(It));
		std::vector<Cplx> rhs(size(Is));

		for (int i=0; i<size(Is); i++)
			rhs[i] = x[Is[i]];

		MvProd(lhs,M,rhs);

		for (int i=0; i<size(It); i++)
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
Real CompressionRate(const HMatrix& hmat, const MPI_Comm& comm=MPI_COMM_WORLD){

	Real mycomp = 0.;
	Real size = ( (hmat.tabt).size() )*( (hmat.tabs).size() );
	const std::vector<LowRankMatrix>& MyFarFieldMats  = hmat.MyFarFieldMats;
	const std::vector<SubMatrix>&     MyNearFieldMats = hmat.MyNearFieldMats;

	for(int j=0; j<MyFarFieldMats.size(); j++){
		Real nr   = nb_rows(MyFarFieldMats[j]);
		Real nc   = nb_cols(MyFarFieldMats[j]);
		Real rank = rank_of(MyFarFieldMats[j]);
		mycomp += rank*(nr + nc)/size;
	}

	for(int j=0; j<MyNearFieldMats.size(); j++){
		Real nr   = nb_rows(MyNearFieldMats[j]);
		Real nc   = nb_cols(MyNearFieldMats[j]);
		mycomp += nr*nc/size;
	}

	Real comp = 0;
	MPI_Allreduce(&mycomp, &comp, 1, MPI_DOUBLE, MPI_SUM, comm);

	return 1-comp;
}

void print_stats(const HMatrix& A, const MPI_Comm& comm=MPI_COMM_WORLD){
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
