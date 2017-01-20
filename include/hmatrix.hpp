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
	
	
	vector<LowRankMatrix> FarFieldMat;
	vector<SubMatrix>     NearFieldMat;
	
	vector<Block*>		  Tasks;
	vector<Block*>		  MyBlocks;
	
	Block* BuildBlockTree(const Cluster&, const Cluster&);
	
	bool UpdateBlocks(const Cluster&, const Cluster&);
	
	void ScatterTasks();
	void ComputeBlocks();
	
	vector<LowRankMatrix> MyFarFieldMats;
	vector<SubMatrix>     MyNearFieldMats;
	
	const int& reqrank;
	
public:
	
	HMatrix(const VirtualMatrix&, const vectR3&, const vectReal&, const vectInt&, const vectR3&, const vectReal&, const vectInt&, const int& reqrank=-1, const MPI_Comm& comm=MPI_COMM_WORLD); // To be used with two different clusters
	HMatrix(const VirtualMatrix&, const vectR3&, const vectReal&, const vectInt&, const int& reqrank=-1, const MPI_Comm& comm=MPI_COMM_WORLD); // To be used with one cluster
	//	friend void DisplayPartition(const HMatrix&, char const* const);

	~HMatrix() {
		for (int i=0; i<Tasks.size(); i++)
			delete Tasks[i];
	}

	friend const int& nb_rows(const HMatrix& A){ return nb_rows(A.mat);}

	friend const int& nb_cols(const HMatrix& A){ return nb_cols(A.mat);}
	
	friend const int nb_lrmats(const HMatrix& A, const MPI_Comm& comm=MPI_COMM_WORLD){ int res=A.MyFarFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;}
    
	friend const int nb_densemats(const HMatrix& A, const MPI_Comm& comm=MPI_COMM_WORLD){ int res=A.MyNearFieldMats.size(); MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, comm); return res;}
    
	friend void MvProd(vectCplx&, const HMatrix&, const vectCplx&);
	
	friend void MvProdMPI(vectCplx& f, const HMatrix& A, const vectCplx& x, const MPI_Comm& comm=MPI_COMM_WORLD){
		int rankWorld, sizeWorld;
    	MPI_Comm_size(comm, &sizeWorld);
    	MPI_Comm_rank(comm, &rankWorld);
		
		double time = MPI_Wtime();
		
		assert(nb_rows(A.mat)==size(f) && nb_cols(A.mat)==size(x)); fill(f,0.);
// 			for (int i =0;i<size(f);i++){
// 				std::cout << f[i] << std::endl;
// 			}
		const vector<LowRankMatrix>&    MyFarFieldMats  = A.MyFarFieldMats;
		const vector<SubMatrix>&        MyNearFieldMats = A.MyNearFieldMats;
		
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
			
			//ConstSubVectCplx xx(x,Is);
			//SubVectCplx ff(f,It);
			
			std::vector<Cplx> lhs(size(It));
			std::vector<Cplx> rhs(size(Is));
			
			for (int i=0; i<size(Is); i++)
				rhs[i] = x[Is[i]];

			//std::cout << lhs.size() << " " << rhs.size() << endl;

			MvProd(lhs,M,rhs);
			
			for (int i=0; i<size(It); i++)
				f[It[i]] += lhs[i];
		}

		time = MPI_Wtime() - time;
		
		double meantime;
		double maxtime;
		MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
		MPI_Reduce(&time, &meantime, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
		if (rankWorld == 0) {
			meantime /= sizeWorld;
			cout << "MvProd: \t mean = " << meantime << ", \t max = " << maxtime << endl;
		}
		
		MPI_Allreduce(MPI_IN_PLACE, &f.front(), f.size(), MPI_DOUBLE_COMPLEX, MPI_SUM, comm);	
	}
	
	// 1- !!!
	friend Real CompressionRate(const HMatrix& hmat, const MPI_Comm& comm=MPI_COMM_WORLD){
	
		Real mycomp = 0.;
		Real size = ( (hmat.tabt).size() )*( (hmat.tabs).size() );
		const vector<LowRankMatrix>& MyFarFieldMats  = hmat.MyFarFieldMats;
		const vector<SubMatrix>&     MyNearFieldMats = hmat.MyNearFieldMats;
	
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
	
	friend void Output(const HMatrix&, string filename);
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

Block* HMatrix::BuildBlockTree(const Cluster& t, const Cluster& s){
	//std::stack<Block*> st;
	//st.push(new Block(tgt,src));
	
	//while (!st.empty()){
	//	Block* curr = st.top();
	//	st.pop();
	{
		Block* B = new Block(t,s);
		int bsize = size(num_(t))*size(num_(s));
		B->ComputeAdmissibility();
		if( B->IsAdmissible() ){
			Tasks.push_back(B);
			return NULL;
		//int nr = num_(t).size();
		//int nc = num_(s).size();
		//const vectInt& I = num_(t);
		//const vectInt& J = num_(s);
		//SubMatrix submat = SubMatrix(mat,I,J);
		//LowRankMatrix lrm(submat,I,J,t,s,reqrank);
		//if((nr+nc)<(nr*nc)){ // If the flag (given by the lrmatrix constructor) is different from -5
			//FarFieldMat.push_back(lrm); // we keep the computed ACA approximation of the block	
			//return; // and we terminate this step of the recursion
		//}
        // otherwise, we go on with the decomposition of the block (indeed there isn't a else!);
        // rank_of(lrm)=-5 happens when the required precision given by epsilon can't be reached with an advantageous rank (not advantageous in terms of the complexity of the matrix-vector product)
		}
		else if( s.IsLeaf() ){
			if( t.IsLeaf() ){
				//const vectInt& I = num_(t);
				//const vectInt& J = num_(s);
				//NearFieldMat.push_back(SubMatrix(mat,I,J));
				//Tasks.push_back(B);
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
				//st.push(new Block(son_(t,0),s));
				//st.push(new Block(son_(t,1),s));
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
				//st.push(new Block(t,son_(s,0)));
				//st.push(new Block(t,son_(s,1)));
			}
			else{
				Block* r1 = BuildBlockTree(son_(t,0),son_(s,0));
				Block* r2 = BuildBlockTree(son_(t,0),son_(s,1));
				Block* r3 = BuildBlockTree(son_(t,1),son_(s,0));
				Block* r4 = BuildBlockTree(son_(t,1),son_(s,1));
				if ((bsize <= maxblocksize) && (r1 != NULL) && (r2 != NULL) && (r3 != NULL) && (r4 != NULL)) {
					delete r1;
					delete r2;
					delete r3;
					delete r4;
					return B;
				}
				else {
					if (r1 != NULL) Tasks.push_back(r1);					
					if (r2 != NULL) Tasks.push_back(r2);
					if (r3 != NULL) Tasks.push_back(r3);					
					if (r4 != NULL) Tasks.push_back(r4);	
					return NULL;
				}				
				//st.push(new Block(son_(t,0),son_(s,0)));
				//st.push(new Block(son_(t,0),son_(s,1)));
				//st.push(new Block(son_(t,1),son_(s,0)));
				//st.push(new Block(son_(t,1),son_(s,1)));
			}
		}
		//delete(curr);
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
		//SubMatrix submat = SubMatrix(mat,I,J);
		LowRankMatrix lrm(mat,I,J,t,s);
		if(rank_of(lrm)!=-5){
			MyFarFieldMats.push_back(lrm);
			return true;
		}
	}
	if( s.IsLeaf() ){
		if( t.IsLeaf() ){
// 			MyNearFieldMats.push_back(SubMatrix(mat,I,J));
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
			bool b1 = UpdateBlocks(son_(t,0),son_(s,0));
			bool b2 = UpdateBlocks(son_(t,0),son_(s,1));
			bool b3 = UpdateBlocks(son_(t,1),son_(s,0));
			bool b4 = UpdateBlocks(son_(t,1),son_(s,1));
			if ((bsize <= maxblocksize) && (b1 != true) && (b2 != true) && (b3 != true) && (b4 != true)) 
				return false;
			else {
				if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,0))));
				if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,1))));
				if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,0))));
				if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,1))));
				return true;
			}
			
			
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
			//SubMatrix submat = SubMatrix(mat,I,J);
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
						bool b1 = UpdateBlocks(son_(t,0),son_(s,0));
						bool b2 = UpdateBlocks(son_(t,0),son_(s,1));
						bool b3 = UpdateBlocks(son_(t,1),son_(s,0));
						bool b4 = UpdateBlocks(son_(t,1),son_(s,1));
						if ((b1 != true) && (b2 != true) && (b3 != true) && (b4 != true)) 
							MyNearFieldMats.push_back(SubMatrix(mat,I,J));
						else {
							if (b1 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,0))));
							if (b2 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,0)),num_(son_(s,1))));
							if (b3 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,0))));
							if (b4 != true) MyNearFieldMats.push_back(SubMatrix(mat,num_(son_(t,1)),num_(son_(s,1))));
						}
						
					}
				}
			}
		}
		else {
			MyNearFieldMats.push_back(SubMatrix(mat,I,J));
		}
    }	
}

HMatrix::HMatrix(const VirtualMatrix& mat0,
		 const vectR3& xt0, const vectReal& rt, const vectInt& tabt0,
		 const vectR3& xs0, const vectReal& rs, const vectInt& tabs0, const int& reqrank0, const MPI_Comm& comm):

mat(mat0), xt(xt0), xs(xs0), tabt(tabt0), tabs(tabs0),reqrank(reqrank0) {
	assert( nb_rows(mat)==tabt.size() && nb_cols(mat)==tabs.size() );
	
	int rankWorld, sizeWorld;
    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    vector<double> myttime(4), maxtime(4), meantime(4);
	
	
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
	if (rankWorld == 0) {
		meantime /= sizeWorld;
		cout << "Cluster tree: \t mean = " << meantime[0] << ", \t max = " << maxtime[0] << endl;
		cout << "Block tree: \t mean = " << meantime[1] << ", \t max = " << maxtime[1] << endl;
		cout << "Scatter tasks: \t mean = " << meantime[2] << ", \t max = " << maxtime[2] << endl;
		cout << "Blocks: \t mean = " << meantime[3] << ", \t max = " << maxtime[3] << endl;
	}
	
}

HMatrix::HMatrix(const VirtualMatrix& mat0,
		 const vectR3& xt0, const vectReal& rt, const vectInt& tabt0, const int& reqrank0, const MPI_Comm& comm):

mat(mat0), xt(xt0), xs(xt0), tabt(tabt0), tabs(tabt0),reqrank(reqrank0) {
	assert( nb_rows(mat)==tabt.size() && nb_cols(mat)==tabs.size() );
	
	int rankWorld, sizeWorld;
    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    vector<double> myttime(4), maxtime(4), meantime(4);
	
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
	if (rankWorld == 0) {
		meantime /= sizeWorld;
		cout << "Cluster tree: \t mean = " << meantime[0] << ", \t max = " << maxtime[0] << endl;
		cout << "Block tree: \t mean = " << meantime[1] << ", \t max = " << maxtime[1] << endl;
		cout << "Scatter tasks: \t mean = " << meantime[2] << ", \t max = " << maxtime[2] << endl;
		cout << "Blocks: \t mean = " << meantime[3] << ", \t max = " << maxtime[3] << endl;
	}
}

void Output(const HMatrix& hmat, string filename){
	//string path=GetOutputPath()+"/"+filename;
    string path=filename;
	
	ofstream outputfile(path.c_str());
	
	if (!outputfile){
		cerr << "Output file cannot be created in "+path << endl;
		exit(1);
	}
	else{
		
		const vector<LowRankMatrix>& FarFieldMat  = hmat.FarFieldMat;
		//		const vector<SubMatrix>&     NearFieldMat = hmat.NearFieldMat;
		
		for(int i=0; i<FarFieldMat.size(); i++){
			
			vectInt ir = ir_(FarFieldMat[i]);
			vectInt ic = ic_(FarFieldMat[i]);
			Real local_compression = CompressionRate(FarFieldMat[i]);
			
			for (int j=0;j<ir.size();j++){
				for (int k=0;k<ic.size();k++){
					outputfile<<ir[j]<<" "<<ic[k]<<" "<<local_compression<<endl;
				}
			}
		}
		
		//		for(int i=0; i<NearFieldMat.size(); i++){
		//			vectInt ir = ir_(NearFieldMat[i]);
		//			vectInt ic = ic_(NearFieldMat[i]);
		//			for (int j=0;j<ir.size();j++){
		//				for (int k=0;k<ic.size();k++){
		//					outputfile<<ir[j]<<" "<<ic[k]<<" "<<1<<endl;
		//				}
		//			}
		//		}
	}
	
}




// Representation graphique de la partition en bloc
//void DisplayPartition(const HMatrix& hmat, char const * const name){
//
//	const vector<Block>& FarField = hmat.FarField;
//	const vectR3& xt = hmat.xt;
//	const vectR3& xs = hmat.xs;;
//
//	// Representation graphique
//	const int  Ns = xs.size();
//	const Real ds = 1./Real(Ns-1);
//	const int  Nt = xt.size();
//	const Real dt = 1./Real(Nt-1);
//
//	ofstream file; file.open(name);
//	for(int j=0; j<FarField.size(); j++){
//
//		const Cluster& t = tgt_(FarField[j]);
//		const vectInt& It = num_(t);
//		Real at = (It[0]-0.5)*dt;
//		Real bt = (It[It.size()-1]+0.5)*dt;
//
//		const Cluster& s = src_(FarField[j]);
//		const vectInt& Is = num_(s);
//		Real as = (Is[0]-0.5)*ds;
//		Real bs = (Is[Is.size()-1]+0.5)*ds;
//
//		file << as << "\t" << at << "\n";
//		file << bs << "\t" << at << "\n";
//		file << bs << "\t" << bt << "\n";
//		file << as << "\t" << bt << "\n";
//		file << as << "\t" << at << "\n";
//		file << endl;
//
//	}
//	file.close();
//
//}


void MvProd(vectCplx& f, const HMatrix& A, const vectCplx& x){
	assert(size(f)==size(x)); fill(f,0.);
	
	const vector<LowRankMatrix>&    FarFieldMat  = A.FarFieldMat;
	const vector<SubMatrix>&        NearFieldMat = A.NearFieldMat;
	
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
}
#endif
