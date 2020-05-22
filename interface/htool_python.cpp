#include <Python.h>
#include "hmatrix.hpp"
#include "multihmatrix.hpp"

using namespace std;
using namespace htool;

#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#else
typedef double K;
#endif

extern "C" {
unsigned short scalar = std::is_same<K, double>::value ? 0 : 1;

void* HMatrixCreateSym(double* pts, int n, void (*getcoef)(int,int,K*),bool symmetric=false) {

  vector<R3> p(n);
	for(int j=0; j<n; j++){
    p[j][0] = pts[3*j];
    p[j][1] = pts[3*j+1];
    p[j][2] = pts[3*j+2];
	}

  // Matrix
	MyMatrix A(p,getcoef);

  // Hmatrix
	HMatrix<K,sympartialACA,GeometricClustering>* H = new HMatrix<K,sympartialACA,GeometricClustering>(A,p,symmetric);

  return H;
}

void* HMatrixCreate(double* pts1, int m, double* pts2, int n, void (*getcoef)(int,int,K*)) {

  vector<R3> p1(m),p2(n);
	for(int j=0; j<m; j++){
    p1[j][0] = pts1[3*j];
    p1[j][1] = pts1[3*j+1];
    p1[j][2] = pts1[3*j+2];
	}

	for(int j=0; j<n; j++){
	p2[j][0] = pts2[3*j];
	p2[j][1] = pts2[3*j+1];
	p2[j][2] = pts2[3*j+2];
	}

  // Matrix
	MyMatrix A(p1,p2,getcoef);

  // Hmatrix
	HMatrix<K,sympartialACA,GeometricClustering>* H = new HMatrix<K,sympartialACA,GeometricClustering>(A,p1,p2);

  return H;
}

void* HMatrixCreatewithsubmatSym(double* pts, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*),bool symmetric=false) {

  vector<R3> p(n);
	for(int j=0; j<n; j++){
    p[j][0] = pts[3*j];
    p[j][1] = pts[3*j+1];
    p[j][2] = pts[3*j+2];
	}

  // Matrix
	MyMatrixwithsubmat A(p,getsubmatrix);

  // Hmatrix
	HMatrix<K,sympartialACA,GeometricClustering>* H = new HMatrix<K,sympartialACA,GeometricClustering>(A,p,symmetric);

  return H;
}

void* HMatrixCreatewithsubmat(double* pts1, int m, double* pts2, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*)) {

  vector<R3> p1(m),p2(n);
	for(int j=0; j<m; j++){
    p1[j][0] = pts1[3*j];
    p1[j][1] = pts1[3*j+1];
    p1[j][2] = pts1[3*j+2];
	}

	for(int j=0; j<n; j++){
	p2[j][0] = pts2[3*j];
	p2[j][1] = pts2[3*j+1];
	p2[j][2] = pts2[3*j+2];
	}

  // Matrix
	MyMatrixwithsubmat A(p1,p2,getsubmatrix);

  // Hmatrix
	HMatrix<K,sympartialACA,GeometricClustering>* H = new HMatrix<K,sympartialACA,GeometricClustering>(A,p1,p2);


  return H;
}

void printinfos(void* H) {
    reinterpret_cast<HMatrix<K,sympartialACA,GeometricClustering>*>(H)->print_infos();
}

void mvprod(void* H, K* x, K* Ax) {
    reinterpret_cast<HMatrix<K,sympartialACA,GeometricClustering>*>(H)->mvprod_global(x,Ax);
}

int getnlrmat(void* H) {
    return reinterpret_cast<HMatrix<K,partialACA,GeometricClustering>*>(H)->get_nlrmat();
}

int getndmat(void* H) {
    return reinterpret_cast<HMatrix<K,partialACA,GeometricClustering>*>(H)->get_ndmat();
}

int nbrows(void* H) {
    return reinterpret_cast<HMatrix<K,partialACA,GeometricClustering>*>(H)->nb_rows();
}

int nbcols(void* H) {
    return reinterpret_cast<HMatrix<K,sympartialACA,GeometricClustering>*>(H)->nb_cols();
}

void setepsilon(double eps) { SetEpsilon(eps); }
void seteta(double eta) { SetEta(eta); }
void setminclustersize(int m) { SetMinClusterSize(m); }
void setmaxblocksize(int m) { SetMaxBlockSize(m); }

void getpattern(void* pH, int* buf) {
	HMatrix<K,sympartialACA,GeometricClustering>* H = reinterpret_cast<HMatrix<K,sympartialACA,GeometricClustering>*>(pH);

	const std::vector<sympartialACA<K,GeometricClustering>*>& lrmats = H->get_MyFarFieldMats();
	const std::vector<SubMatrix<K>*>& dmats = H->get_MyNearFieldMats();

	int nb = dmats.size() + lrmats.size();

	int sizeworld = H->get_sizeworld();
	int rankworld = H->get_rankworld();

	int nbworld[sizeworld];
	MPI_Allgather(&nb, 1, MPI_INT, nbworld, 1, MPI_INT, H->get_comm());
	int nbg = 0;
	for (int i=0; i<sizeworld; i++) {
		nbg += nbworld[i];
	}

	for (int i=0;i<dmats.size();i++) {
		const SubMatrix<K>& l = *(dmats[i]);
		buf[5*i] = l.get_offset_i();
		buf[5*i+1] = l.nb_rows();
		buf[5*i+2] = l.get_offset_j();
		buf[5*i+3] = l.nb_cols();
		buf[5*i+4] = -1;
	}

	for (int i=0;i<lrmats.size();i++) {
		const LowRankMatrix<K,GeometricClustering>& l = *(lrmats[i]);
		buf[5*(dmats.size()+i)] = l.get_offset_i();
		buf[5*(dmats.size()+i)+1] = l.nb_rows();
		buf[5*(dmats.size()+i)+2] = l.get_offset_j();
		buf[5*(dmats.size()+i)+3] = l.nb_cols();
		buf[5*(dmats.size()+i)+4] = l.rank_of();
	}

	int displs[sizeworld];
	int recvcounts[sizeworld];
	displs[0] = 0;

	for (int i=0; i<sizeworld; i++) {
		recvcounts[i] = 5*nbworld[i];
		if (i > 0)	displs[i] = displs[i-1] + recvcounts[i-1];
	}
	MPI_Gatherv(rankworld==0?MPI_IN_PLACE:buf, recvcounts[rankworld], MPI_INT, buf, recvcounts, displs, MPI_INT, 0, H->get_comm());
}

void get_target_cluster(void* pH, double* x,  double* output, int depth) {
	HMatrix<K,sympartialACA,GeometricClustering>* H = reinterpret_cast<HMatrix<K,sympartialACA,GeometricClustering>*>(pH);

	int sizeworld = H->get_sizeworld();
	int rankworld = H->get_rankworld();

	if (rankworld==0){
		Cluster<GeometricClustering> const * root = &(H->get_cluster_tree_t());
		std::stack< Cluster<GeometricClustering> const *> s;
		s.push(root);

		int size = root->get_size();

		// Permuted geometric points
		for(int i = 0; i<size; ++i) {
			output[i  ]     = x[3*root->get_perm(i)];
			output[i+size]  = x[3*root->get_perm(i)+1];
			output[i+size*2]= x[3*root->get_perm(i)+2];
		}

		int counter = 0;
		while(!s.empty()){
			Cluster<GeometricClustering> const * curr = s.top();
			s.pop();

			if (depth == curr->get_depth()){
				std::fill_n(&(output[3*size+curr->get_offset()]),curr->get_size(),counter);
				counter+=1;
			}

			// Recursion
			if (!curr->IsLeaf()){
				
				for (int p=0;p<curr->get_nb_sons();p++){
					s.push(&(curr->get_son(p)));
				}
			}

		}
	}
}

void get_source_cluster(void* pH, double* x,  double* output, int depth) {
	HMatrix<K,sympartialACA,GeometricClustering>* H = reinterpret_cast<HMatrix<K,sympartialACA,GeometricClustering>*>(pH);

	int sizeworld = H->get_sizeworld();
	int rankworld = H->get_rankworld();

	if (rankworld==0){
		Cluster<GeometricClustering> const * root = &(H->get_cluster_tree_s());
		std::stack< Cluster<GeometricClustering> const *> s;
		s.push(root);

		int size = root->get_size();

		// Permuted geometric points
		for(int i = 0; i<size; ++i) {
			output[i  ]     = x[3*root->get_perm(i)];
			output[i+size]  = x[3*root->get_perm(i)+1];
			output[i+size*2]= x[3*root->get_perm(i)+2];
		}

		int counter = 0;
		while(!s.empty()){
			Cluster<GeometricClustering> const * curr = s.top();
			s.pop();

			if (depth == curr->get_depth()){
				std::fill_n(&(output[3*size+curr->get_offset()]),curr->get_size(),counter);
				counter+=1;
			}

			// Recursion
			if (!curr->IsLeaf()){
				
				for (int p=0;p<curr->get_nb_sons();p++){
					s.push(&(curr->get_son(p)));
				}
			}

		}
	}
}

void* MultiHMatrixCreateSym(double* pts, int n, void (*getcoefs)(int,int,K*), int nm) {

  vector<R3> p(n);
	for(int j=0; j<n; j++){
    p[j][0] = pts[3*j];
    p[j][1] = pts[3*j+1];
    p[j][2] = pts[3*j+2];
	}

  // Matrix
	MyMultiMatrix A(p,getcoefs,nm);

  // MultiHmatrix
	MultiHMatrix<K,MultipartialACA,GeometricClustering>* H = new MultiHMatrix<K,MultipartialACA,GeometricClustering>(A,p,p);

  return H;
}

void* MultiHMatrixCreate(double* pts1, int m, double* pts2, int n, void (*getcoefs)(int,int,K*), int nm) {

  vector<R3> p1(m),p2(n);
	for(int j=0; j<m; j++){
    p1[j][0] = pts1[3*j];
    p1[j][1] = pts1[3*j+1];
    p1[j][2] = pts1[3*j+2];
	}

	for(int j=0; j<n; j++){
	p2[j][0] = pts2[3*j];
	p2[j][1] = pts2[3*j+1];
	p2[j][2] = pts2[3*j+2];
	}

  // Matrix
	MyMultiMatrix A(p1,p2,getcoefs, nm);
	
  // MultiHmatrix
	MultiHMatrix<K,MultipartialACA,GeometricClustering>* H = new MultiHMatrix<K,MultipartialACA,GeometricClustering>(A,p1,p2);

  return H;
}

void* MultiHMatrixCreatewithsubmatSym(double* pts, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*), int nm) {

  vector<R3> p(n);
	for(int j=0; j<n; j++){
    p[j][0] = pts[3*j];
    p[j][1] = pts[3*j+1];
    p[j][2] = pts[3*j+2];
	}

  // Matrix
	MyMultiMatrixwithsubmat A(p,getsubmatrix,nm);

  // Hmatrix
	MultiHMatrix<K,MultipartialACA,GeometricClustering>* H = new MultiHMatrix<K,MultipartialACA,GeometricClustering>(A,p,p);
  return H;
}

void* MultiHMatrixCreatewithsubmat(double* pts1, int m, double* pts2, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*), int nm) {

  vector<R3> p1(m),p2(n);
	for(int j=0; j<m; j++){
    p1[j][0] = pts1[3*j];
    p1[j][1] = pts1[3*j+1];
    p1[j][2] = pts1[3*j+2];
	}

	for(int j=0; j<n; j++){
	p2[j][0] = pts2[3*j];
	p2[j][1] = pts2[3*j+1];
	p2[j][2] = pts2[3*j+2];
	}

  // Matrix
	MyMultiMatrixwithsubmat A(p1,p2,getsubmatrix,nm);

  // Hmatrix
	MultiHMatrix<K,MultipartialACA,GeometricClustering>* H = new MultiHMatrix<K,MultipartialACA,GeometricClustering>(A,p1,p2);
  return H;
}

void* getHMatrix(void* MultiH, int i) {
		HMatrix<K,bareLowRankMatrix,GeometricClustering>* H = &(reinterpret_cast<MultiHMatrix<K,MultipartialACA,GeometricClustering>*>(MultiH)->operator[](i));

  	return H;
}

void MultiHMatrixVecProd(void* MultiH, int i, K* x, K* Ax) {
  	reinterpret_cast<MultiHMatrix<K,MultipartialACA,GeometricClustering>*>(MultiH)->operator[](i).mvprod_global(x,Ax);
}

int nbhmats(void* MultiH) {
    return reinterpret_cast<MultiHMatrix<K,MultipartialACA,GeometricClustering>*>(MultiH)->nb_hmats();
}

int multi_nbrows(void* H) {
    return reinterpret_cast<MultiHMatrix<K,MultipartialACA,GeometricClustering>*>(H)->nb_rows();
}

int multi_nbcols(void* H) {
    return reinterpret_cast<MultiHMatrix<K,MultipartialACA,GeometricClustering>*>(H)->nb_cols();
}

}
