#ifndef HTOOL_PYTHON_HMATRIX_HPP
#define HTOOL_PYTHON_HMATRIX_HPP


#include <Python.h>
#include <htool/htool.hpp>

using namespace std;
using namespace htool;

#ifdef FORCE_COMPLEX
typedef std::complex<double> K;
#else
typedef double K;
#endif

class MyMatrix: public IMatrix<K>{
	const vector<R3>& p1;
	const vector<R3>& p2;
  void (*getcoef)(int,int,K*);

public:
	MyMatrix(const vector<R3>& p10, void (*g)(int,int,K*)):IMatrix(p10.size(),p10.size()),p1(p10),p2(p10),getcoef(g) {}

	MyMatrix(const vector<R3>& p10,const vector<R3>& p20, void (*g)(int,int,K*)):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20),getcoef(g) {}

	K get_coef(const int& i, const int& j)const {
		K r;
		getcoef(i,j,&r);
		return r;
	}
};

class MyMatrixwithsubmat: public IMatrix<K>{
	const vector<R3>& p1;
	const vector<R3>& p2;
	void (*getsubmatrix)(const int*,const int*,int,int,K*);

public:
	MyMatrixwithsubmat(const vector<R3>& p10, void (*g)(const int*,const int*,int,int,K*)):IMatrix(p10.size(),p10.size()),p1(p10),p2(p10),getsubmatrix(g) {}

	MyMatrixwithsubmat(const vector<R3>& p10,const vector<R3>& p20, void (*g)(const int*,const int*,int,int,K*)):IMatrix(p10.size(),p20.size()),p1(p10),p2(p20),getsubmatrix(g) {}

	SubMatrix<K> get_submatrix(const std::vector<int>& I, const std::vector<int>& J) const {
		SubMatrix<K> mat(I,J);
		getsubmatrix(I.data(),J.data(),I.size(),J.size(),mat.data());
		return mat;
	}

	K get_coef(const int& i, const int& j)const {
		K r;
		getsubmatrix(&i,&j,1,1,&r);
		return r;
	}

};

// extern "C" {
// unsigned short scalar = std::is_same<K, double>::value ? 0 : 1;

// void* HMatrixCreateSym(double* pts, int n, void (*getcoef)(int,int,K*)) {

//   vector<R3> p(n);
// 	for(int j=0; j<n; j++){
//     p[j][0] = pts[3*j];
//     p[j][1] = pts[3*j+1];
//     p[j][2] = pts[3*j+2];
// 	}

//   // Matrix
// 	MyMatrix A(p,getcoef);

//   // Hmatrix
// 	HMatrix<partialACA,K>* H = new HMatrix<partialACA,K>(A,p);

//   return H;
// }

// void* HMatrixCreate(double* pts1, int m, double* pts2, int n, void (*getcoef)(int,int,K*)) {
// std::cout << m <<" "<<n<< std::endl;
//   vector<R3> p1(m),p2(n);
// 	for(int j=0; j<m; j++){
//     p1[j][0] = pts1[3*j];
//     p1[j][1] = pts1[3*j+1];
//     p1[j][2] = pts1[3*j+2];
// 	}

// 	for(int j=0; j<n; j++){
// 	p2[j][0] = pts2[3*j];
// 	p2[j][1] = pts2[3*j+1];
// 	p2[j][2] = pts2[3*j+2];
// 	}

//   // Matrix
// 	MyMatrix A(p1,p2,getcoef);

//   // Hmatrix
// 	HMatrix<partialACA,K>* H = new HMatrix<partialACA,K>(A,p1,p2);

//   return H;
// }

// void* HMatrixCreatewithsubmatSym(double* pts, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*)) {

//   vector<R3> p(n);
// 	for(int j=0; j<n; j++){
//     p[j][0] = pts[3*j];
//     p[j][1] = pts[3*j+1];
//     p[j][2] = pts[3*j+2];
// 	}

//   // Matrix
// 	MyMatrixwithsubmat A(p,getsubmatrix);

//   // Hmatrix
// 	HMatrix<partialACA,K>* H = new HMatrix<partialACA,K>(A,p);

//   return H;
// }

// void* HMatrixCreatewithsubmat(double* pts1, int m, double* pts2, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*)) {

//   vector<R3> p1(m),p2(n);
// 	for(int j=0; j<m; j++){
//     p1[j][0] = pts1[3*j];
//     p1[j][1] = pts1[3*j+1];
//     p1[j][2] = pts1[3*j+2];
// 	}

// 	for(int j=0; j<n; j++){
// 	p2[j][0] = pts2[3*j];
// 	p2[j][1] = pts2[3*j+1];
// 	p2[j][2] = pts2[3*j+2];
// 	}

//   // Matrix
// 	MyMatrixwithsubmat A(p1,p2,getsubmatrix);

//   // Hmatrix
// 	HMatrix<partialACA,K>* H = new HMatrix<partialACA,K>(A,p1,p2);

//   return H;
// }

// void printinfos(void* H) {
//     reinterpret_cast<HMatrix<partialACA,K>*>(H)->print_infos();
// }

// void mvprod(void* H, K* x, K* Ax) {
//     reinterpret_cast<HMatrix<partialACA,K>*>(H)->mvprod_global(x,Ax);
// }

// int getnlrmat(void* H) {
//     return reinterpret_cast<HMatrix<partialACA,K>*>(H)->get_nlrmat();
// }

// int getndmat(void* H) {
//     return reinterpret_cast<HMatrix<partialACA,K>*>(H)->get_ndmat();
// }

// int nbrows(void* H) {
//     return reinterpret_cast<HMatrix<partialACA,K>*>(H)->nb_rows();
// }

// int nbcols(void* H) {
//     return reinterpret_cast<HMatrix<partialACA,K>*>(H)->nb_cols();
// }

// void setepsilon(double eps) { SetEpsilon(eps); }
// void seteta(double eta) { SetEta(eta); }
// void setminclustersize(int m) { SetMinClusterSize(m); }
// void setmaxblocksize(int m) { SetMaxBlockSize(m); }

// void getpattern(void* pH, int* buf) {
// 	HMatrix<partialACA,K>* H = reinterpret_cast<HMatrix<partialACA,K>*>(pH);

// 	const std::vector<partialACA<K>*>& lrmats = H->get_MyFarFieldMats();
// 	const std::vector<SubMatrix<K>*>& dmats = H->get_MyNearFieldMats();

// 	int nb = dmats.size() + lrmats.size();

// 	int sizeworld = H->get_sizeworld();
// 	int rankworld = H->get_rankworld();

// 	int nbworld[sizeworld];
// 	MPI_Allgather(&nb, 1, MPI_INT, nbworld, 1, MPI_INT, H->get_comm());
// 	int nbg = 0;
// 	for (int i=0; i<sizeworld; i++) {
// 		nbg += nbworld[i];
// 	}

// 	for (int i=0;i<dmats.size();i++) {
// 		const SubMatrix<K>& l = *(dmats[i]);
// 		buf[5*i] = l.get_offset_i();
// 		buf[5*i+1] = l.nb_rows();
// 		buf[5*i+2] = l.get_offset_j();
// 		buf[5*i+3] = l.nb_cols();
// 		buf[5*i+4] = -1;
// 	}

// 	for (int i=0;i<lrmats.size();i++) {
// 		const LowRankMatrix<K>& l = *(lrmats[i]);
// 		buf[5*(dmats.size()+i)] = l.get_offset_i();
// 		buf[5*(dmats.size()+i)+1] = l.nb_rows();
// 		buf[5*(dmats.size()+i)+2] = l.get_offset_j();
// 		buf[5*(dmats.size()+i)+3] = l.nb_cols();
// 		buf[5*(dmats.size()+i)+4] = l.rank_of();
// 	}

// 	int displs[sizeworld];
// 	int recvcounts[sizeworld];
// 	displs[0] = 0;

// 	for (int i=0; i<sizeworld; i++) {
// 		recvcounts[i] = 5*nbworld[i];
// 		if (i > 0)	displs[i] = displs[i-1] + recvcounts[i-1];
// 	}
// 	MPI_Gatherv(rankworld==0?MPI_IN_PLACE:buf, recvcounts[rankworld], MPI_INT, buf, recvcounts, displs, MPI_INT, 0, H->get_comm());
// }

// }

#endif