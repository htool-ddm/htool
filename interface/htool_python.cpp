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
	const vector<R3>& p;
  void (*getcoef)(int,int,K*);

public:
	MyMatrix(const vector<R3>& p1, void (*g)(int,int,K*)):IMatrix(p1.size(),p1.size()),p(p1),getcoef(g) {}

	K get_coef(const int& i, const int& j)const {
		K r;
		getcoef(i,j,&r);
		return r;
	}
};

class MyMatrixwithsubmat: public IMatrix<K>{
	const vector<R3>& p;
	void (*getsubmatrix)(const int*,const int*,int,int,K*);

public:
	MyMatrixwithsubmat(const vector<R3>& p1, void (*g)(const int*,const int*,int,int,K*)):IMatrix(p1.size(),p1.size()),p(p1),getsubmatrix(g) {}

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

extern "C" {
unsigned short scalar = std::is_same<K, double>::value ? 0 : 1;

void* HMatrixCreate(double* pts, int n, void (*getcoef)(int,int,K*)) {

  vector<R3> p(n);
	for(int j=0; j<n; j++){
    p[j][0] = pts[3*j];
    p[j][1] = pts[3*j+1];
    p[j][2] = pts[3*j+2];
	}

  // Matrix
	MyMatrix A(p,getcoef);

  SetEpsilon(1e-2);
  SetEta(10);
  SetMinClusterSize(10);

  // Hmatrix
	HMatrix<partialACA,K>* H = new HMatrix<partialACA,K>(A,p);

  return H;
}

void* HMatrixCreatewithsubmat(double* pts, int n, void (*getsubmatrix)(const int*, const int*, int,int,K*)) {

  vector<R3> p(n);
	for(int j=0; j<n; j++){
    p[j][0] = pts[3*j];
    p[j][1] = pts[3*j+1];
    p[j][2] = pts[3*j+2];
	}

  // Matrix
	MyMatrixwithsubmat A(p,getsubmatrix);

  SetEpsilon(1e-2);
  SetEta(10);
  SetMinClusterSize(10);

  // Hmatrix
	HMatrix<partialACA,K>* H = new HMatrix<partialACA,K>(A,p);

  return H;
}

void printinfos(void* H) {
    reinterpret_cast<HMatrix<partialACA,K>*>(H)->print_infos();
}

void mvprod(void* H, K* x, K* Ax) {
    reinterpret_cast<HMatrix<partialACA,K>*>(H)->mvprod_global(x,Ax);
}

}