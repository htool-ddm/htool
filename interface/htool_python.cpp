#include <Python.h>
#include <htool/htool.hpp>

using namespace std;
using namespace htool;

class MyMatrix: public IMatrix<double>{
	const vector<R3>& p;
  double (*getcoef)(int,int);

public:
	MyMatrix(const vector<R3>& p1, double (*g)(int,int)):IMatrix(p1.size(),p1.size()),p(p1),getcoef(g) {}

	double get_coef(const int& i, const int& j)const {return getcoef(i,j);}
};

extern "C" {

void* HMatrixCreate(double* pts, int n, double (*getcoef)(int,int)) {

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
	HMatrix<partialACA,double>* H = new HMatrix<partialACA,double>(A,p);

  return H;
}

void printinfos(void* H) {
    reinterpret_cast<HMatrix<partialACA,double>*>(H)->print_infos();
}

void mvprod(void* H, double* x, double* Ax) {
    reinterpret_cast<HMatrix<partialACA,double>*>(H)->mvprod_global(x,Ax);
}

}