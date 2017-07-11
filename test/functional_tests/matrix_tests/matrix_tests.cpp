#include "htool/matrix.hpp"

using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
  bool test =0;

  //// Matrix - double
  // Constructor
  Matrix<double> Md(10,5);
  Matrix<double> Pd(5,10);
  test = test || !(Md.nb_rows()==10);
  cout << "nb_rows = "<< Md.nb_rows() << endl;
  test = test || !(Md.nb_cols()==5);
  cout << "nb_cols = "<< Md.nb_cols() << endl;


  // Access operator
  for (int i =0 ; i<10;i++){
    for (int j =0;j<5;j++){
      Md(i,j)=i+j;
      Pd(j,i)=2;
    }
  }
  // Assignement operator
  Matrix<double> Nd=Md;
  test = test || !(normFrob(Nd-Md)<1e-16);
  cout << "Md : "<< Md <<endl;
  cout << "Nd : "<< Nd <<endl;


  // Getters for strides
  vector<double> diff = {1,2,3,4,5};
  test = test || !(norm2(Md.get_row(1)-diff)<1e-16);
  cout << "Md second row : "<< Md.get_row(1) << endl;
  diff = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  test = test || !(norm2(Md.get_col(3)-diff)<1e-16);
  cout << "Md fourth col : "<< Md.get_col(3) << endl;

  // Setters for strides
  std::vector<double> rowd(5,2);
  std::vector<double> cold(10,2);
  Md.set_row(1,rowd);
  diff = {2, 2, 2, 2, 2};
  test = test || !(norm2(Md.get_row(1)-diff)<1e-16);
  Md.set_col(4,cold);
  diff = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  test = test || !(norm2(Md.get_col(4)-diff)<1e-16);
  cout << "Set second row and fifth col of Md : "<< Md << endl;

  // Matrix vector product
  std::vector <double> md(5,1);
  diff = {8, 10, 16, 20, 24, 28, 32, 36, 40, 44};
  test = test || !(norm2(Md*md-diff)<1e-16);
  cout << "Matrix vector product : "<< Md*md<<endl;

  // Matrix argmax
  test = test || !(argmax(Md).first-9==0);
  test = test || !(argmax(Md).second-3==0);
  cout << "Md's argmax : "<< argmax(Md).first<<" "<< argmax(Md).second << endl;

  //// Matrix - complex double
  // Constructor
  Matrix<complex<double> > Mcd(10,5);
  Matrix<complex<double> > Pcd(5,10);

  // Access operator
  for (int i =0 ; i<10;i++){
    for (int j =0;j<5;j++){
      Mcd(i,j)=i+j;
      Pcd(j,i)=2;
    }
  }
  // Assignement operator
  Matrix<complex<double> > Ncd=Mcd;
  test = test || !(normFrob(Ncd-Mcd)<1e-16);
  cout << "Mcd : "<< Mcd <<endl;
  cout << "Ncd : "<< Ncd <<endl;
  cout << "Pcd : "<< Pcd <<endl;

  // Getters for strides
  vector<complex<double>> diffc = {1,2,3,4,5};
  test = test || !(norm2(Mcd.get_row(1)-diffc)<1e-16);
  cout << "Mcd second row : "<< Mcd.get_row(1) << endl;
  diffc = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  test = test || !(norm2(Mcd.get_col(3)-diffc)<1e-16);
  cout << "Mcd fourth row : "<< Mcd.get_col(3) << endl;

  // Setters for strides
  std::vector<std::complex<double> > rowcd(5,2);
  std::vector<std::complex<double> > colcd(10,2);
  Mcd.set_row(1,rowcd);
  diffc = {2, 2, 2, 2, 2};
  test = test || !(norm2(Mcd.get_row(1)-diffc)<1e-16);
  Mcd.set_col(4,colcd);
  diffc = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  test = test || !(norm2(Mcd.get_col(4)-diffc)<1e-16);
  cout << "Set second row and fifth col of Md : "<< Mcd << endl;

  // Matrix vector product
  std::vector <complex<double> > mcd(5,1);
  diffc = {8, 10, 16, 20, 24, 28, 32, 36, 40, 44};
  test = test || !(norm2(Mcd*mcd-diffc)<1e-16);
  cout << "Matrix vector product : "<< Mcd*mcd<<endl;

  // Matrix argmax
  test = test || !(argmax(Mcd).first-9==0);
  test = test || !(argmax(Mcd).second-3==0);
  cout << "Mcd's argmax : "<< argmax(Mcd).first<<" "<< argmax(Mcd).second << endl;

  // Submatrix - double
  std::vector<int> ir = {0,1};
  std::vector<int> ic = {0,1};
  SubMatrix<double> SMd(Md,ir,ic);
  double diffs = 0;
  for (int i =0;i<ir.size();i++){
    for (int j =0;j<ic.size();j++){
      diffs += abs(Md(ir[i],ic[j])-SMd(i,j));
    }
  }
  test = test || !(diffs<1e-16);
  cout<< "Md : "<<Md<<endl;
  cout<< "SMd : "<<SMd<<endl;


  return test;
}
