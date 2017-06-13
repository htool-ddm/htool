#include "htool/matrix.hpp"

using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
  //// Matrix - double
  // Constructor
  Matrix<double> Md(10,5);
  Matrix<double> Pd(5,10);
  cout << "nb_rows = "<< Md.nb_rows() << endl;
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
  cout << "Md : "<< Md <<endl;
  cout << "Nd : "<< Nd <<endl;
  cout << "Pd : "<< Pd <<endl;

  // Getters for strides
  cout << "Md second row : "<< Md.get_row(1) << endl;
  cout << "Md fourth col : "<< Md.get_col(3) << endl;

  // Setters for strides
  std::vector<double> rowd(5,2);
  std::vector<double> cold(10,2);
  Md.set_row(1,rowd);
  Md.set_col(4,cold);
  cout << "Set second row and fifth col of Md : "<< Md << endl;

  // Matrix vector product
  std::vector <double> md(5,1);
  cout << "Matrix vector product : "<< Md*md<<endl;

  // Matrix argmax
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
  cout << "Mcd : "<< Mcd <<endl;
  cout << "Ncd : "<< Ncd <<endl;
  cout << "Pcd : "<< Pcd <<endl;

  // Getters for strides
  cout << "Mcd second row : "<< Mcd.get_row(1) << endl;
  cout << "Mcd fourth row : "<< Mcd.get_col(3) << endl;

  // Setters for strides
  std::vector<std::complex<double> > rowcd(5,2);
  std::vector<std::complex<double> > colcd(10,2);
  Mcd.set_row(1,rowcd);
  Mcd.set_col(4,colcd);
  cout << "Set second row and fifth col of Md : "<< Mcd << endl;

  // Matrix vector product
  std::vector <complex<double> > mcd(5,1);
  cout << "Matrix vector product : "<< Mcd*mcd<<endl;

  // Matrix argmax
  cout << "Mcd's argmax : "<< argmax(Mcd).first<<" "<< argmax(Mcd).second << endl;

  return 0;
}
