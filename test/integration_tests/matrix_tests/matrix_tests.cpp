#include "htool/matrix.hpp"

using namespace std;
using namespace htool;
int main(int argc, char const *argv[]) {
  //// Vectors
  // int
  vector<int> ai(10,0);
  vector<int> bi(10,0);

  int counti=0;
  for(int& i: ai){
    i=counti;
    counti++;}
  cout<<"ai = "<<ai<<endl;

  copy( ai.begin(),ai.end(),bi.begin());
  cout<<"bi = "<<bi<<endl;

  cout <<"ai+bi       = "<< ai + bi <<endl;
  cout <<"ai-bi       = "<< ai - bi <<endl;
  cout <<"ai*3        = "<<ai*3<<endl;
  cout <<"ai/3        = "<<ai/3<<endl;
  cout <<"dprod(ai,bi)= "<< dprod(ai,bi)<<endl;
  cout <<"norm2(ai)   = "<< norm2(ai)<<endl;
  cout <<"argmax(ai)  = "<< argmax(ai)<<endl;
  cout <<"max(ai+bi)  = "<< max(ai+bi)<<endl;
  cout <<"mean(ai)    = "<<mean(ai)<<endl;

  ai*=3;
  cout <<"ai*=3    ai = "<<ai<<endl;
  ai/=3;
  cout <<"ai/=3    ai = "<<ai<<endl;


  // double
  vector<double> ad(10,0);
  vector<double> bd(10,0);

  double countd=0.1;
  for(double& i: ad){
    i=countd;
    countd++;}
  cout<<"ad = "<<ad<<endl;

  copy( ad.begin(),ad.end(),bd.begin());
  cout<<"bd = "<<bd<<endl;

  cout <<"ad+bd       = "<< ad + bd <<endl;
  cout <<"ad-bd       = "<< ad - bd <<endl;
  cout <<"ad*3        = "<<ad*3<<endl;
  cout <<"ad/3        = "<<ad/3<<endl;
  cout <<"dprod(ad,bd)= "<< dprod(ad,bd)<<endl;
  cout <<"norm2(ad)   = "<< norm2(ad)<<endl;
  cout <<"argmax(ad)  = "<<argmax(ad)<<endl;
  cout <<"max(ad+bd)  = "<<max(ad+bd)<<endl;
  cout <<"mean(ad)    = "<<mean(ad)<<endl;

  ad*=3;
  cout <<"ad*=3    ad = "<<ad<<endl;
  ad/=3;
  cout <<"ad/=3    ad = "<<ad<<endl;

  // complex double
  vector<complex<double> > acd(10,0);
  vector<complex<double> > bcd(10,0);

  complex<double> countcd(0,1);
  for(complex<double>& i: acd){
    i=countcd;
    countcd+=1;}
  cout<<"acd = "<<acd<<endl;

  transform( acd.begin(),acd.end(),bcd.begin(),[](complex<double>u){return u+std::complex<double>(0,1);});
  cout<<"bcd = "<<bcd<<endl;

  cout <<"acd+bcd       = "<< acd + bcd <<endl;
  cout <<"acd-bcd       = "<< acd - bcd <<endl;
  cout <<"acd*3         = "<<acd*3<<endl;
  cout <<"acd/3         = "<<acd/3<<endl;
  cout <<"dprod(acd,bcd)= "<< dprod(acd,bcd)<<endl;
  cout <<"norm2(acd)    = "<< norm2(acd)<<endl;
  cout <<"argmax(acd)   = "<<argmax(acd)<<endl;
  cout <<"max(acd+bcd)  = "<<max(acd+bcd)<<endl;
  cout <<"mean(acd)    = "<<mean(acd)<<endl;

  acd*=3;
  cout <<"acd*=3    acd = "<<acd<<endl;
  acd/=3;
  cout <<"acd/=3    acd = "<<acd<<endl;

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
