#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "lrmat.hpp"
#include <cassert>
#include <Eigen/Dense>
// TODO use lapack instead of eigen
namespace htool{

template< typename T >
class SVD: public LowRankMatrix<T>{

private:
  // Eigen
  static const int Dynamic = Eigen::Dynamic;
  typedef Eigen::Matrix<T, Dynamic, Dynamic>     DenseMatrix;
  typedef Eigen::JacobiSVD<DenseMatrix>          SVDType;
  typedef typename SVDType::SingularValuesType   SgValType;
  typedef typename SVDType::MatrixUType		       UMatrixType;
  typedef typename SVDType::MatrixVType		       VMatrixType;

  // Data member
  std::vector<T> singular_values;


public:
  using LowRankMatrix<T>::LowRankMatrix;

  void build(const IMatrix<T>& A){
    int reqrank=0;
    if (this->rank==0){
      this->U.resize(this->nr,1);
      this->V.resize(1,this->nc);
      return;
    }
    else {
      //// Matrix assembling
      DenseMatrix M(this->nr,this->nc);
      double Norm = 0;
      SubMatrix<T> submat = A.get_submatrix(this->ir,this->ic);
      for (int i=0; i<M.rows(); i++){
        for (int j=0; j<M.cols(); j++){
          M(i,j) = submat(i, j);
          Norm+= std::pow(std::abs(M(i,j)),2);
        }
      }
      Norm=sqrt(Norm);

      //// SVD
      SVDType svd(M,Eigen::ComputeThinU | Eigen::ComputeThinV );

      const SgValType& sv = svd.singularValues();
      singular_values.resize(sv.size());
      for(int j=0; j<sv.size(); j++){
        singular_values[j]=sv[j];
      }
      const UMatrixType& uu = svd.matrixU();
      const VMatrixType& vv = svd.matrixV();

      if (this->rank==-1){

        // typename std::vector<T>::iterator it =lower_bound(singular_values.begin(),singular_values.end(),this->epsilon,[](T a, T b){return std::abs(a)>std::abs(b);});
        // Compute Frobenius norm of the approximation error
        int j=singular_values.size();
        double svd_norm=0;
        while (j>0 && std::sqrt(svd_norm)/Norm<this->epsilon){
          j=j-1;
          svd_norm+=std::pow(std::abs(singular_values[j]),2);
        
        }
        
        reqrank=j;
        
        // if (it != singular_values.begin()) {
        //   reqrank=int(it-singular_values.begin())-1;
        // } 
        // std::cout << singular_values << std::endl;
        // std::cout <<reqrank<< std::endl;
        // std::cout<<int(it-singular_values.begin())-1<<" "<<this->nr<<" "<<this->nc<<std::endl;
        if (reqrank*(this->nr+this->nc) > (this->nr*this->nc)){
          reqrank=-1;
        }
        this->rank=reqrank;

      }
      else{
        reqrank=std::min(this->rank,std::min(this->nr,this->nc));
      }

      if (this->rank>0){
        this->U.resize(M.rows(),reqrank);
        this->V.resize(reqrank,M.cols());

        for (int i=0;i<M.rows();i++){
          for (int j=0;j<reqrank;j++){
            this->U(i,j)=uu(i,j)*sv[j];
          }
        }
        for (int i=0;i<reqrank;i++){
          for (int j=0;j<M.cols();j++){
              this->V(i,j)=vv(j,i);
            }
        }
      }
    }
  }

	void build(const IMatrix<T>& A, const Cluster& t, const std::vector<R3> xt,const std::vector<int> tabt, const Cluster& s, const std::vector<R3> xs, const std::vector<int>tabs){
    this->build(A);
  }
  T get_singular_value(int i){return singular_values[i];}

};

}


#endif
