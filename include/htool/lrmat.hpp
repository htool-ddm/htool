#ifndef LRMAT_HPP
#define LRMAT_HPP

#include <vector>
#include "cluster.hpp"
#include "matrix.hpp"
namespace htool{

template<typename T>
class LowRankMatrix: public Parametres{


protected:
  // Data member
  int rank, nr, nc;
  Matrix<T>  U,V;
  const std::vector<int>& ir;
  const std::vector<int>& ic;
  const Cluster& t;
  const Cluster& s;

  LowRankMatrix(){};
  LowRankMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, const Cluster& t0, const Cluster& s0,int rank0=-1):rank(rank0),nr(ir0.size()),nc(ic0.size()),U(ir0.size(),1),V(1,ic0.size()),ir(ir0),ic(ic0),t(t0),s(s0){
  }
  LowRankMatrix(const LowRankMatrix&) = default; // copy constructor
  LowRankMatrix& operator=(const LowRankMatrix&) = default; // copy assignement operator
  LowRankMatrix(LowRankMatrix&&) = default; // move constructor
  LowRankMatrix& operator=(LowRankMatrix&&) = default; // move assignement operator


public:
  int nb_rows() const {return this->nr;}
  int nb_cols() const{return this->nc;}
  int rank_of() const {return this->rank;}
  std::vector<T> operator*(const std::vector<T> a){
    return this->U*(this->V*a);
  }
  T get_U(int i, int j) const {return U(i,j);}
  T get_V(int i, int j) const {return V(i,j);}

  double compression_rate(){return (1 - ( this->rank*( 1./double(this->nr) + 1./double(this->nc)))); }

  // friend Real NormFrob(const ACA& m){
	// 	/*
	// 	const std::vector<vectCplx>& u = m.u;
	// 	const std::vector<vectCplx>& v = m.v;
	// 	const int& rank = m.rank;
	// 	*/
  //
	// 	Cplx frob = 0.;
  //
	// 	for (int j=0;j<m.nr;j++)
	// 	for (int k=0;k<m.nc;k++){
	// 		Cplx aux=0;
	// 			for (int l=0;l<m.rank;l++){
	// 				aux += m.U(j,l) * m.V(l,k);
	// 			}
	// 		frob+=pow(abs(aux),2);
	// 	}
  //
	// 	/*
	// 	for(int j=0; j<rank; j++){
	// 		for(int k=0; k<rank; k++){
	// 			frob += dprod(v[k],v[j])*dprod(u[k],u[j]) ;
	// 		}
	// 	}
	// 	*/
	// 	return sqrt(abs(frob));
	// }

  friend std::ostream& operator<<(std::ostream& os, const LowRankMatrix& m){
    os << "rank:\t" << m.rank << std::endl;
    os << "nr:\t"   << m.nr << std::endl;
    os << "nc:\t"   << m.nc << std::endl;
    os << "U:\n";
    os<< m.U << std::endl;
    os<< m.V << std::endl;
    // for(int j=0; j<m.nr; j++){
    //   for(int k=0; k<m.rank; k++){
    //     std::cout << m.U(j,k) << "\t";}
    //   std::cout << "\n";}
    // os << "\nv:\n";
    // for(int j=0; j<m.nc; j++){
    //   for(int k=0; k<m.rank; k++){
    //     std::cout << m.V(k,j) << "\t";
    //   }
    //   std::cout << "\n";
    // }

    return os;
  }
};

template<typename T>
double Frobenius_relative_error(const LowRankMatrix<T>& lrmat, const IMatrix<T>& ref, int reqrank){
  assert(reqrank<=lrmat.rank_of());
  T norm= 0;
  T err = 0;
  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      T aux=ref.get_coef(j,k);
      norm+= std::pow(std::abs(aux),2);
      for (int l=0;l<std::min(reqrank,lrmat.rank_of());l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  err =err/norm;
  return std::sqrt(err);
}


template<typename T>
double Frobenius_absolute_error(const LowRankMatrix<T>& lrmat, const IMatrix<T>& ref, int reqrank){
  T err = 0;
  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      T aux=ref.get_coef(j,k);
      for (int l=0;l<std::min(reqrank,lrmat.rank_of());l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}




}

#endif