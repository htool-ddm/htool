#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../wrappers/wrapper_lapack.hpp"
#include "lrmat.hpp"


namespace htool {

  inline double cheb_node(int k, int L){return std::cos((double)(2*k+1)/(double)(2*L)*M_PI);}
  inline double T(double x, int k){return cos((double)(k)*acos(x));}
  inline void getTr(int L, double *Tr){
    for(int k = 0; k < L; k++){
      double r = cheb_node(k,L); for(int i = 0; i < L; i++){(Tr+k*L)[i] = T(r,i);}}}
  inline void getTs  (double u, int L, double *  Tu){
    for(int i = 0; i < L; i++){  Tu[i] =   T(u,i);}}
  inline void C1D(double *Tx, double *Tr, int L, double *S){
    for(int k = 0; k < L; k++){
      double res = 1.;
      double *Trk = Tr+k*L;
      for(int j = 1; j < L; j++){
	res += 2. * Tx[j] * Trk[j];}
      S[k] = res/(double)(L);}
  }
  inline void part_to_cheb(double x, double y, double z, double* Mu, int Lc){
    double *Cx = new double[Lc];
    double *Cy = new double[Lc];
    double *Cz = new double[Lc];
    for(int i = 0; i < Lc; i++){
      Cx[i] = C1D(i,x,Lc);
      Cy[i] = C1D(i,y,Lc);
      Cz[i] = C1D(i,z,Lc);
    }
    for(int i = 0; i < Lc; i++){
      for(int j = 0; j < Lc; j++){
	for(int k = 0; k < Lc; k++){
	  Mu[i*Lc*Lc + j*Lc + k] = Cx[i] * Cy[j] * Cz[k];
	}
      }
    }
  }
  
  template <typename T>
  class IBcomp final : public VirtualLowRankGenerator<T> {
    
  private:
  public:
    using VirtualLowRankGenerator<T>::VirtualLowRankGenerator;
    // TO BE COMPLETED!
  };
  
} // namespace htool



#endif
