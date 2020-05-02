#ifndef HTOOL_MULTIPARTIALACA_HPP
#define HTOOL_MULTIPARTIALACA_HPP

#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>
#include "multilrmat.hpp"
#include "../types/multimatrix.hpp"


namespace htool {

template<typename T, typename ClusterImpl>
class MultipartialACA: public MultiLowRankMatrix<T,ClusterImpl>{


public:
	//=========================//
	//    PARTIAL PIVOT ACA    //
	//=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
	using MultiLowRankMatrix<T,ClusterImpl>::MultiLowRankMatrix;

	void build(const MultiIMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs){
		if(this->rank == 0){
			for (int l=0;l<this->nm;l++){
				this->LowRankMatrices[l].U.resize(this->nr,1);
				this->LowRankMatrices[l].V.resize(1,this->nc);
			}
		}
		else{

			//// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
			double dist=1e30;
			int I=0;
			for (int i =0;i<int(this->nr/Parametres::ndofperelt);i++){
				double aux_dist= norm2(xt[tabt[this->ir[i*Parametres::ndofperelt]]]-t.get_ctr());
				if (dist>aux_dist){
					dist=aux_dist;
					I=i*Parametres::ndofperelt;
				}
			}
			// Partial pivot
			int J=0;
			int q = 0;
			int reqrank = this->rank;
			std::vector<Matrix<T> > uu, vv;
			std::vector<bool> visited_row(this->nr,false);
			std::vector<bool> visited_col(this->nc,false);

			std::vector<double> frob(this->nm,0);
			std::vector<double> aux(this->nm,0);

			double stopping_criterion=0;

			// Either we have a required rank
			// Or it is negative and we have to check the relative error between two iterations.
			//But to do that we need a least two iterations.
			while (((reqrank > 0) && (q < std::min(reqrank,std::min(this->nr,this->nc))) ) ||
			       ((reqrank < 0) && (q==0 || sqrt(stopping_criterion)>this->epsilon))) {
				// Next current rank
				q+=1;

				if (q*(this->nr+this->nc) > (this->nr*this->nc)) { // the next current rank would not be advantageous
                    q=-1;
					break;
				}
				else{
					Matrix<T> r(this->nc,this->nm),c(this->nr,this->nm);
					std::vector<T> coefs(this->nm);

					// Compute the first cross
					//==================//
					// Look for a column
					double pivot = 0.;

					MultiSubMatrix<T> row = A.get_submatrices(std::vector<int> {this->ir[I]},this->ic);

					for (int l=0;l<this->nm;l++){
						for(int k=0; k<this->nc; k++){
							r(k,l) = row[l](0,k);
							for(int j=0; j<uu.size(); j++){
								r(k,l) += -uu[j](I,l)*vv[j](k,l);
							}
							if( std::abs(r(k,l))>pivot && !visited_col[k] ){
								J=k; pivot=std::abs(r(k,l));}
						}

					}

					visited_row[I] = true;
					std::vector<T> gamma(this->nm);
					for (int l=0;l<this->nm;l++){
						gamma[l]=T(1.)/r(J,l);
					}
					//==================//
					// Look for a line
					if( std::abs(min(r.get_row(J))) > 1e-15 ){
						double cmax = 0.;
						MultiSubMatrix<T> col = A.get_submatrices(this->ir,std::vector<int> {this->ic[J]});
						for (int l=0;l<this->nm;l++){
							for(int j=0; j<this->nr; j++){
								c(j,l) = col[l](j,0);
								for(int k=0; k<uu.size(); k++){
									c(j,l) += -uu[k](j,l)*vv[k](J,l);
								}
								c(j,l) = gamma[l]*c(j,l);
								if( std::abs(c(j,l))>cmax && !visited_row[j] ){
									I=j; cmax=std::abs(c(j,l));}
							}
						}

						visited_col[J] = true;

						// Test if no given rank
						if (reqrank<0){
							stopping_criterion=0;
							for(int l=0; l<this->nm; l++){
								// Error estimator
								T frob_aux = 0.;
								aux[l] = std::abs(dprod(c.get_col(l),c.get_col(l))*dprod(r.get_col(l),r.get_col(l)));
								// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
								for(int j=0; j<uu.size(); j++){
									frob_aux += dprod(r.get_col(l),vv[j].get_col(l))*dprod(c.get_col(l),uu[j].get_col(l));
								}
								// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
								frob[l] += aux[l] + 2*std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
								//==================//

								double test = aux[l]/frob[l];
								if (stopping_criterion<test){
									stopping_criterion=test;
								}

							}


							
						}
						// Matrix<T> M=A.get_submatrix(this->ir,this->ic);
						// uu.push_back(M.get_col(J));
						// vv.push_back(M.get_row(I)/M(I,J));
						// New cross added
						uu.push_back(c);
						vv.push_back(r);

					}
					else{
						// std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;
						q-=1;
						break;
					}
				}
			}

			// Final rank
			this->rank=q;
			if (this->rank>0){
				for (int l=0;l<this->nm;l++){
					this->LowRankMatrices[l].rank=q;
					this->LowRankMatrices[l].U.resize(this->nr,this->rank);
					this->LowRankMatrices[l].V.resize(this->rank,this->nc);
					for (int k=0;k<this->rank;k++){
						this->LowRankMatrices[l].U.set_col(k,uu[k].get_col(l));
						this->LowRankMatrices[l].V.set_row(k,vv[k].get_col(l));
					}
				}
			}
		}
	}
};
}
#endif
