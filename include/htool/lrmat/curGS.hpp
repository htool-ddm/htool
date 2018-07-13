// CUR_GS implementation
// Approximate an admissible submatrix
// Can take an Approximation rank as input or find it adaptively using the same stopping criterion from ACAp

#ifndef HTOOL_CURGS_HPP
#define HTOOL_CURGS_HPP

#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>
#include "lrmat.hpp"

#include "../types/matrix.hpp"
#include "../types/hmatrix.hpp"
#include "../wrappers/wrapper_mpi.hpp"
#include "../wrappers/wrapper_hpddm.hpp"

namespace htool {

template<typename T>
class curGS: public LowRankMatrix<T>{


public:
	//===================================//
	// CUR with Gravity-Centers sampling //
	//===================================//


	curGS(const std::vector<int>& ir0, const std::vector<int>& ic0, int rank0=-1): LowRankMatrix<T>(ir0,ic0,rank0){}

	curGS(const std::vector<int>& ir0, const std::vector<int>& ic0,int offset_i0, int offset_j0, int rank0=-1): LowRankMatrix<T>(ir0,ic0,offset_i0,offset_j0,rank0){}


	void build(const IMatrix<T>& A, const Cluster& t, const std::vector<R3> xt,const std::vector<int> tabt, const Cluster& s, const std::vector<R3> xs, const std::vector<int>tabs){

		if(this->rank == 0){
			this->U.resize(this->nr,1);
			this->V.resize(1,this->nc);
		}
		else{



			double dist=1e30;
			int I=0;
			for (int i =0;i<int(this->nr/Parametres::ndofperelt);i++){
				double aux_dist= norm2(xt[tabt[this->ir[i*Parametres::ndofperelt]]]-t.get_ctr());
				if (dist>aux_dist){
					dist=aux_dist;
					I=i*Parametres::ndofperelt;
				}
			}

			auto vs = s.get_ctr();

			int k = this->rank;

			if (k<0) k=4; // minimum value to start curGS adaptively

			int l = ceil(log(k)/log(2));
			int tt; // oversampling
			if(k > pow(2,l-1) && k>2 ) tt = pow(2,l+1);
			else tt = pow(2,l);

			int h = log(tt)/log(2);
			if(k==1) h=1;


			std::vector<int> Jx;
			std::vector<int> Sx;
			std::vector<R3> ctrs;

			// Perform a Binary tree of depth h on cluster s
			s.get_offset(Jx,h);
			s.get_size(Sx,h);
			s.get_ctr(ctrs,h);


			// Geometric Sampling GS_GC:
			std::vector<int> Jv(Jx.size()); // Indices of points the closest to gravity centers of subdomains

			for (int j = 0; j < Jx.size(); j++) {
				dist=1e30;
				for (int i = Jx[j] + this->offset_j;  i< Jx[j] + Sx[j] + this->offset_j; i++){
					double aux_dist= norm2(xs[tabs[this->ic[i*Parametres::ndofperelt]]]- ctrs[j] );
					if (dist>aux_dist){
						dist=aux_dist;
						Jv[j]=  this->ic[i*Parametres::ndofperelt];
					}
				}
			}


			int m=this->nr;
			int ng=this->nc;


			std::vector<int> permv(m,0);
			for (size_t i = 0; i < m; i++) permv[i]=i;

			std::vector<int> permw(ng,0);
			for (size_t i = 0; i < ng; i++) permw[i]=i;

			SubMatrix<T> C = A.get_submatrix(permv, Jv); // Defining sampling columns


			int n=C.nb_cols();


			// ======================================
			// Selecting k best columns using QR on C
			// ======================================

			int lda=m;
			int info;
			int lwork=-1;

			std::vector<int> Jpivot(n,0);
			std::vector<double> work(n);
			std::vector<double> tau(n);

			HPDDM::Lapack<T>::geqp3(&m,&n,  C.data(), &lda, Jpivot.data(),tau.data(), work.data(),  &lwork, nullptr, &info);

			lwork = (int)std::real(work[0]);
			work.resize(lwork);

			HPDDM::Lapack<T>::geqp3(&m,&n,  C.data(), &lda, Jpivot.data(),tau.data(), work.data(),  &lwork, nullptr, &info);


			std::vector<int> Jf(n,0); // For adaptive CUR_GCS

			// std::cout << "columns Jpivot = " <<  Jpivot << '\n';

			for (int i = 0; i < n; i++) {
					Jf[i] = Jv[Jpivot[i]-1];
			}

			// std::cout << "columns Jv = " <<  Jv << '\n';
			// std::cout << "K Selected columns J = " <<  Jf << '\n';

			// Getting best rows
			C.resize(m,k);

			Matrix<T> Q(k,m);
			for (int i=0; i<k ; i++)
				for (int j=0; j<m; j++)
					if(i==j) Q(i,j) = 1;
					else Q(i,j) = 0;


			// Lapack operations
			// ===============================================
			//	Getting Q1

			lwork = -1;
			// note that we need to change lda from m to k!!
			HPDDM::Lapack<T>::mqr("R", "T", &k, &m, &k,  C.data(), &lda, tau.data(), Q.data(), &k, work.data(), &lwork, &info);


			lwork = (int)std::real(work[0]);
			work.resize(lwork);

			HPDDM::Lapack<T>::mqr("R", "T", &k, &m, &k,  C.data(), &lda, tau.data(), Q.data(), &k, work.data(), &lwork, &info);

			// Lapack operations
			// ===============================================
			//	Getting Rows from a QR on Q_1^T

			lwork = -1;
			std::vector<int> If(m,0);

			HPDDM::Lapack<T>::geqp3(&k,&m,  Q.data(), &k, If.data(),tau.data(), work.data(),  &lwork, nullptr, &info);

			lwork = (int)std::real(work[0]);
			work.resize(lwork);

			HPDDM::Lapack<T>::geqp3(&k,&m,  Q.data(), &k, If.data(),tau.data(), work.data(),  &lwork, nullptr, &info);


			If.resize(n); // For adaptive CUR_GCS
			std::for_each(If.begin(), If.end(), [](int& d) { d-=1;});
			// std::cout << "K Selected rows    I = " <<  If << '\n';
			// std::cout << "===============================================" << '\n';


		// Skeleton Approximation
		// ======================================

		std::vector<double> Gam; // For calculating Mk = \prod_{i=1}^{k} \delta_i
		// Partial pivot
			int J = 0;
			int q = 0;
			int count = 0;

			int reqrank = this->rank;
			std::vector<std::vector<T> > uu, vv;
			std::vector<bool> visited_row(this->nr,false);
			std::vector<bool> visited_col(this->nc,false);

			double frob = 0;
			double aux  = 0;

			int Iaux, Jaux; // save sampling from ACAp for comparisons

			// If run adaptively
			while (((reqrank > 0) && (q < k) ) ||
			       ((reqrank < 0) && (sqrt(aux/frob)>this->epsilon || q==0)))
			{
				// Next current rank
				q+=1;

				if (q*(this->nr+this->nc) > (this->nr*this->nc)) { // the next current rank would not be advantageous
                    q=-1;
										std::cout << "No need compression" << '\n';
					break;
				}
				else{

					std::vector<T> r(this->nc,0),c(this->nr,0);

					// Get row index
					double pivot = 0.;
					I = If[count];

					SubMatrix<T> row = A.get_submatrix(std::vector<int> {I}, permw);

					for(int k=0; k<this->nc; k++){
						r[k] = row(0,k);

						for(int j=0; j<uu.size(); j++){
							r[k] += -uu[j][I]*vv[j][k];
						}

						// To select the maximum element in the row
						if( std::abs(r[k])>pivot && !visited_col[k] ){
							Jaux=k; pivot=std::abs(r[k]);}
					}

					visited_row[I] = true;

					// Get col index
					J = Jf[count];

					// ensure det(Mk) > 0
					// std::cout << "value of r[J], J=" << J << " and " << r[J] << '\n';
					Gam.push_back(r[J]);


					if( std::abs(r[J]) < 1e-15){
						std::cout << "!!! zero pivot chosen" << '\n';
						J = Jaux; }

						T gamma = T(1.)/r[J];

					if( std::abs(r[J]) > 1e-15 ){

						double cmax = 0.;

						SubMatrix<T> col = A.get_submatrix(permv, std::vector<int> {J});

						for(int j=0; j<this->nr; j++){
							c[j] = col(j,0);//A.get_coef(this->ir[j],this->ic[J]);

							for(int k=0; k<uu.size(); k++){
								c[j] += -uu[k][j]*vv[k][J];
							}

							c[j] = gamma*c[j];

							if( std::abs(c[j])>cmax && !visited_row[j] ){
								Iaux =j; cmax=std::abs(c[j]);}
						}


						visited_col[J] = true;


						// ===========================
						// Test if no given rank
						// ===========================

						if (reqrank<0){
							// Error estimator
							T frob_aux = 0.;
							aux = std::abs(dprod(c,c)*dprod(r,r));
							// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
							for(int j=0; j<uu.size(); j++){
								frob_aux += dprod(r,vv[j])*dprod(c,uu[j]);
							}
							// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
							frob += aux + 2*std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
							//==================//
						}


						uu.push_back(c);
						vv.push_back(r);

					}
					else{
						// std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;
						q-=1;
						break;
					}

				count++;

				}


			} // end  while

			// Final rank
			this->rank=q;
			// std::cout << "Approximation rank = " << this->rank << ", for an error tolerance ="<< this->epsilon <<'\n';

			if (this->rank>0){
				this->U.resize(this->nr,this->rank);
				this->V.resize(this->rank,this->nc);

				for (int k=0;k<this->rank;k++){
					this->U.set_col(k,uu[k]);
					this->V.set_row(k,vv[k]);
				}
			}


		} //end global else
	} // end build


}; //end class

} //end namespace
#endif
