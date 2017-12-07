#ifndef HTOOL_SPMATRIX_HPP
#define HTOOL_SPMATRIX_HPP

#include <iostream>
#include <complex>
#include <vector>
#include <cassert>
#include <Eigen/Sparse>
#include <Eigen/Dense>
//#include "point.hpp"

namespace htool {
//================================//
//      DECLARATIONS DE TYPE      //
//================================//

//typedef vector<Cplx>    vectCplx;

//=================================================================//
//                         CLASS SPARSE MATRIX
/******************************************************************//**
* Class for sparse matrices (in coordinate list format).
* Its member objects are:
*   - I: vector of the row indices,
*   - J: vector of the column indices,
*   - K: vector of the (complex) coefficients of the matrix,
*   - nr: the number of rows,
*   - nc: the number of columns.
*********************************************************************/

class SpMatrix{

private:

    std::vector<int> I,J;
    std::vector<Cplx> K;
	int  nr;
	int  nc;

public:

	//! ### Default constructor
	/*!
	 Initializes the matrix to the size 0*0.
  */
	SpMatrix(): nr(0), nc(0){}


	//! ### Another constructor
	/*!
	 Initializes the matrix with _nrp_ rows and _ncp_ columns,
     _Ip_ as vector of the row indices, _Jp_ as vector of the column indices,
     _Kp_ as vector of the coefficients of the matrix.
  */
	SpMatrix(const std::vector<int>& Ip, const std::vector<int>& Jp, std::vector<Cplx>& Kp, const int& nrp, const int& ncp):
    I(Ip), J(Jp), K(Kp), nr(nrp), nc(ncp) {}


	//! ### Copy constructor
	/*!
  */
	SpMatrix(const SpMatrix& A):
	I(A.I), J(A.J), K(A.K), nr(A.nr), nc(A.nc){}


	//! ### Assignement operator with a sparse matrix input argument
	/*!
	 Copies the _I_, _J_, _K_ of the input _A_ argument
	 (which is a sparse matrix) into the vectors of
	 calling instance.
  */
	void operator=(const SpMatrix& A){
		assert( nr==A.nr && nc==A.nc);
        I = A.I; J = A.J; K = A.K;}


	//! ### Matrix-vector product
	/*!
	 The input parameter _u_ is the input vector
	 (i.e. the right operand).
  */
	vectCplx operator*(const vectCplx& u){
        int ncoef = I.size();
        vectCplx v(nr,0.);
        for(int j=0; j<ncoef; j++)
                v[I[j]]+= K[j]*u[J[j]];
        return v;}


	//! ### Matrix-vector product
	/*!
	 Another instanciation of the matrix-vector product
	 that avoids the generation of temporary instance for the
	 output vector. This routine achieves the operation

	 lhs = m*rhs

	 The left and right operands (_lhs_ and _rhs_) are templated
	 and can then be of any type (not necessarily of type vectCplx).
  */
	template <typename LhsType, typename RhsType>
	friend void MvProd(LhsType& lhs, const SpMatrix& m, const RhsType& rhs){
        int ncoef = m.I.size();
        for(int j=0; j<ncoef; j++)
            lhs[m.I[j]]+= m.K[j]*rhs[m.J[j]];
	}


    //! ### Modifies the size of the matrix
    /*!
     Changes the size of the matrix so that
     the number of rows is set to _nbr_ and
     the number of columns is set to _nbc_ and
     the sizes of the 3 member vectors are set to _nbcoef_.
     */
    void resize(const int nbr, const int nbc, const int nbcoef){
        assert(nbcoef<=nbr*nbc);
        nr = nbr; nc = nbc;
        I.resize(nbcoef); J.resize(nbcoef); K.resize(nbcoef);
    }


    //! ### Access to row indices
    /*!
     Returns the _i_th row index of the input argument _A_.
     */
    int& I_(const int i){
        assert(i<I.size());
        return I[i];
    }


    //! ### Access to column indices
    /*!
     Returns the _i_th column index of the input argument _A_.
     */
    int& J_(const int i){
        assert(i<J.size());
        return J[i];
    }


    //! ### Access to coefficients
    /*!
     Returns the _i_th coefficients inside _K_ of the input argument _A_.
     */
    Cplx& K_(const int i){
        assert(i<K.size());
        return K[i];
    }


	//! ### Access to number of rows
	/*!
	 Returns the number of rows of the input argument _A_.
  */
	friend const int& nb_rows(const SpMatrix& A){ return A.nr;}


	//! ### Access to number of columns
	/*!
	 Returns the number of columns of the input argument _A_.
  */
	friend const int& nb_cols(const SpMatrix& A){ return A.nc;}


    //! ### Access to number of non zero coefficients
    /*!
     Returns the number of non zero coefficients of the input argument _A_.
     */
	friend int nb_coeff(const SpMatrix& A){ return A.I.size();}


    //! ### Compute the compression rate
    /*!
     1 - number of non zero coefficients/(nb_rows*nb_columns)
     */
    friend Real CompressionRate(const SpMatrix& A){
        Real comp;
        comp = ((double) A.I.size())/((double)(A.nr*A.nc)); // number of non zero coefficients/(nb_rows*nb_columns)
        return (1-comp);
    }


};

}
#endif
