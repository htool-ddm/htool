#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include "point.hpp"
#include "blas.hpp"

namespace htool {
//================================//
//      DECLARATIONS DE TYPE      //
//================================//
typedef std::pair<int,int>            Int2;

//================================//
//            VECTEUR             //
//================================//
template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}


template<typename T>
std::vector<T> operator+(const std::vector<T>& a,const std::vector<T>& b){
	assert(a.size()==b.size());
	std::vector<T> result(a.size(),0);
	std::transform (a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

	return result;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& a,const std::vector<T>& b){
	assert(a.size()==b.size());
	std::vector<T> result(a.size(),0);
	std::transform (a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());

	return result;
}

template<typename T, typename V>
std::vector<T> operator*(V value,const std::vector<T>& a)
{
	std::vector<T> result(a.size(),0);
	std::transform (a.begin(), a.end(), result.begin(), std::bind1st(std::multiplies<T>(),value));

	return result;
}

template<typename T, typename V>
std::vector<T> operator*(const std::vector<T>& b,V value)
{
	return value*b;
}

template<typename T, typename V>
std::vector<T> operator/(const std::vector<T>& a,V value)
{
  std::vector<T> result(a.size(),0);
	std::transform (a.begin(), a.end(), result.begin(), std::bind2nd(std::divides<T>(),value));

	return result;
}


template<typename T>
T dprod(const std::vector<T>& a,const std::vector<T>& b){
	return std::inner_product(a.begin(),a.end(),b.begin(),T());
}
template<typename T>
std::complex<T> dprod(const std::vector<std::complex<T> >& a,const std::vector<std::complex<T> >& b){
	return std::inner_product(a.begin(),a.end(),b.begin(),std::complex<T>(),std::plus<std::complex<T> >(), [](std::complex<T>u,std::complex<T>v){return u*std::conj<T>(v);});
}


template<typename T>
double norm2(const std::vector<T>& u){return std::sqrt(std::abs(dprod(u,u)));}

template<typename T>
T max(const std::vector<T>& u){
  return *std::max_element(u.begin(),u.end(),[](T a, T b){return std::abs(a)<std::abs(b);});
}

template<typename T>
int argmax(const std::vector<T>& u){
  return std::max_element(u.begin(),u.end(),[](T a, T b){return std::abs(a)<std::abs(b);})-u.begin();
}

typedef std::vector<Cplx>    vectCplx;
typedef std::vector<double>    vectReal;
typedef std::vector<int>     vectInt;
typedef std::vector<R3>      vectR3;


template<typename T, typename V>
void operator*=(std::vector<T>& a, const V& value){
  std::transform (a.begin(), a.end(), a.begin(), std::bind1st(std::multiplies<T>(),value));
}

template<typename T, typename V>
void operator/=(std::vector<T>& a, const V& value){
  std::transform (a.begin(), a.end(), a.begin(), std::bind2nd(std::divides<T>(),value));
}


double mean(const std::vector<double>& u){
	double res = 0;
	for(int j=0; j<u.size(); j++)
		res += u[j];
	res /= u.size();
	return res;
}

//================================//
//      CLASSE SUBVECTOR          //
//================================//

template <typename VecType>
class SubVec{

private:
	VecType&       U;
	const vectInt& I;
	const int      size;

	typedef typename VecType::value_type ValType;

public:

	SubVec(VecType& U0, const vectInt& I0): U(U0), I(I0), size(I0.size()) {}
	SubVec(const SubVec&); // Pas de constructeur par recopie

	ValType& operator[](const int& k) {return U[I[k]];}
	const ValType& operator[](const int& k) const {return U[I[k]];}

	void operator=(const ValType& v){
		for(int k=0; k<size; k++){ U[I[k]]=v;}}

	template <typename RhsType>
	ValType operator,(const RhsType& rhs) const {
		ValType lhs = 0.;
		for(int k=0; k<size; k++){lhs += U[I[k]]*rhs[k];}
		return lhs;
	}

	friend int size(const SubVec& sv){ return sv.size;}

	friend std::ostream& operator<<(std::ostream& os, const SubVec& u){
		for(int j=0; j<u.size; j++){ os << u[j] << "\t";}
		return os;}

};

typedef SubVec<std::vector<Cplx> > SubVectCplx;
typedef SubVec<const std::vector<Cplx> > ConstSubVectCplx;



//=================================================================//
//                         CLASS MATRIX
//*****************************************************************//
template<typename T>
class IMatrix{

protected:
  IMatrix(){};
  IMatrix(const IMatrix&)             = default; // copy constructor
  IMatrix& operator= (const IMatrix&) = default; // copy assignement operator
  IMatrix(IMatrix&&)                  = default; // move constructor
  IMatrix& operator=(IMatrix&&)       = default; // move assignement operator

	int  nr;
	int  nc;

public:

	// virtual ~IMatrix(){}


  virtual T get_coef(const int& j, const int& k) const =0;

	//! ### Access to number of rows
	/*!
	 Returns the number of rows of the input argument _A_.
  */
	const int& nb_rows() const{ return nr;}


	//! ### Access to number of columns
	/*!
	 Returns the number of columns of the input argument _A_.
  */
	const int& nb_cols() const{ return nc;}


};

template<typename T>
class Matrix: public IMatrix<T>{

protected:

	std::vector<T> mat;


public:

	//! ### Default constructor
	/*!
	 Initializes the matrix to the size 0*0.
  */
	Matrix(){
		this->nr = 0;
		this->nc = 0;
	}


	//! ### Another constructor
	/*!
	 Initializes the matrix with _nbr_ rows and _nbc_ columns,
	 and fills the matrix with zeros.
  */
	Matrix(const int& nbr, const int& nbc){
		this->mat.resize(nbr*nbc,0);
		this->nr = nbr;
		this->nc = nbc;
	}

	//! ### Copy constructor
	/*!
  */
	Matrix(const Matrix& A) = default;

  //! ### Copy assignement operator with matrix input argument
	/*!
	 Copies the value of the entries of the input _A_
	 (which is a matrix) argument into the entries of
	 calling instance.
  */
	void operator=(const Matrix& A){
		assert( this->nr==A.nr && this->nc==A.nc);
		this->mat = A.mat;}

  //! ### Copy assignement operator with scalar input argument
  /*!
   Sets the values of the entries of the calling instance
   to the input value _z_.
  */
  void operator=(const T& z){
    std::fill (this->mat.begin(), this->mat.end(),z);
  }

  //! ### Move constructor
	/*!
	 Initializes the matrix to the size 0*0.
  */
	Matrix(Matrix&&) = default;

  //! ### Copy assignement operator with matrix input argument
	/*!
	 Copies the value of the entries of the input _A_
	 (which is a matrix) argument into the entries of
	 calling instance.
  */
	Matrix& operator=(Matrix&& A){
		assert( this->nr==A.nr && this->nc==A.nc);
		this->mat = std::move(A.mat);
    this->nr  = std::move(A.nr);
    this->nc  = std::move(A.nc);

    return *this;
  }


	//! ### Access operator
	/*!
	 If _A_ is the instance calling the operator
	 _A.get_coef(j,k)_ returns the entry of _A_ located
	 jth row and kth column.
	 */

	T get_coef(const int& j, const int& k) const{
        return this->mat[j+k*this->nr];
  }

  //! ### Access operator
  /*!
   If _A_ is the instance calling the operator
   _A(j,k)_ returns the entry of _A_ located
   jth row and kth column. Modification of the entries
   are allowed.
	 */
    T& operator()(const int& j, const int& k){
      return this->mat[j+k*this->nr];
    }


    //! ### Access operator
    /*!
     If _A_ is the instance calling the operator
     _A(j,k)_ returns the entry of _A_ located
     jth row and kth column. Modification of the
     entries are forbidden.
  */
    const T& operator()(const int& j, const int& k) const {
        return this->mat[j+k*this->nr];
    }

  //! ### Access operator
	/*!
	 If _A_ is the instance calling the operator
	 _A.get_stridedslice(i,j,k)_ returns the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_. Modification forbidden
	 */

   std::vector<T> get_stridedslice( int start, int length, int stride ) const
   {
       std::vector<T> result;
       result.reserve( length );
       const T *pos = &mat[start];
       for( int i = 0; i < length; i++ ) {
           result.push_back(*pos);
           pos += stride;
       }
       return result;
   }

  //! ### Access operator
	/*!
	 If _A_ is the instance calling the operator
	 _A.get_row(j)_ returns the jth row of _A_.
	 */

  std::vector<T> get_row( int row) const{
    return this->get_stridedslice(row,this->nc,this->nr);
  }

  //! ### Access operator
	/*!
	 If _A_ is the instance calling the operator
	 _A.get_col(j)_ returns the jth col of _A_.
	 */

  std::vector<T> get_col( int col) const{
    return this->get_stridedslice(col*this->nr,this->nr,1);
  }

  //! ### Access operator
 /*!
 If _A_ is the instance calling the operator
 _A.set_stridedslice(i,j,k,a)_ puts a in the slice of _A_ containing every element from _start_ to _start_+_lenght with a step of _stride_.
  */

 void set_stridedslice( int start, int length, int stride, const std::vector<T>& a){
   assert(length==a.size());
   T *pos = &mat[start];
   for( int i = 0; i < length; i++ ) {
       *pos=a[i];
       pos += stride;
   }
 }

  //! ### Access operator
  /*!
  If _A_ is the instance calling the operator
  _A.set_row(i,a)_ puts a in the ith row of _A_.
  */
  void set_row( int row, const std::vector<T>& a){
    set_stridedslice(row,this->nc,this->nr,a);
  }

  //! ### Access operator
  /*!
  If _A_ is the instance calling the operator
  _A.set_row(i,a)_ puts a in the row of _A_.
  */
  void set_col( int col, const std::vector<T>& a){
    set_stridedslice(col*this->nr,this->nr,1,a);
  }

	//! ### Modifies the size of the matrix
	/*!
	 Changes the size of the matrix so that
	 the number of rows is set to _nbr_ and
	 the number of columns is set to _nbc_.
  */
	void resize(const int nbr, const int nbc, T value=0){
		this->mat.resize(nbr*nbc, value); this->nr = nbr; this->nc = nbc;}

	//! ### Matrix-scalar product
	/*!
	 */

	friend Matrix operator*(const Matrix& A, const T& a){
		Matrix R = A;
		for (int i=0;i<A.nr;i++){
			for (int j=0;j<A.nc;j++){
				R(i,j)=R(i,j)*a;
			}
		}
		return R;
	}
	friend Matrix operator*(const T& a,const Matrix& A){
		return A*a;
	}

	//! ### Matrix sum
	/*!
	 */

	Matrix operator+(const Matrix& A){
		assert(this->nr==A.nr && this->nc==A.nc);
		Matrix R = A;
		for (int i=0;i<A.nr;i++){
			for (int j=0;j<A.nc;j++){
				R(i,j)=this->mat[i+j*this->nr]+A(i,j);
			}
		}
		return R;
	}

	//! ### Matrix -
	/*!
	 */

	Matrix operator-(const Matrix& A){
		assert(this->nr==A.nr && this->nc==A.nc);
		Matrix R = A;
		for (int i=0;i<A.nr;i++){
			for (int j=0;j<A.nc;j++){
				R(i,j)=this->mat[i+j*this->nr]-A(i,j);
			}
		}
		return R;
	}

  //! ### Matrix-Matrix product
	/*!
  */
	Matrix operator*(const Matrix& A){
		assert(this->nc==A.nr);
		Matrix R(this->nr,A.nc);
		for (int i=0;i<this->nr;i++){
			for (int j=0;j<A.nc;j++){
				for (int k=0;k<A.nr;k++){
					R(i,j)+=this->mat[i+k*this->nr]*A(k,j);
				}
			}
		}
		return R;
	}

	//! ### Matrix-std::vector product
	/*!
  */

	std::vector<T> operator*(const std::vector<T>& rhs){
		int nr = this->nr;
		int nc = this->nc;
		T alpha = 1;
		int lda = nr;
		int incx =1;
		T beta =0;
		int incy = 1;
		char n='N';
    std::vector<T> lhs(nr);
		Blas<T>::gemv(&n, &nr , &nc, &alpha, &(this->mat[0]) , &lda, &rhs[0], &incx, &beta, &lhs[0], &incy);
    return lhs;

 		/*
 		for(int j=0; j<m.nr; j++){
 			for(int k=0; k<m.nc; k++){
 				lhs[j]+= m.mat(j,k)*rhs[k];
 			}
 		}
 		*/
	}

	friend std::ostream& operator<<(std::ostream& out, const Matrix& m){
    if ( !(m.mat.empty()) ) {
      std::cout<< m.nr << " " << m.nc <<std::endl;
      for (int i=0;i<m.nr;i++){
        std::vector<T> row = m.get_row(i);
        std::copy (row.begin(), row.end(), std::ostream_iterator<T>(out, "\t"));
        out << std::endl;
    }
    }
    return out;
  }


	//! ### Looking for the entry of maximal modulus
	/*!
	 Returns the number of row and column of the entry
	 of maximal modulus in the matrix _A_.
  */
	friend std::pair<int,int> argmax(const Matrix<T>& M) {
    int p = argmax(M.mat);
		return std::pair<int,int> (p% M.nr,(int) p/M.nr);
	}


    friend int matrix_to_bytes(const Matrix& A, const std::string& file){

		std::ofstream out(file,std::ios::out | std::ios::binary | std::ios::trunc);

    	if(!out) {
    		std::cout << "Cannot open file.";
    		return 1;
   		}

		/*
    	for(int i = 0; i < A.nr; i++){
        	for(int j = 0; j < A.nc; j++){
            	double cf = real(A(i,j));
				out.write((char *) &cf, sizeof cf);
        	}
    	}
    	*/
    	int rows=nb_rows(A), cols=nb_cols(A);
    	out.write((char*) (&rows), sizeof(int));
    	out.write((char*) (&cols), sizeof(int));
    	out.write((char*) &(A.mat[0]), rows*cols*sizeof(Cplx) );

    	out.close();
    	return 0;
	}

	friend int bytes_to_matrix(const std::string& file, Matrix& A){

		std::ifstream in(file,std::ios::in | std::ios::binary);

    	if(!in) {
    		std::cout << "Cannot open file.";
    		return 1;
   		}

    	int rows=0, cols=0;
    	in.read((char*) (&rows), sizeof(int));
    	in.read((char*) (&cols), sizeof(int));
    	A.resize(rows,cols);
    	in.read( (char *) &(A.mat[0]) , rows*cols*sizeof(Cplx) );

    	in.close();
    	return 0;
	}

	friend double squared_absolute_error (const Matrix& m1, const Matrix& m2){
		assert(nb_rows(m1)==nb_rows(m2) && nb_cols(m1)==nb_cols(m2));
		double err=0;
		for (int j=0;j<m1.nr;j++){
			for (int k=0;k<m1.nc;k++){

				err+=std::pow(std::abs(m1(j,k)-m2(j,k)),2);
			}
		}
		return err;
	}

};

//! ### Computation of the Frobenius norm
/*!
Computes the Frobenius norm of the input matrix _A_.
*/
template<typename T>
double normFrob (const Matrix<T>& A){
    double norm=0;
    for (int j=0;j<A.nb_rows();j++){
        for (int k=0;k<A.nb_cols();k++){
            norm = norm + std::pow(std::abs(A(j,k)),2);
        }
    }
    return sqrt(norm);
}

//================================//
//      CLASSE SOUS-MATRICE       //
//================================//
template<typename T>
class SubMatrix : public Matrix<T>{

	std::vector<int> ir;
	std::vector<int> ic;


public:

  SubMatrix(const IMatrix<T>& mat0, const std::vector<int>& ir0, const std::vector<int>& ic0):
    ir(ir0), ic(ic0) {
    this->nr =  ir0.size();
    this->nc =  ic0.size();
  	this->mat.resize(this->nr, this->nc);
  	for (int i=0; i<this->nr; i++)
  		for (int j=0; j<this->nc; j++)
  			this->mat(i,j) = mat0.get_coef(ir[i], ic[j]);
  }

  SubMatrix(const SubMatrix& m) {
  	this->mat = m.mat;
    ir=m.ir;
    ic=m.ic;
    this->nr=m.nr;
    this->nc=m.nc;
  }

  const vectInt& ir_(){ return ir;}

  const vectInt& ic_(){ return ic;}

};
} // namespace


#endif
