#ifndef HTOOL_MATRIX_HPP
#define HTOOL_MATRIX_HPP

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <iterator>
#include "../wrappers/wrapper_blas.hpp"
#include "vector.hpp"

namespace htool {


//=================================================================//
//                         CLASS MATRIX
//*****************************************************************//
template<typename T>
class Matrix;

template<typename T>
class SubMatrix;

template<typename T>
class IMatrix{
protected:
    // Data members
    int  nr;
    int  nc;

    // Constructors and cie
    IMatrix()                           = delete;  // no default constructor


    IMatrix(int nr0,int nc0): nr(nr0), nc(nc0){}

public:

    IMatrix(IMatrix&&)                  = default; // move constructor
    IMatrix& operator=(IMatrix&&)       = default; // move assignement operator
    IMatrix(const IMatrix&)             = default; // copy constructor
    IMatrix& operator= (const IMatrix&) = default; // copy assignement operator

    virtual T get_coef(const int& j, const int& k) const =0;

    // TODO: improve interface
    virtual SubMatrix<T> get_submatrix(const std::vector<int>& J, const std::vector<int>& K) const
    {
        // std::cout << "coucou" << std::endl;
        SubMatrix<T> mat(J,K);
        for (int i=0; i<mat.nb_rows(); i++)
        for (int j=0; j<mat.nb_cols(); j++)
        mat(i,j) = this->get_coef(J[i], K[j]);
        return mat;
    }


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

    virtual ~IMatrix() {};
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
    Matrix():IMatrix<T>(0,0){}


    //! ### Another constructor
    /*!
    Initializes the matrix with _nbr_ rows and _nbc_ columns,
    and fills the matrix with zeros.
    */
    Matrix(const int& nbr, const int& nbc): IMatrix<T>(nbr,nbc){
        this->mat.resize(nbr*nbc,0);
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
        this->mat = A.mat;
    }

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
    // SubMatrix<T> get_submatrix(const std::vector<int>& J, const std::vector<int>& K) const
    // {
    //   SubMatrix<T> mat(J,K);
    // 	for (int i=0; i<mat.nb_rows(); i++)
    // 		for (int j=0; j<mat.nb_cols(); j++)
    // 			mat(i,j) = this->get_coef(J[i], K[j]);
    //   return mat;
    // }

    //! ### Access operator
    /*!
    If _A_ is the instance calling the operator
    _A.get_coef(j,k)_ returns the entry of _A_ located
    jth row and kth column.
    */

    // Matrix<T> get_submatrix(const std::vector<int>& J, const std::vector<int>& K) const{
    //       return SubMatrix<T>(*this,J,K) ;
    // }

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
    */

    const std::vector<T>& get_mat(){return this->mat;}


    //! ### Access operator
    /*!
    */

    T *  data() {return this->mat.data();}

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
        this->mat.resize(nbr*nbc, value); this->nr = nbr; this->nc = nbc;
    }

    //! ### Matrix-scalar product
    /*!
    */

    friend Matrix operator*(const Matrix& A, const T& a){
        Matrix R(A.nr,A.nc);
        for (int i=0;i<A.nr;i++){
            for (int j=0;j<A.nc;j++){
                R(i,j)=A(i,j)*a;
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
        Matrix R(A.nr,A.nc);
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
        Matrix R(A.nr,A.nc);
        for (int i=0;i<A.nr;i++){
            for (int j=0;j<A.nc;j++){
                R(i,j)=this->mat[i+j*this->nr]-A(i,j);
            }
        }
        return R;
    }

    //! ### Matrix-std::vector product
    /*!
    */

    std::vector<T> operator*(const std::vector<T>& rhs) const{
        std::vector<T> lhs(this->nr);
        this->mvprod(rhs.data(),lhs.data(),1);
        return lhs;
    }

    //! ### Matrix-Matrix product
    /*!
    */
    Matrix operator*(const Matrix& B) const{
        assert(this->nc==B.nr);
        Matrix R(this->nr,B.nc);
        this->mvprod(&(B.mat[0]),&(R.mat[0]),B.nc);
        return R;
    }

    //! ### Interface with blas gemm
    /*!
    */
    void mvprod(const T* const in, T* const out, const int& mu=1) const{
        int nr = this->nr;
        int nc = this->nc;
        T alpha = 1;
        T beta =0;
        int lda =  nr;

        if (mu==1){
            char n='N';
            int incx =1;
            int incy = 1;
            Blas<T>::gemv(&n, &nr , &nc, &alpha, &(this->mat[0]) , &lda, in, &incx, &beta, out, &incy);
        }
        else{
            char transa ='N';
            char transb ='N';
            int M = nr;
            int N = mu;
            int K = nc;
            int ldb = nc;
            int ldc = nr;
            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, &(this->mat[0]),
            &lda, in , &ldb, &beta, out,&ldc);
        }
    }

    //! ### Special mvprod
    /*!
    */
    void mvprod_row_major(const T* const in, T* const out, const int& mu, char op = 'N') const{
        int nr = this->nr;
        int nc = this->nc;
        T alpha = 1;
        T beta =0;
        int lda =  nr;

        if (mu==1){
            int incx =1;
            int incy = 1;
            Blas<T>::gemv(&op, &nr , &nc, &alpha, &(this->mat[0]) , &lda, in, &incx, &beta, out, &incy);
        }
        else{
            int lda =  mu;
            char transa ='N';
            char transb ='T';
            int M = mu;
            int N = nr;
            int K = nc;
            int ldb =  nr;
            int ldc = mu;

            if (op=='C'){
                transb='N';
                N=nc;
                K=nr;
            }

            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in,
                &lda, &(this->mat[0]), &ldb, &beta, out,&ldc);
            }
    }

    //! ### Special add_mvprod
    /*!
    */
    void add_mvprod_row_major(const T* const in, T* const out, const int& mu, char op='N') const{
        int nr = this->nr;
        int nc = this->nc;
        T alpha = 1;
        T beta =1;


        if (mu==1){
            int lda =  nr;
            int incx =1;
            int incy = 1;
            Blas<T>::gemv(&op, &nr , &nc, &alpha, &(this->mat[0]) , &lda, in, &incx, &beta, out, &incy);
        }
        else{
            int lda =  mu;
            char transa ='N';
            char transb ='T';
            int M = mu;
            int N = nr;
            int K = nc;
            int ldb =  nr;
            int ldc = mu;

            if (op=='C'){
                transb='N';
                N=nc;
                K=nr;
            }


            Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, in, &lda, &(this->mat[0]), &ldb, &beta, out,&ldc);
        }
    }

    void add_mvprod_row_major_sym(const T* const in, T* const out, const int& mu) const{
        int nr = this->nr;
        int nc = this->nc;
        T alpha = 1;
        T beta =1;
        char UPLO = 'L';


        if (mu==1){
            int lda =  nr;
            int incx =1;
            int incy = 1;
            Blas<T>::symv(&UPLO, &nr, &alpha, &(this->mat[0]), &lda,in, &incx, &beta, out, &incy);
        }
        else{
            int lda =  nr;
            char side = 'R';
            int M = mu;
            int N = nr;
            int ldb =  mu;
            int ldc = mu;

            Blas<T>::symm(&side, &UPLO, &M, &N, &alpha, &(this->mat[0]),&lda, in, &ldb, &beta, out,&ldc);

        }
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

    //! ### Looking for the entry of maximal modulus
    /*!
    Save a Matrix in a file (bytes)
    */
    int matrix_to_bytes(const std::string& file){

        std::ofstream out(file,std::ios::out | std::ios::binary | std::ios::trunc);

        if(!out) {
            std::cout << "Cannot open file."<<std::endl;
            return 1;
        }
        int rows = this->nr;
        int cols = this->nc;
        out.write((char*) (&rows), sizeof(int));
        out.write((char*) (&cols), sizeof(int));
        out.write((char*) &(mat[0]), rows*cols*sizeof(T) );

        out.close();
        return 0;
    }

    //! ### Looking for the entry of maximal modulus
    /*!
    Load a matrix from a file (bytes)
    */
    int bytes_to_matrix(const std::string& file){

        std::ifstream in(file,std::ios::in | std::ios::binary);

        if(!in) {
            std::cout << "Cannot open file."<<std::endl;
            return 1;
        }

        int rows=0, cols=0;
        in.read((char*) (&rows), sizeof(int));
        in.read((char*) (&cols), sizeof(int));
        mat.resize(rows*cols);
        this->nr=rows;
        this->nc=cols;
        in.read( (char *) &(mat[0]) , rows*cols*sizeof(T) );

        in.close();
        return 0;
    }


    int raw_save(const std::string& file){
        std::ofstream out(file);

        if(!out) {
            std::cout << "Cannot open file."<<std::endl;
            return 1;
        }
        int rows = this->nr;
        int cols = this->nc;
        out<<rows<<" "<<cols<<std::endl;
        for (int i=0;i<rows;i++){
            std::vector<T> row = this->get_row(i);
            std::copy (row.begin(), row.end(), std::ostream_iterator<T>(out, "\t"));
            out << std::endl;
        }
        out.close();
        return 0;
    }

    // To be used with dlmread
    int matlab_save(const std::string& file){
        std::ofstream out(file);
        out << std::setprecision(18);
        if(!out) {
            std::cout << "Cannot open file."<<std::endl;
            return 1;
        }
        int rows = this->nr;
        int cols = this->nc;
        // out<<rows<<" "<<cols<<std::endl;
        for (int i=0;i<rows;i++){
            std::vector<T> row = this->get_row(i);
            for (int j=0;j<cols;j++){
                out<<std::real(row[j]);
                if (std::imag(row[j])<0){
                    out<<std::imag(row[j])<<"i\t";
                }
                else if (std::imag(row[j])==0){
                    out<<"+"<<0<<"i\t";
                }
                else{
                    out<<"+"<<std::imag(row[j])<<"i\t";
                }
            }
            out << std::endl;
        }
        out.close();
        return 0;
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
    // TODO: remove ir and ic
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;

public:
    SubMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0) : Matrix<T>(ir0.size(),ic0.size()), ir(ir0), ic(ic0),offset_i(0), offset_j(0) {}

    SubMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, const int& offset_i0, const int& offset_j0) : Matrix<T>(ir0.size(),ic0.size()), ir(ir0), ic(ic0),offset_i(offset_i0), offset_j(offset_j0) {}

    SubMatrix(const IMatrix<T>& mat0, const std::vector<int>& ir0, const std::vector<int>& ic0): Matrix<T>(ir0.size(),ic0.size()), ir(ir0), ic(ic0), offset_i(0),offset_j(0) {

        // Matrix<T> test ;
        // std::cout << (&test)->mat << std::endl;
        *this = mat0.get_submatrix(ir0,ic0);
    }

    SubMatrix( const IMatrix<T>& mat0, const std::vector<int>& ir0, const std::vector<int>& ic0, const int& offset_i0, const int& offset_j0): Matrix<T>(ir0.size(),ic0.size()), ir(ir0), ic(ic0) {

        *this = mat0.get_submatrix(ir0,ic0);
        offset_i=offset_i0;
        offset_j=offset_j0;
    }

    SubMatrix(const SubMatrix& m) {
        this->mat = m.mat;
        this->ir=m.ir;
        this->ic=m.ic;
        this->offset_i=m.offset_i;
        this->offset_j=m.offset_j;
        this->nr=m.nr;
        this->nc=m.nc;
    }

    // Mostly same operators as in Matrix, need CRTP to factorize
    // Operators
    SubMatrix operator+(const SubMatrix& A){
        assert(this->nr==A.nr && this->nc==A.nc);
        SubMatrix R(*this);
        for (int i=0;i<A.nr;i++){
            for (int j=0;j<A.nc;j++){
                R(i,j)+=A(i,j);
            }
        }
        return R;
    }
    friend SubMatrix operator*(const SubMatrix& A, const T& a){
        SubMatrix R(A);
        for (int i=0;i<A.nr;i++){
            for (int j=0;j<A.nc;j++){
                R(i,j)=A(i,j)*a;
            }
        }
        return R;
    }
    friend SubMatrix operator*(const T& a,const SubMatrix& A){
        return A*a;
    }
    // Getters
    std::vector<int> get_ir() const{ return this->ir;}
    std::vector<int> get_ic() const{ return this->ic;}
    int get_offset_i() const{ return this->offset_i;}
    int get_offset_j() const{ return this->offset_j;}
    void set_offset_i(int offset) {  this->offset_i=offset;}
    void set_offset_j(int offset) {  this->offset_j=offset;}

};
} // namespace


#endif
