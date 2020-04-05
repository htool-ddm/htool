#ifndef HTOOL_MultiMatrix_HPP
#define HTOOL_MultiMatrix_HPP

#include <vector>

namespace htool {

template<typename T>
class MultiSubMatrix;

template<typename T>
class MultiIMatrix{
protected:
    // Data members
    int  nr;
    int  nc;
    int  nm;

    // Constructors and cie
    MultiIMatrix()                           = delete;  // no default constructor


    MultiIMatrix(int nr0,int nc0, int nm0): nr(nr0), nc(nc0), nm(nm0){}

public:

    MultiIMatrix(MultiIMatrix&&)                  = default; // move constructor
    MultiIMatrix& operator=(MultiIMatrix&&)       = default; // move assignement operator
    MultiIMatrix(const MultiIMatrix&)             = default; // copy constructor
    MultiIMatrix& operator= (const MultiIMatrix&) = default; // copy assignement operator

    virtual std::vector<T> get_coefs(const int& j, const int& k) const =0;

    // TODO: improve interface
    virtual MultiSubMatrix<T> get_submatrices(const std::vector<int>& J, const std::vector<int>& K) const
    {
        
        MultiSubMatrix<T> mat(J,K,nm);
        std::vector<T> coefs(nm);
        for (int i=0; i<mat.nb_rows(); i++){
            for (int j=0; j<mat.nb_cols(); j++){
                coefs=this->get_coefs(J[i], K[j]);
                for (int l=0; l<this->nm; l++){
                    mat[l](i,j) = coefs[l];
                }
            }
        }

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

    //! ### Access to number of columns
    /*!
    Returns the number of matrices.
    */
    const int& nb_matrix() const{ return nm;}

    virtual ~MultiIMatrix() {};
};


// template<typename T>
// class MultiMatrix: public MultiIMatrix<T>{
// private:

//     std::vector<Matrix<T>> Matrices;

// public:

//     MultiMatrix(const int& nbr, const int& nbc, const int& nm): MultiIMatrix<T>(nbr,nbc,nm){
//         for (int l=0;l<nm;l++){
//             this->Matrices.emplace_back(nbr,nbc);
//         }
//     }

//     std::vector<T> get_coefs(const int& j, const int& k) const{
//         std::vector<T> coefs(this->nm,0);
//         for (int l=0;l<this->nm;l++){
//             coefs[l]=this->Matrices[l][j+k*this->nr];
//         }
//         return coefs;
//     }
    
    
//     Matrix<T>&  operator[](int j){return Matrices[j];}; 
//     const Matrix<T>&  operator[](int j) const {return Matrices[j];}; 


// };


template<typename T>
class MultiSubMatrix : public MultiIMatrix<T>{
private:

    std::vector<SubMatrix<T>> SubMatrices;
    // TODO: remove ir and ic
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;

public:

    MultiSubMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, int nm) : MultiIMatrix<T>(ir0.size(),ic0.size(),nm), ir(ir0), ic(ic0),offset_i(0), offset_j(0),SubMatrices(nm,SubMatrix<T>(ir0,ic0)) {
    }

   MultiSubMatrix(const MultiIMatrix<T>& mat0, const std::vector<int>& ir0, const std::vector<int>& ic0): MultiIMatrix<T>(ir0.size(),ic0.size(),mat0.nb_matrix()), ir(ir0), ic(ic0), offset_i(0),offset_j(0),SubMatrices(mat0.nb_matrix(),SubMatrix<T>(ir0,ic0)) {
        *this = mat0.get_submatrices(ir0,ic0);
    }

    MultiSubMatrix( const MultiIMatrix<T>& mat0, const std::vector<int>& ir0, const std::vector<int>& ic0, const int& offset_i0, const int& offset_j0): MultiIMatrix<T>(ir0.size(),ic0.size(),mat0.nb_matrix()), ir(ir0), ic(ic0),offset_i(offset_i0), offset_j(offset_j0),SubMatrices(mat0.nb_matrix(),SubMatrix<T>(ir0,ic0,offset_i0,offset_j0)) {

        *this = mat0.get_submatrices(ir0,ic0);
        offset_i=offset_i0;
        offset_j=offset_j0;
        for (int l=0;l<mat0.nb_matrix();l++){
            SubMatrices[l].set_offset_i(offset_i);
            SubMatrices[l].set_offset_j(offset_j);
        }
    }

    std::vector<T> get_coefs(const int& j, const int& k) const{
        std::vector<T> coefs(this->nm,0);
        for (int l=0;l<this->nm;l++){
            coefs[l]=this->SubMatrices[l](j,k);
        }
        return coefs;
    }


    SubMatrix<T>&  operator[](int j){return SubMatrices[j];}; 
    const SubMatrix<T>&  operator[](int j) const {return SubMatrices[j];}; 

};
} //namespace

#endif