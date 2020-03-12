#ifndef HTOOL_MultiMatrix_HPP
#define HTOOL_MultiMatrix_HPP

#include <vector>

namespace htool {

template<typename T>
class MultiIMatrix{
protected:
    // Data members
    int  nr;
    int  nc;
    int nb_matrix;

    // Constructors and cie
    MultiIMatrix()                           = delete;  // no default constructor


    MultiIMatrix(int nr0,int nc0): nr(nr0), nc(nc0){}

public:

    MultiIMatrix(MultiIMatrix&&)                  = default; // move constructor
    MultiIMatrix& operator=(MultiIMatrix&&)       = default; // move assignement operator
    MultiIMatrix(const MultiIMatrix&)             = default; // copy constructor
    MultiIMatrix& operator= (const MultiIMatrix&) = default; // copy assignement operator

    virtual std::vector<T> get_coef(const int& j, const int& k) const =0;

    // // TODO: improve interface
    // virtual SubMatrix<T> get_submatrix(const std::vector<int>& J, const std::vector<int>& K) const
    // {
    //     // std::cout << "coucou" << std::endl;
    //     SubMatrix<T> mat(J,K);
    //     for (int i=0; i<mat.nb_rows(); i++)
    //     for (int j=0; j<mat.nb_cols(); j++)
    //     mat(i,j) = this->get_coef(J[i], K[j]);
    //     return mat;
    // }


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
    const int& nb_matrix() const{ return nb_matrix;}

    virtual ~MultiIMatrix() {};
};

} //namespace

#endif