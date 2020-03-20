#ifndef HTOOL_BARELRMAT_HPP
#define HTOOL_BARELRMAT_HPP

#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>
#include "lrmat.hpp"
#include "../multilrmat/multilrmat.hpp"



// To be used with multilrmat
namespace htool {


// Forward declaration
template< typename T > class MultipartialACA;

template<typename T>
class bareLowRankMatrix: public LowRankMatrix<T>{
private:

    // Friend
    friend class MultipartialACA<T>;

public:
    using LowRankMatrix<T>::LowRankMatrix;
    
	void build(const IMatrix<T>& A, const Cluster& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster& s, const std::vector<R3>& xs, const std::vector<int>& tabs){}
};
}
#endif
