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
template< typename T, typename ClusterImpl> class MultipartialACA;

template<typename T, typename ClusterImpl>
class bareLowRankMatrix: public LowRankMatrix<T, ClusterImpl>{
private:

    // Friend
    friend class MultipartialACA<T, ClusterImpl>;

public:
    using LowRankMatrix<T,ClusterImpl>::LowRankMatrix;
    
	void build(const IMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs){}
};
}
#endif
