#ifndef HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP
#define HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP

#include "blocks.hpp"

namespace htool {


// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
template<typename ClusterImpl>
struct RjasanowSteinbach{
    static bool ComputeAdmissibility(const Cluster<ClusterImpl>& target, const Cluster<ClusterImpl>& source, double eta){
        bool admissible =  2*std::min(target.get_rad(),source.get_rad()) < eta* std::max((norm2(target.get_ctr()-source.get_ctr())-target.get_rad()-source.get_rad() ),0.)  ;
        return admissible;
    }
};


}
#endif