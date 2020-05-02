#ifndef HTOOL_CLUSTERING_GEOMETRIC_HPP
#define HTOOL_CLUSTERING_GEOMETRIC_HPP

#include "../clustering/cluster.hpp"

namespace htool {


template <typename ClusterImpl>
class Block: public Parametres{

private:

	const Cluster<ClusterImpl>* t;
	const Cluster<ClusterImpl>* s;
	int Admissible;

public:
	Block(const Cluster<ClusterImpl>& t0, const Cluster<ClusterImpl>& s0):  t(&t0), s(&s0), Admissible(-1) {};
	Block(const Block& b): t(b.t), s(b.s), Admissible(b.Admissible) {};
	Block& operator=(const Block& b){t=b.t; s=b.s; Admissible=b.Admissible; return *this;}
	const Cluster<ClusterImpl>& tgt_()const {return *(t);}
	const Cluster<ClusterImpl>& src_() const {return *(s);}
	void ComputeAdmissibility() {
		// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
		Admissible =  2*std::min((*t).get_rad(),(*s).get_rad()) < eta* std::max((norm2((*t).get_ctr()-(*s).get_ctr())-(*t).get_rad()-(*s).get_rad() ),0.)  ;
	}
	bool IsAdmissible() const{
		assert(Admissible != -1);
		return Admissible;
	}
	// friend std::ostream& operator<<(std::ostream& os, const Block& b){
	// 	os << "src:\t" << b.src_() << std::endl; os << "tgt:\t" << b.tgt_(); return os;}

};

struct comp_block
{   
    template <typename ClusterImpl>
    inline bool operator() (const Block<ClusterImpl>* block1, const Block<ClusterImpl>* block2)
    {
        if (block1->tgt_().get_offset()==block2->tgt_().get_offset()){
            return block1->src_().get_offset()<block2->src_().get_offset();
        }
        else {
            return block1->tgt_().get_offset()<block2->tgt_().get_offset();
        }
    }
};

}

#endif