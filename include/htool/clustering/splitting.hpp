#ifndef HTOOL_CLUSTERING_SPLITTING_HPP
#define HTOOL_CLUSTERING_SPLITTING_HPP

#include "cluster.hpp"

namespace htool {


enum class SplittingTypes {GeometricSplitting, RegularSplitting};


template<typename ClusterImpl> 
std::vector<std::vector<int>> geometric_splitting(const std::vector<R3>& x, const std::vector<int>& tab, std::vector<int>& num,  Cluster<ClusterImpl> const * const curr_cluster, int nb_sons, R3 dir){
	std::vector<std::vector<int>> numbering(nb_sons);

	// Geometry of current cluster
	int nb_pt = curr_cluster->get_size();
	R3 xc = curr_cluster->get_ctr();

	// For 2 sons, we can use the center of the cluster
	if (nb_sons==2){
		for(int j=0; j<nb_pt; j++){
			R3 dx = x[tab[num[j]]] - xc;

			if( (dir,dx)>0 ){
				numbering[0].push_back(num[j]);
			}
			else{
				numbering[1].push_back(num[j]);
			}
		}

	}
	// Otherwise we have to something more
	else if (num.size()>1){
		const auto minmax  = std::minmax_element(num.begin(),num.end(),[&](int a, int b){ return (x[tab[a]] - xc,dir)<(x[tab[b]] - xc,dir)  ;});
		R3 min = x[tab[*(minmax.first)]];
		R3 max = x[tab[*(minmax.second)]];
		double length = (max-min,dir)/(double)nb_sons;
		for(int j=0; j<nb_pt; j++){
			R3 dx = x[tab[num[j]]] ;
			
			int index = ((dx-min,dir))/length;
			index = (index==nb_sons) ? index-1 : index; // for max
			numbering[index].push_back(num[j]);
		}
	}

	return numbering;
}

template<typename ClusterImpl> 
std::vector<std::vector<int>> regular_splitting(const std::vector<R3>& x, const std::vector<int>& tab, std::vector<int>& num,  Cluster<ClusterImpl> const * const curr_cluster, int nb_sons, R3 dir){

	std::vector<std::vector<int>> numbering(nb_sons);
	R3 xc = curr_cluster->get_ctr();

	std::sort(num.begin(),num.end(),[&](int a, int b){return (x[tab[a]] - xc,dir)<(x[tab[b]] - xc,dir);});

	int size_numbering = num.size()/nb_sons;
	int count_size = 0;
	for (int p=0;p<nb_sons-1;p++){
		numbering[p].resize(size_numbering);
		std::copy_n(num.begin()+count_size,size_numbering,numbering[p].begin());
		count_size+=size_numbering;
	}

	numbering.back().resize(num.size()-count_size);
	std::copy(num.begin()+count_size,num.end(),numbering.back().begin());

	return numbering;
}


}


#endif