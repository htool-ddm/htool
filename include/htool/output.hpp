#ifndef HTOOL_OUTPUT_HPP
#define HTOOL_OUTPUT_HPP
#include <string>
#include <fstream>



namespace htool{

void	Write_gmsh_nodes(const std::vector<int> labels, const std::string& inputname, const std::string& outputname){
	std::ifstream src(inputname.c_str());
	std::ofstream file(outputname);

	file << src.rdbuf();

	//== Nodedata
	file << "$NodeData\n";
	file << 1<<std::endl; // number-of-string-tags
	file <<outputname<<std::endl;
	file <<"1"<<std::endl; // number-of-real-tags
	file <<"0.0"<<std::endl;
	file <<"3"<<std::endl; // number-of-int-tags
	file <<"0"<<std::endl;
	file <<"1"<<std::endl;
	file <<labels.size()<<std::endl;
	for (int j=0;j<labels.size();j++){
		file << j+1 <<" " <<labels[j]<<std::endl;
	}
	file << "$EndNodeData\n";

}
}// namespace

#endif
